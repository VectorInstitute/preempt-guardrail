"""Replace names, money, and ages in text with fake values, and undo it.

- Names: format-preserving encryption (FPE, reversible) or random names.
- Money: FPE (reversible) or m-LDP noise.
- Ages: m-LDP noise.

``decrypt`` uses state saved by the last ``encrypt`` call, so call both on the
same ``Sanitizer`` instance.

Known limitation: values are matched as plain substrings ("Ann" also hits
"Anna"), and some decrypt paths restore only the first match. See
https://github.com/VectorInstitute/preempt-guardrail/issues/11.
"""

import os
import random
import re
import unicodedata
from typing import Any, Optional

import names
import numpy as np
from pyfpe_ff3 import FF3Cipher

from preempt.utils import check, load_data, make_names_dataset


# A name is encrypted as the 3-digit index of each word in its alphabet.
DIGITS_PER_NAME = 3
NO_NAME = "999"
ALPHABET_SIZE = 1000
# FF3 needs at least 6 digits, so small amounts are padded with this prefix.
MONEY_FPE_PREFIX = "999999"


class Sanitizer:
    """Encrypt and decrypt names, money, and ages detected by ``ner_model``."""

    def __init__(
        self,
        ner_model: Any,
        key: str = "EF4359D8D580AA4F7F036D6F04FC6A94",
        tweak: str = "D8E7920AFA330A73",
        seed: Optional[int] = None,
    ):
        self.cipher_fn = FF3Cipher(key, tweak, allow_small_domain=True, radix=10)
        self.ner = ner_model
        # Set by encrypt(), read by decrypt().
        self.entity_mapping: list[dict[str, Optional[str]]] = []  # {original: fake} per text
        self.entity_lookup: list[list[str]] = []  # [first_names, last_names] used for name FPE
        self._names_dataset: Optional[dict[str, list[str]]] = None
        self.reseed(seed)

    def reseed(self, seed: Optional[int]) -> None:
        """Reset the RNGs used by m-LDP noise and random names."""
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)

    # --- Public API -----------------------------------------------------------

    def encrypt(
        self, inputs: list[str], epsilon: float = 0.1, entity: str = "Name", use_mdp: bool = False, use_fpe: bool = True
    ) -> tuple[list[str], list[int]]:
        """Replace every detected ``entity`` ("Name", "Money", or "Age") in ``inputs``.

        Returns the sanitized texts and the indexes of texts where a value could
        not be encrypted.
        """
        extracted = self.ner.extract(inputs, entity_type=entity)[entity]
        if entity == "Name":
            self.entity_mapping, self.entity_lookup = self.encrypt_names(extracted, use_fpe=use_fpe)
        elif entity == "Money":
            if not (use_fpe or use_mdp):
                raise ValueError("Money needs use_fpe or use_mdp to be True.")
            self.entity_mapping = self.encrypt_money(extracted, use_fpe=use_fpe, epsilon=epsilon)
        elif entity == "Age":
            self.entity_mapping = self.encrypt_age(extracted, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown entity: {entity!r}")

        sanitized, invalid = [], []
        for i, (line, fakes) in enumerate(zip(inputs, self.entity_mapping)):
            replaced = line
            for original, fake in fakes.items():
                if fake is None:
                    invalid.append(i)
                    break
                replaced = _nfc(replaced).replace(_nfc(original), fake)
            sanitized.append(replaced)
        return sanitized, invalid

    def decrypt(  # noqa: PLR0917
        self,
        inputs: list[str],
        entity: str = "Name",
        extracted: Optional[list[list[str]]] = None,
        use_mdp: bool = False,  # noqa: ARG002 -- kept for API compatibility; m-LDP is used when not use_fpe
        use_fpe: bool = True,
        use_cache: bool = False,
    ) -> list[str]:
        """Restore the original values in texts produced by ``encrypt``.

        Fake values to undo come from ``extracted`` (detected in the sanitized
        texts; NER is run if omitted), or from the last ``encrypt`` call when
        ``use_cache`` is True. Ages always use the last ``encrypt`` call.
        """
        if extracted is None:
            extracted = self.ner.extract(inputs, entity_type=entity)[entity]
        if entity == "Name":
            return self.decrypt_names(inputs, extracted, use_cache=use_cache)
        if entity == "Money":
            return self.decrypt_money(inputs, extracted, use_fpe=use_fpe, use_cache=use_cache)
        if entity == "Age":
            return self.decrypt_age(inputs)
        raise ValueError(f"Unknown entity: {entity!r}")

    # --- Names ----------------------------------------------------------------

    def encrypt_names(
        self, inputs: list[list[str]], use_fpe: bool = True
    ) -> tuple[list[dict[str, Optional[str]]], list[list[str]]]:
        """Map each name to a fake name, per text.

        With FPE, the words of full names are also encrypted on their own, so a
        later "Sarah" is replaced after "Sarah Lopez". Without FPE, fake names
        are random and can't be decrypted.

        Returns ``{original: fake}`` per text and the ``[first_names, last_names]``
        alphabets that ``decrypt_names`` needs.
        """
        if not use_fpe:
            mappings: list[dict[str, Optional[str]]] = []
            for text_names in inputs:
                fakes: dict[str, Optional[str]] = {}
                for name in text_names:
                    fakes.setdefault(name, names.get_full_name(gender="male"))
                mappings.append(fakes)
            return mappings, [[], []]

        first_names, last_names = self._name_alphabets(inputs)
        mappings = []
        for text_names in inputs:
            fakes = {}
            first_words, last_words = _name_words(text_names)
            for name in text_names + first_words + last_words:
                if name not in fakes:
                    fakes[name] = self._encrypt_name(name, first_names, last_names)
            mappings.append(fakes)
        return mappings, [first_names, last_names]

    def decrypt_names(
        self, inputs: list[str], extracted: Optional[list[list[str]]], use_cache: bool = False
    ) -> list[str]:
        """Replace fake names in ``inputs`` with the originals."""
        first_names, last_names = self.entity_lookup
        decrypted = []
        for i, line in enumerate(inputs):
            restored = line
            for fake in self._decrypt_targets(i, extracted, use_cache):
                try:
                    original = self._decrypt_name(fake, first_names, last_names)
                except Exception as err:
                    raise ValueError(f"Decrypt target {fake!r} not found in names dataset!") from err
                restored = restored.replace(fake, original)
            decrypted.append(restored)
        return decrypted

    def _encrypt_name(self, name: str, first_names: list[str], last_names: list[str]) -> str:
        """FPE-encrypt one name into a name from the same alphabets."""
        if " " in name:
            words = name.split()
            digits = _index(first_names, words[0]) + "".join(_index(last_names, w) for w in words[1:])
            return _digits_to_name(self.fpe_encrypt(digits), first_names, last_names)
        try:
            digits = _index(first_names, name) + NO_NAME
            return _digits_to_name(self.fpe_encrypt(digits), first_names, last_names)
        except ValueError:
            # Not a first name, so encrypt it as a last name.
            digits = NO_NAME + _index(last_names, name)
            return _digits_to_name(self.fpe_encrypt(digits), last_names, last_names)

    def _decrypt_name(self, name: str, first_names: list[str], last_names: list[str]) -> str:
        """Undo ``_encrypt_name``."""
        if " " in name and name not in first_names:
            words = name.split()
            digits = _index(first_names, words[0]) + "".join(_index(last_names, w) for w in words[1:])
            return _digits_to_name(self.fpe_decrypt(digits), first_names, last_names, skip_no_name=True)
        digits = _index(first_names, name) + NO_NAME
        return _digits_to_name(self.fpe_decrypt(digits), first_names, last_names)

    def _name_alphabets(self, inputs: list[list[str]]) -> tuple[list[str], list[str]]:
        """Build the first- and last-name lists that FPE indexes into.

        Every input name must be in its list, so the inputs are moved to the end
        of the dataset names. The lists therefore depend on the inputs, which is
        why ``decrypt_names`` reuses the ones saved by ``encrypt``.
        """
        first_words, last_words = _name_words([name for text_names in inputs for name in text_names])
        dataset = self._load_names_dataset()
        return _make_alphabet(dataset["first_names"], first_words), _make_alphabet(dataset["last_names"], last_words)

    def _load_names_dataset(self) -> dict[str, list[str]]:
        """Load the dataset of first and last names once, generating it if missing."""
        if self._names_dataset is not None:
            return self._names_dataset
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "names_dataset.json"),
            "preempt/names_dataset.json",
            f"{os.getcwd()}/names_dataset.json",
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            path = candidates[-1]
            print("Names dataset not found. Generating...")
            make_names_dataset(path)
            print(f"Saved at {path}")
        self._names_dataset = load_data(path)
        return self._names_dataset

    # --- Money ----------------------------------------------------------------

    def encrypt_money(
        self, inputs: list[list[str]], use_fpe: bool = True, epsilon: Optional[float] = None
    ) -> list[dict[str, Optional[str]]]:
        """Map each amount to an FPE-encrypted or m-LDP-noised amount, per text.

        Amounts that can't be parsed map to None.
        """
        mappings = []
        for amounts in inputs:
            fakes: dict[str, Optional[str]] = {}
            for amount in amounts:
                if amount in fakes:
                    continue
                digits = amount.replace(",", "")
                try:
                    if use_fpe:
                        fakes[amount] = self.fpe_encrypt(MONEY_FPE_PREFIX + digits)
                    else:
                        value = int(float(digits))
                        fakes[amount] = str(self.m_epsilon(value, *_money_bounds(value), epsilon))
                except (ValueError, TypeError):
                    fakes[amount] = None
            mappings.append(fakes)
        return mappings

    def decrypt_money(
        self, inputs: list[str], extracted: Optional[list[list[str]]], use_fpe: bool = True, use_cache: bool = False
    ) -> list[str]:
        """Replace fake amounts in ``inputs`` with the originals."""
        decrypted = []
        for i, line in enumerate(inputs):
            restored = line
            fake_to_original = {fake: original for original, fake in self.entity_mapping[i].items()}
            for fake in self._decrypt_targets(i, extracted, use_cache):
                if not use_fpe:
                    restored = _replace_first_word(restored, fake, fake_to_original[fake])
                elif len(fake) >= len(MONEY_FPE_PREFIX):
                    original = self.fpe_decrypt(fake)[len(MONEY_FPE_PREFIX) :]
                    if fake in restored:
                        restored = restored.replace(fake, original)
                    else:  # e.g. "1.250" was written as "1,250"
                        restored = restored.replace(fake.replace(".", ","), original.replace(".", ","))
            decrypted.append(restored)
        return decrypted

    # --- Age ------------------------------------------------------------------

    def encrypt_age(self, inputs: list[list[str]], epsilon: Optional[float] = None) -> list[dict[str, Optional[str]]]:
        """Map each age to an m-LDP-noised age in [10, 99], per text.

        Ages that can't be parsed map to None.
        """
        mappings = []
        for ages in inputs:
            fakes: dict[str, Optional[str]] = {}
            for age in ages:
                if age in fakes:
                    continue
                try:
                    fakes[age] = str(self.m_epsilon(int(age), 10, 99, epsilon))
                except (ValueError, TypeError):
                    fakes[age] = None
            mappings.append(fakes)
        return mappings

    def decrypt_age(self, inputs: list[str]) -> list[str]:
        """Restore ages saved by the last ``encrypt`` call."""
        decrypted = []
        for line, fakes in zip(inputs, self.entity_mapping):
            restored = line
            for original, fake in fakes.items():
                if fake is not None:
                    restored = _replace_first_word(restored, fake, original)
            decrypted.append(restored)
        return decrypted

    # --- Building blocks ------------------------------------------------------

    def fpe_encrypt(self, value: str) -> str:
        """FPE-encrypt the digits of ``value``, keeping its punctuation."""
        return _restore_format(self.cipher_fn.encrypt(_digits_only(value)), str(value))

    def fpe_decrypt(self, value: str) -> str:
        """Undo ``fpe_encrypt``."""
        return _restore_format(self.cipher_fn.decrypt(_digits_only(value)), str(value))

    def m_epsilon(self, x: int, n_lower: int, n_upper: int, epsilon: float, discretization_size: int = 100) -> int:
        """Noise ``x`` with m-LDP: sample near ``x`` within [n_lower, n_upper].

        Smaller ``epsilon`` means more noise.
        """
        n_lower, n_upper = int(n_lower), int(n_upper)
        total_range = n_upper - n_lower
        x_scaled = (x - n_lower) * discretization_size / total_range
        p = np.exp(-np.abs(x_scaled - np.arange(discretization_size)) * epsilon / 2)
        sample = self.rng.choice(discretization_size, 1, p=p / p.sum())[0]
        return int(sample * total_range / discretization_size + n_lower)

    def _decrypt_targets(self, i: int, extracted: Optional[list[list[str]]], use_cache: bool) -> list[str]:
        """Fakes to undo in text ``i``: detected, or saved by ``encrypt``."""
        if extracted is None or use_cache:
            return [fake for fake in dict.fromkeys(self.entity_mapping[i].values()) if fake is not None]
        return extracted[i]


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _index(alphabet: list[str], name: str) -> str:
    """Return ``name``'s position in ``alphabet`` as 3 digits, e.g. "007"."""
    i = check(alphabet, name)
    if i is None:
        raise ValueError(f"{name!r} is not in the names alphabet.")
    return str(i).zfill(DIGITS_PER_NAME)


def _digits_to_name(digits: str, first_names: list[str], last_names: list[str], skip_no_name: bool = False) -> str:
    """Turn 3-digit indexes into a name: a first name, then last names."""
    chunks = [digits[i : i + DIGITS_PER_NAME] for i in range(0, len(digits), DIGITS_PER_NAME)]
    words = [first_names[int(chunks[0])]]
    words += [last_names[int(c)] for c in chunks[1:] if not (skip_no_name and c == NO_NAME)]
    return " ".join(words)


def _name_words(all_names: list[str]) -> tuple[list[str], list[str]]:
    """Split names into first words and the remaining words of full names."""
    first_words = [name.split()[0] if " " in name else name for name in all_names]
    last_words = [word for name in all_names if " " in name for word in name.split()[1:]]
    return first_words, last_words


def _make_alphabet(dataset_names: list[str], input_names: list[str]) -> list[str]:
    """Return 1000 single-word names that include every input name, near the end."""
    alphabet = [name for name in dataset_names if " " not in name]
    for name in input_names:
        if name in alphabet:
            alphabet.remove(name)
    alphabet = alphabet[:ALPHABET_SIZE]
    return alphabet[: min(-len(input_names) - 1, -2)] + list(input_names) + [alphabet[-1]]


def _money_bounds(amount: int) -> tuple[int, int]:
    """m-LDP noise range for an amount."""
    if amount == 1:
        return 1, 2
    if amount < 100:
        return 2, 1000
    if amount < 1000:
        return 100, 10000
    return 1000, 10000


def _digits_only(value: str) -> str:
    return re.sub(r"[$,. ]", "", str(value))


def _restore_format(digits: str, reference: str) -> str:
    """Put the non-digit characters of ``reference`` back into ``digits``."""
    if len(digits) != len(reference):
        for i, char in enumerate(reference):
            if not char.isdigit():
                digits = digits[:i] + char + digits[i:]
    return digits


def _replace_first_word(text: str, old: str, new: str) -> str:
    """Replace the first whole-word occurrence of ``old`` in ``text``."""
    return re.sub(r"\b" + re.escape(old) + r"\b", new, text, count=1)
