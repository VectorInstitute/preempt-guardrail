"""Replace names, money, and ages in text with fake values, and undo it.

- Names: a keyed permutation of fixed first- and last-name lists (reversible
  with the key alone), or random names. Words outside the lists get a keyed
  hash pick that only the ``encrypt`` call's saved state can reverse.
- Money: FPE (reversible) or m-LDP noise. FPE appends a non-zero suffix so a
  round amount does not make FF3 reject the value.
- Ages: m-LDP noise.

``decrypt`` restores only the surrogates saved by the last ``encrypt`` call on
this instance. ``key_only=True`` inverts listed names and FPE amounts with the
key alone. A value that cannot be encrypted is replaced with ``[REDACTED]``;
the other values in that text are still replaced.

Known limitation: values are matched as plain substrings ("Ann" also hits
"Anna"), and some decrypt paths restore only the first match. See
https://github.com/VectorInstitute/preempt-guardrail/issues/11.
"""

import hashlib
import itertools
import logging
import os
import random
import re
import secrets
import unicodedata
from typing import Any, Optional

import names
import numpy as np
from pyfpe_ff3 import FF3Cipher

from preempt.legacy import LegacyNERShim
from preempt.utils import load_data, make_names_dataset


logger = logging.getLogger(__name__)

# Size of each fixed name list. Raising it needs a names dataset with at least
# this many first names and this many (distinct) last names.
ALPHABET_SIZE = 1000
# FF3 rejects a numeral whose half becomes all zeros. Appending this keeps the
# right half of a round amount from starting as zeros. It is stripped on decrypt.
MONEY_FPE_SUFFIX = "999999"
# Stands in for a value that could not be encrypted, so the original is not returned.
REDACTED = "[REDACTED]"


def generate_key() -> tuple[str, str]:
    """Return a random hex ``(key, tweak)``: a 256-bit key and a 64-bit tweak."""
    return secrets.token_hex(32), secrets.token_hex(8)


class Sanitizer:
    """Encrypt and decrypt names, money, and ages detected by ``ner_model``."""

    def __init__(
        self,
        ner_model: Any,
        key: str,
        tweak: str,
        seed: Optional[int] = None,
    ):
        """Set up the FF3 cipher and name permutations for ``key`` and ``tweak``.

        Anyone with ``key`` and ``tweak`` can reverse FPE names and amounts,
        so keep them secret. Use ``generate_key`` to make a fresh pair.
        ``ner_model`` is either an object with ``extract`` or a ``Recognizer``,
        which is wrapped in ``LegacyNERShim``.
        """
        self.cipher_fn = FF3Cipher(key, tweak, allow_small_domain=True, radix=10)
        self._key, self._tweak = key, tweak
        if hasattr(ner_model, "recognize") and not hasattr(ner_model, "extract"):
            ner_model = LegacyNERShim(ner_model)
        self.ner = ner_model
        # Set by encrypt(), read by decrypt().
        self.entity_mapping: list[dict[str, Optional[str]]] = []  # {original: fake} per text
        # {fake word: original word} per text, for name words outside the fixed lists.
        self.entity_lookup: list[dict[str, str]] = []
        self._names_dataset: Optional[dict[str, list[str]]] = None
        self._name_lists: dict[str, list[str]] = {}  # {"first" | "last": names}
        self._name_index: dict[str, dict[str, int]] = {}
        self._name_perm: dict[str, np.ndarray] = {}
        self._name_inverse: dict[str, np.ndarray] = {}
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

        A value that cannot be encrypted is replaced with ``[REDACTED]`` and its
        row index is returned in the second list. Other values in that row are
        still replaced.
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
            failed = False
            for original, fake in fakes.items():
                if fake is None:
                    failed = True
                    replaced = _nfc(replaced).replace(_nfc(original), REDACTED)
                    continue
                replaced = _nfc(replaced).replace(_nfc(original), fake)
            if failed:
                invalid.append(i)
            sanitized.append(replaced)
        return sanitized, invalid

    def decrypt(  # noqa: PLR0917
        self,
        inputs: list[str],
        entity: str = "Name",
        extracted: Optional[list[list[str]]] = None,
        use_mdp: bool = False,  # noqa: ARG002 -- kept for API compatibility; m-LDP is used when not use_fpe
        use_fpe: bool = True,
        use_cache: bool = False,  # noqa: ARG002 -- stateful decrypt always uses saved surrogates
        key_only: bool = False,
    ) -> list[str]:
        """Restore values that ``encrypt`` replaced.

        By default only surrogates saved by the last ``encrypt`` call on this
        instance are restored. A name or amount that was never encrypted is
        left as it is, including one the recognizer missed and one added later.

        Pass ``key_only=True`` to invert listed names and FPE amounts with the
        key alone. Targets then come from ``extracted``, or from a fresh
        recognizer pass when ``extracted`` is omitted. An amount FF3 rejects
        is left in place. Ages have no key-only inverse; they still use the
        saved state.

        ``use_cache`` is accepted for existing callers. Stateful decrypt always
        uses the saved surrogates.
        """
        if key_only and extracted is None and entity != "Age":
            extracted = self.ner.extract(inputs, entity_type=entity)[entity]
        if entity == "Name":
            return self.decrypt_names(inputs, extracted, key_only=key_only)
        if entity == "Money":
            return self.decrypt_money(inputs, extracted, use_fpe=use_fpe, key_only=key_only)
        if entity == "Age":
            return self.decrypt_age(inputs)
        raise ValueError(f"Unknown entity: {entity!r}")

    # --- Names ----------------------------------------------------------------

    def encrypt_names(
        self, inputs: list[list[str]], use_fpe: bool = True
    ) -> tuple[list[dict[str, Optional[str]]], list[dict[str, str]]]:
        """Map each name to a fake name, per text.

        With FPE, each word is swapped on its own (first word in the first-name
        list, later words in the last-name list), so "Sarah Lopez", "Sarah" and
        "Lopez" map consistently, within and across calls. Without FPE, fake
        names are random and can't be decrypted.

        Returns ``{original: fake}`` per text and, per text, the
        ``{fake word: original word}`` entries for words outside the lists.
        """
        if not use_fpe:
            mappings: list[dict[str, Optional[str]]] = []
            for text_names in inputs:
                fakes: dict[str, Optional[str]] = {}
                for name in text_names:
                    fakes.setdefault(name, names.get_full_name(gender="male"))
                mappings.append(fakes)
            return mappings, [{} for _ in inputs]

        self._load_name_lists()
        mappings, fallbacks = [], []
        for text_names in inputs:
            roles = self._word_roles(text_names)
            word_fakes = {w: self._permute_word(w, role) for w, role in roles.items() if w in self._name_index[role]}
            # Unlisted words get a keyed pick that avoids surrogates used here.
            used = set(word_fakes.values())
            fallback: dict[str, str] = {}
            for word, role in roles.items():
                if word not in word_fakes:
                    fake = self._fallback_word(word, role, used)
                    word_fakes[word] = fake
                    fallback[fake] = word
                    used.add(fake)

            fakes = {}
            for name in text_names:
                fakes.setdefault(name, " ".join(word_fakes[w] for w in _nfc(name).split()))
            for word, fake in word_fakes.items():
                fakes.setdefault(word, fake)
            mappings.append(fakes)
            fallbacks.append(fallback)
        return mappings, fallbacks

    def decrypt_names(
        self,
        inputs: list[str],
        extracted: Optional[list[list[str]]],
        use_cache: bool = False,  # noqa: ARG002 -- stateful decrypt always uses saved surrogates
        key_only: bool = False,
    ) -> list[str]:
        """Replace fake names in ``inputs`` with the originals.

        By default only surrogates from the last ``encrypt`` call are undone.
        ``key_only=True`` inverts every target with the key. Words from the
        fixed lists need only the key; words that were outside the lists need
        the state saved by ``encrypt``. Words that are neither are left as
        they are.
        """
        self._load_name_lists()
        decrypted = []
        for i, line in enumerate(inputs):
            fallback = self.entity_lookup[i] if i < len(self.entity_lookup) else {}
            restored = line
            for fake in self._decrypt_targets(i, extracted, key_only):
                restored = restored.replace(fake, self._decrypt_name(fake, fallback))
            decrypted.append(restored)
        return decrypted

    def _decrypt_name(self, fake: str, fallback: dict[str, str]) -> str:
        """Undo the word swaps in one fake name."""
        words = _nfc(fake).split()
        if len(words) == 1:
            # The lists share no names, so a lone word's list is unambiguous.
            roles = ["first" if words[0] in self._name_index["first"] else "last"]
        else:
            roles = ["first"] + ["last"] * (len(words) - 1)
        restored = []
        for word, role in zip(words, roles):
            if word in fallback:
                restored.append(fallback[word])
            elif (i := self._name_index[role].get(word)) is not None:
                restored.append(self._name_lists[role][self._name_inverse[role][i]])
            else:
                restored.append(word)
        return " ".join(restored)

    def _word_roles(self, text_names: list[str]) -> dict[str, str]:
        """Assign each name word to the "first" or "last" list, once per text."""
        later_words = {w for name in text_names for w in _nfc(name).split()[1:]}
        roles: dict[str, str] = {}
        for name in text_names:
            words = _nfc(name).split()
            for pos, word in enumerate(words):
                if word in roles:
                    continue
                if len(words) > 1:
                    roles[word] = "first" if pos == 0 else "last"
                elif word in later_words or (
                    word in self._name_index["last"] and word not in self._name_index["first"]
                ):
                    roles[word] = "last"
                else:
                    roles[word] = "first"
        return roles

    def _permute_word(self, word: str, role: str) -> str:
        return self._name_lists[role][self._name_perm[role][self._name_index[role][word]]]

    def _fallback_word(self, word: str, role: str, used: set[str]) -> str:
        """Pick a list name for an unlisted word by keyed hash, skipping ``used``."""
        names_list = self._name_lists[role]
        for attempt in itertools.count():
            digest = hashlib.sha256(f"{self._key}:{self._tweak}:{role}:{word}:{attempt}".encode()).digest()
            fake = names_list[int.from_bytes(digest[:8], "big") % len(names_list)]
            if fake not in used:
                return fake
        raise AssertionError("unreachable")

    def _load_name_lists(self) -> None:
        """Build the fixed first/last name lists and their keyed permutations once.

        The lists depend only on the names dataset, never on the inputs, so a
        name's surrogate depends only on the name and the key. The last-name
        list excludes every first name, so the two lists share no names.
        """
        if self._name_lists:
            return
        dataset = self._load_names_dataset()
        first = _clean_names(dataset["first_names"], exclude=set())
        last = _clean_names(dataset["last_names"], exclude=set(first))
        if len(first) < ALPHABET_SIZE or len(last) < ALPHABET_SIZE:
            raise ValueError(
                f"Names dataset has {len(first)} first and {len(last)} last names; need {ALPHABET_SIZE} of each."
            )
        for role, names_list in (("first", first[:ALPHABET_SIZE]), ("last", last[:ALPHABET_SIZE])):
            seed = int.from_bytes(hashlib.sha256(f"{self._key}:{self._tweak}:{role}".encode()).digest()[:8], "big")
            perm = np.random.default_rng(seed).permutation(ALPHABET_SIZE)
            self._name_lists[role] = names_list
            self._name_index[role] = {name: i for i, name in enumerate(names_list)}
            self._name_perm[role] = perm
            self._name_inverse[role] = np.argsort(perm)

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
            logger.info("Names dataset not found. Generating %s", path)
            make_names_dataset(path)
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
                        fakes[amount] = self.fpe_encrypt(digits + MONEY_FPE_SUFFIX)
                    else:
                        value = int(float(digits))
                        fakes[amount] = str(self.m_epsilon(value, *_money_bounds(value), epsilon))
                except (ValueError, TypeError):
                    fakes[amount] = None
            mappings.append(fakes)
        return mappings

    def decrypt_money(
        self,
        inputs: list[str],
        extracted: Optional[list[list[str]]],
        use_fpe: bool = True,
        use_cache: bool = False,  # noqa: ARG002 -- stateful decrypt always uses saved surrogates
        key_only: bool = False,
    ) -> list[str]:
        """Replace fake amounts in ``inputs`` with the originals.

        By default only surrogates from the last ``encrypt`` call are undone.
        ``key_only=True`` inverts FPE amounts with the key and skips a value
        FF3 rejects, so one amount cannot fail the batch.
        """
        decrypted = []
        for i, line in enumerate(inputs):
            restored = line
            mapping = self.entity_mapping[i] if i < len(self.entity_mapping) else {}
            fake_to_original = {fake: original for original, fake in mapping.items() if fake is not None}
            for fake in self._decrypt_targets(i, extracted, key_only):
                if not use_fpe:
                    original = fake_to_original.get(fake)
                    if original is None:
                        continue
                    restored = _replace_first_word(restored, fake, original)
                    continue
                original = self._unwrap_money(fake)
                if original is None:
                    continue
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

    def _unwrap_money(self, fake: str) -> Optional[str]:
        """Return the original digits of an FPE amount.

        ``None`` when ``fake`` is too short to be one of ours, FF3 rejects it,
        or decrypting it does not end in the suffix this class appends.
        """
        if len(_digits_only(fake)) <= len(MONEY_FPE_SUFFIX):
            return None
        try:
            decoded = self.fpe_decrypt(fake)
        except (ValueError, TypeError):
            return None
        if not decoded.endswith(MONEY_FPE_SUFFIX):
            return None
        return decoded[: -len(MONEY_FPE_SUFFIX)]

    def _saved_fakes(self, index: int) -> list[str]:
        """Surrogates saved by ``encrypt`` for text ``index``, in insertion order."""
        if index >= len(self.entity_mapping):
            return []
        return [fake for fake in dict.fromkeys(self.entity_mapping[index].values()) if fake is not None]

    def _decrypt_targets(self, index: int, extracted: Optional[list[list[str]]], key_only: bool) -> list[str]:
        """Values to undo in text ``index``.

        Stateful decrypt uses the surrogates ``encrypt`` saved. Key-only
        decrypt uses ``extracted`` and does not require that call.
        """
        if not key_only:
            return self._saved_fakes(index)
        if extracted is None:
            return []
        return list(extracted[index])


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _clean_names(dataset_names: list[str], exclude: set[str]) -> list[str]:
    """Return unique alphabetic 3+ letter names in dataset order, minus ``exclude``."""
    cleaned: dict[str, None] = {}
    for raw in dataset_names:
        name = _nfc(raw)
        if name.isalpha() and len(name) > 2 and name not in exclude:
            cleaned.setdefault(name, None)
    return list(cleaned)


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
