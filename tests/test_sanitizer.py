"""Round-trip tests for ``preempt.sanitizer.Sanitizer``."""

from preempt.contracts import EntitySpan
from preempt.sanitizer import Sanitizer, generate_key


KEY, TWEAK = "EF4359D8D580AA4F7F036D6F04FC6A94", "D8E7920AFA330A73"


class FixedNER:
    """NER stand-in that returns the same detections for any texts."""

    def __init__(self, values: list[list[str]]) -> None:
        self.values = values

    def extract(self, texts: list[str], entity_type: str) -> dict[str, list[list[str]]]:
        """Return a copy of the fixed detections."""
        return {entity_type: [list(v) for v in self.values]}


class AgeRecognizer:
    """Recognizer stand-in that tags every two-digit number as an age."""

    def recognize(self, text: str) -> list[EntitySpan]:
        """Return an AGE span for each two-digit word."""
        spans = []
        for word in text.split():
            if word.isdigit() and len(word) == 2:
                start = text.index(word)
                spans.append(EntitySpan(start, start + 2, word, "AGE", 1.0, "stub"))
        return spans


def test_recognizer_is_wrapped_for_extract() -> None:
    """A span recognizer can be passed straight to Sanitizer."""
    texts = ["She is 52 today."]
    sanitizer = Sanitizer(AgeRecognizer(), KEY, TWEAK, seed=0)
    encrypted, invalid = sanitizer.encrypt(texts, entity="Age")

    assert not invalid
    assert sanitizer.decrypt(encrypted, entity="Age") == texts


def test_names_fpe_round_trip() -> None:
    """FPE names change, including a standalone first name, and decrypt back."""
    texts = ["Sarah Lopez met John. Sarah was late.", "Mary Ann Smith called Garcia."]
    sanitizer = Sanitizer(FixedNER([["Sarah Lopez", "John"], ["Mary Ann Smith", "Garcia"]]), KEY, TWEAK)
    encrypted, invalid = sanitizer.encrypt(texts, entity="Name")

    assert not invalid
    assert "Sarah" not in encrypted[0] and "John" not in encrypted[0]
    assert sanitizer.decrypt(encrypted, entity="Name", use_cache=True) == texts


def _listed_names(n: int) -> list[str]:
    """``n`` full names whose words are all in the fixed name lists."""
    sanitizer = Sanitizer(None, KEY, TWEAK)
    sanitizer._load_name_lists()
    first, last = sanitizer._name_lists["first"], sanitizer._name_lists["last"]
    return [f"{first[i * 37]} {last[i * 53]}" for i in range(n)]


def test_names_differ_and_are_stable_across_calls() -> None:
    """Different names get different surrogates, the same in any call."""
    sanitizer = Sanitizer(None, KEY, TWEAK)
    names = _listed_names(5) + ["Wei Zhang", "Yasi Gholami"]
    alone = {n: sanitizer.encrypt_names([[n]])[0][0][n] for n in names}
    batched = sanitizer.encrypt_names([[n] for n in reversed(names)])[0]

    assert len(set(alone.values())) == len(names)
    assert all(alone[n] == m[n] for n, m in zip(reversed(names), batched))
    assert all(alone[n] != n for n in names)


def test_partial_mentions_match_the_full_name() -> None:
    """A lone first or last name maps to the matching word of the full surrogate."""
    full = _listed_names(1)[0]
    first, last = full.split()
    fakes = Sanitizer(None, KEY, TWEAK).encrypt_names([[full, first, last]])[0][0]

    assert fakes[full] == f"{fakes[first]} {fakes[last]}"
    assert len(fakes[first].split()) == 1


def test_listed_names_decrypt_with_the_key_alone() -> None:
    """A fresh Sanitizer with the same key restores names from the fixed lists."""
    full = _listed_names(1)[0]
    texts = [f"{full} called."]
    encrypted, _ = Sanitizer(FixedNER([[full]]), KEY, TWEAK).encrypt(texts, entity="Name")

    fresh = Sanitizer(None, KEY, TWEAK)
    fake = encrypted[0].removesuffix(" called.")
    assert fresh.decrypt(encrypted, entity="Name", extracted=[[fake]]) == texts


def test_unlisted_names_round_trip_without_collisions() -> None:
    """Unlisted names get distinct surrogates in a text and decrypt via saved state."""
    texts = ["Yasi Gholami met Roya and Hamid Aziz."]
    sanitizer = Sanitizer(FixedNER([["Yasi Gholami", "Roya", "Hamid Aziz"]]), KEY, TWEAK)
    encrypted, invalid = sanitizer.encrypt(texts, entity="Name")
    fakes = sanitizer.entity_mapping[0]

    assert not invalid
    assert len({fakes["Yasi Gholami"], fakes["Roya"], fakes["Hamid Aziz"]}) == 3
    assert sanitizer.decrypt(encrypted, entity="Name", use_cache=True) == texts


def test_money_fpe_round_trip() -> None:
    """FPE amounts decrypt back to the originals."""
    texts = ["It costs $85000 a year.", "Pay 450 now and 12 later."]
    sanitizer = Sanitizer(FixedNER([["85000"], ["450", "12"]]), KEY, TWEAK)
    encrypted, _ = sanitizer.encrypt(texts, entity="Money")

    assert encrypted != texts
    assert sanitizer.decrypt(encrypted, entity="Money", use_cache=True) == texts


def test_money_mldp_round_trip_keeps_commas() -> None:
    """m-LDP amounts are restored exactly as written."""
    texts = ["Rent is 1,250 a month."]
    sanitizer = Sanitizer(FixedNER([["1,250"]]), KEY, TWEAK, seed=0)
    encrypted, _ = sanitizer.encrypt(texts, entity="Money", use_fpe=False, use_mdp=True)

    assert sanitizer.decrypt(encrypted, entity="Money", use_fpe=False, use_mdp=True, use_cache=True) == texts


def test_age_mldp_is_seeded_and_reversible() -> None:
    """Noised ages stay in [10, 99], repeat for the same seed, and decrypt back."""
    texts = ["She is 52 and he is 30."]
    sanitizer = Sanitizer(FixedNER([["52", "30"]]), KEY, TWEAK, seed=1)
    encrypted, _ = sanitizer.encrypt(texts, epsilon=1.0, entity="Age")

    assert all(10 <= int(fake) <= 99 for fake in sanitizer.entity_mapping[0].values() if fake)
    sanitizer.reseed(1)
    assert sanitizer.encrypt(texts, epsilon=1.0, entity="Age")[0] == encrypted
    assert sanitizer.decrypt(encrypted, entity="Age") == texts


def test_unparsable_values_are_reported() -> None:
    """Values that can't be encrypted map to None and their text is flagged."""
    sanitizer = Sanitizer(FixedNER([["fifty"]]), KEY, TWEAK)
    _, invalid = sanitizer.encrypt(["He is fifty."], entity="Age")

    assert invalid == [0]
    assert sanitizer.entity_mapping == [{"fifty": None}]


def test_generated_keys_differ_and_round_trip() -> None:
    """Each generated pair is fresh and works with the FF3 cipher."""
    key, tweak = generate_key()
    texts = ["It costs $85000 a year."]
    sanitizer = Sanitizer(FixedNER([["85000"]]), key, tweak)
    encrypted, _ = sanitizer.encrypt(texts, entity="Money")

    assert generate_key() != (key, tweak)
    assert sanitizer.decrypt(encrypted, entity="Money", use_cache=True) == texts
