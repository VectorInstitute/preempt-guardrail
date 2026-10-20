"""Round-trip tests for ``preempt.sanitizer.Sanitizer``."""

from preempt.sanitizer import Sanitizer


class FixedNER:
    """NER stand-in that returns the same detections for any texts."""

    def __init__(self, values: list[list[str]]) -> None:
        self.values = values

    def extract(self, texts: list[str], entity_type: str) -> dict[str, list[list[str]]]:
        """Return a copy of the fixed detections."""
        return {entity_type: [list(v) for v in self.values]}


def test_names_fpe_round_trip() -> None:
    """FPE names change, including a standalone first name, and decrypt back."""
    texts = ["Sarah Lopez met John. Sarah was late.", "Mary Ann Smith called Garcia."]
    sanitizer = Sanitizer(FixedNER([["Sarah Lopez", "John"], ["Mary Ann Smith", "Garcia"]]))
    encrypted, invalid = sanitizer.encrypt(texts, entity="Name")

    assert not invalid
    assert "Sarah" not in encrypted[0] and "John" not in encrypted[0]
    assert sanitizer.decrypt(encrypted, entity="Name", use_cache=True) == texts


def test_money_fpe_round_trip() -> None:
    """FPE amounts decrypt back to the originals."""
    texts = ["It costs $85000 a year.", "Pay 450 now and 12 later."]
    sanitizer = Sanitizer(FixedNER([["85000"], ["450", "12"]]))
    encrypted, _ = sanitizer.encrypt(texts, entity="Money")

    assert encrypted != texts
    assert sanitizer.decrypt(encrypted, entity="Money", use_cache=True) == texts


def test_money_mldp_round_trip_keeps_commas() -> None:
    """m-LDP amounts are restored exactly as written."""
    texts = ["Rent is 1,250 a month."]
    sanitizer = Sanitizer(FixedNER([["1,250"]]), seed=0)
    encrypted, _ = sanitizer.encrypt(texts, entity="Money", use_fpe=False, use_mdp=True)

    assert sanitizer.decrypt(encrypted, entity="Money", use_fpe=False, use_mdp=True, use_cache=True) == texts


def test_age_mldp_is_seeded_and_reversible() -> None:
    """Noised ages stay in [10, 99], repeat for the same seed, and decrypt back."""
    texts = ["She is 52 and he is 30."]
    sanitizer = Sanitizer(FixedNER([["52", "30"]]), seed=1)
    encrypted, _ = sanitizer.encrypt(texts, epsilon=1.0, entity="Age")

    assert all(10 <= int(fake) <= 99 for fake in sanitizer.entity_mapping[0].values() if fake)
    sanitizer.reseed(1)
    assert sanitizer.encrypt(texts, epsilon=1.0, entity="Age")[0] == encrypted
    assert sanitizer.decrypt(encrypted, entity="Age") == texts


def test_unparseable_values_are_reported() -> None:
    """Values that can't be encrypted map to None and their text is flagged."""
    sanitizer = Sanitizer(FixedNER([["fifty"]]))
    _, invalid = sanitizer.encrypt(["He is fifty."], entity="Age")

    assert invalid == [0]
    assert sanitizer.entity_mapping == [{"fifty": None}]
