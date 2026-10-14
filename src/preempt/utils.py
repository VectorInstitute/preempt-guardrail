"""Name-list helpers for the sanitizer."""

import json
import unicodedata
from typing import Any

from names_dataset import NameDataset


def check(name_list: list[str], target: str) -> int | None:
    """For checking if a target name exists in a given list of names."""
    for i, n in enumerate(name_list):
        if finder(n, target):
            return i
    return None


def finder(word1: str, word2: str) -> bool:
    """Return whether two names match after Unicode normalization."""
    encoded1 = unicodedata.normalize("NFC", word1)
    encoded2 = unicodedata.normalize("NFC", word2)
    return encoded1 == encoded2


def make_names_dataset(save_path: str) -> None:
    """For making a new dataset of names for sanitizing names."""
    name_data = NameDataset()

    countries = ["US", "GB", "FR"]
    first_names = []
    last_names = []

    for country in countries:
        first_names = first_names + name_data.get_top_names(n=1000, gender="M", country_alpha2=country)[country]["M"]
    for country in countries:
        last_names = (
            last_names + name_data.get_top_names(n=1000, country_alpha2=country, use_first_names=False)[country]
        )

    # Get rid of duplicates
    remove_fns = ["Saba", "Lanka", "Deblog", "Donas", "Bk"]
    remove_lns = ["guez", "quez", "ecour", "Mai", "ü", "é", "Behal", "Bk"]
    first_names = list(set(first_names))
    last_names = list(set(last_names))

    temp = []
    for i in range(len(first_names)):
        flag = 0
        for rfn in remove_fns:
            if rfn in first_names[i]:
                flag = 1
                break
        if flag == 0:
            temp.append(first_names[i])

    first_names = temp[:]
    for i in range(len(last_names)):
        flag = 0
        for rfn in remove_lns:
            if rfn in last_names[i]:
                flag = 1
                break
        if flag == 0:
            temp.append(last_names[i])

    last_names = temp[:]
    save_fn({"first_names": first_names, "last_names": last_names}, save_path)


def save_fn(dataset: dict[str, Any], fp: str) -> None:
    """Save a JSON-formatted dictionary to a file path."""
    with open(fp, "w") as handle:
        json.dump(dataset, handle, indent=2, sort_keys=True)


def load_data(fp: str) -> dict[str, Any]:
    """Load a JSON file."""
    with open(fp, "r") as handle:
        return json.load(handle)
