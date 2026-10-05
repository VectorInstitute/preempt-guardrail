"""Map ai4privacy labels onto the canonical taxonomy.

Labels absent from ``SOURCE_TO_LABEL`` are ignored during training.
"""

from preempt.contracts import LABELS


SOURCE_TO_LABEL = {
    "FIRSTNAME": "NAME",
    "MIDDLENAME": "NAME",
    "LASTNAME": "NAME",
    "AGE": "AGE",
    "AMOUNT": "MONEY",
    "EMAIL": "EMAIL",
    "PHONENUMBER": "PHONE",
    "STREET": "ADDRESS",
    "BUILDINGNUMBER": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "SSN": "ID",
    "IBAN": "ID",
    "ACCOUNTNUMBER": "ID",
    "CREDITCARDNUMBER": "ID",
    "VEHICLEVIN": "ID",
    "MASKEDNUMBER": "ID",
}


def check_label_map() -> None:
    """Raise ValueError when a mapped label is outside the canonical set."""
    unknown = sorted({label for label in SOURCE_TO_LABEL.values() if label not in LABELS})
    if unknown:
        raise ValueError(f"SOURCE_TO_LABEL has unknown labels: {', '.join(unknown)}")
