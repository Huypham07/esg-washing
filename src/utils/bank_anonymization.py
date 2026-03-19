"""
Bank Name Anonymization Mapping
For research publication - protects bank identities
"""

BANK_MAPPING = {
    'agribank': 'Bank A',
    'bidv': 'Bank B',
    'bsc': 'Bank C',
    'mbbank': 'Bank D',
    'shb': 'Bank E',
    'techcombank': 'Bank F',
    'vietabank': 'Bank G',
    'vietcombank': 'Bank H',
    'viettinbank': 'Bank I',
    'vpbank': 'Bank J',
}

# Reverse mapping for lookup
REVERSE_MAPPING = {v: k for k, v in BANK_MAPPING.items()}

def anonymize_bank_name(name: str) -> str:
    """Convert real bank name to anonymized code"""
    return BANK_MAPPING.get(name.lower(), name)

def deanonymize_bank_name(code: str) -> str:
    """Convert anonymized code back to real name"""
    return REVERSE_MAPPING.get(code, code)
