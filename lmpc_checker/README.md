# Legal Metrology Compliance Rule Engine

A Python-based compliance validation system for e-commerce product data against the Indian Legal Metrology (Packaged Commodities) Rules, 2011.

## Overview

This project implements a `ComplianceValidator` class that validates product information extracted from e-commerce platforms against legal metrology requirements. The system is designed to integrate with OCR/AI pipelines that extract structured product data from images.

## Features

- **Comprehensive Rule Coverage**: Implements 13 specific rules covering all major Legal Metrology requirements
- **Flexible Input Handling**: Handles missing or `None` values gracefully (common in OCR scenarios)
- **Severity Classification**: Categorizes violations as Critical, High, Medium severity
- **Detailed Validation Reports**: Returns structured violation reports with rule IDs, descriptions, and affected fields
- **Multiple Validation Patterns**: Supports regex-based format validation, date validation, and conditional logic

## Project Structure

```
d:\lmpc_checker\
├── compliance_validator.py     # Main ComplianceValidator class
├── main.py                     # Demonstration script with examples
├── test_examples.py           # Comprehensive test suite
└── README.md                  # This documentation
```

## Installation

1. Ensure Python 3.8+ is installed
2. Install required dependencies:
   ```bash
   pip install rule-engine
   ```

## Usage

### Basic Usage

```python
from compliance_validator import ComplianceValidator

# Create validator instance
validator = ComplianceValidator()

# Define product data
product_data = {
    "product_id": "SKU-ELEC-001",
    "category": "Electronics", 
    "manufacturer_details": "Noise, Gurugram, Haryana, 122001",
    "importer_details": None,
    "net_quantity": "1 Unit",
    "mrp": "₹1999",
    "unit_sale_price": None,
    "country_of_origin": "India",
    "date_of_manufacture": "08/2024",
    "date_of_import": None,
    "best_before_date": None,
    "consumer_care": "help@go-noise.com, +91 88821 32132",
    "dimensions": "4.6 cm (1.83 inch) HD Display",
    "contents": "1N Smartwatch, 1N Charging Cable, 1N Manual"
}

# Validate the product
violations = validator.validate(product_data)

# Check results
if violations:
    print(f"Found {len(violations)} violations:")
    for violation in violations:
        print(f"- {violation['rule_id']}: {violation['description']}")
else:
    print("✅ Product is compliant!")
```

### Expected Input Data Structure

The validator expects a dictionary with the following fields (any field can be `None`):

```python
{
    "product_id": str,              # Product identifier
    "category": str,                # Product category (Electronics, Groceries, etc.)
    "manufacturer_details": str,    # Manufacturer name and address
    "importer_details": str,        # Importer details (if applicable)
    "net_quantity": str,           # Quantity with unit (e.g., "500g", "1 Unit")
    "mrp": str,                    # Maximum Retail Price (e.g., "₹1999")
    "unit_sale_price": str,        # Unit sale price (for groceries)
    "country_of_origin": str,      # Country where product was made
    "date_of_manufacture": str,    # Manufacturing date (MM/YYYY or DD/MM/YYYY)
    "date_of_import": str,         # Import date (if imported)
    "best_before_date": str,       # Expiry/best before date
    "consumer_care": str,          # Contact information (email/phone)
    "dimensions": str,             # Product dimensions
    "contents": str               # Package contents description
}
```

### Output Structure

The `validate()` method returns a list of violation dictionaries:

```python
[
    {
        "rule_id": "LM_RULE_01_MRP_MISSING",
        "description": "The Maximum Retail Price (MRP) is a mandatory declaration and is missing.",
        "violating_field": "mrp",
        "severity": "critical"
    }
]
```

## Implemented Rules

### Mandatory Field Presence Rules

| Rule ID | Description | Severity |
|---------|-------------|----------|
| `LM_RULE_01_MRP_MISSING` | MRP is missing | Critical |
| `LM_RULE_02_NET_QTY_MISSING` | Net Quantity is missing | Critical |
| `LM_RULE_03_ORIGIN_MISSING` | Country of Origin is missing | Critical |
| `LM_RULE_04_MFG_DETAILS_MISSING` | Manufacturer/Importer details missing | Critical |
| `LM_RULE_05_DATE_MISSING` | Manufacturing/Import date missing | Critical |
| `LM_RULE_06_CARE_INFO_MISSING` | Consumer care details missing | Critical |

### Format and Content Validation Rules

| Rule ID | Description | Severity |
|---------|-------------|----------|
| `LM_RULE_07_MRP_FORMAT` | Invalid MRP format | High |
| `LM_RULE_08_NET_QTY_FORMAT` | Invalid net quantity format | High |
| `LM_RULE_09_DATE_FUTURE` | Date is in the future | High |
| `LM_RULE_10_CARE_INFO_FORMAT` | Invalid contact format | Medium |

### Conditional & Category-Specific Rules

| Rule ID | Description | Severity |
|---------|-------------|----------|
| `LM_RULE_11_BEST_BEFORE_MISSING` | Missing expiry date for applicable categories | High |
| `LM_RULE_12_IMPORT_DATE_MISSING` | Missing import date for imported products | High |
| `LM_RULE_13_USP_MISSING` | Missing unit sale price for groceries | Medium |

## Validation Details

### Supported MRP Formats
- `₹1999`
- `Rs. 1999`
- `Rs.1999`
- `₹ 1999.50`

### Supported Net Quantity Units
- Weight: `g`, `kg`
- Volume: `ml`, `L`
- Length: `cm`, `m`
- Count: `N`, `Unit`, `Units`, `Piece`, `Pieces`

### Supported Date Formats
- `MM/YYYY` (e.g., `08/2024`)
- `DD/MM/YYYY` (e.g., `15/08/2024`)
- `MM-YYYY` and `DD-MM-YYYY`
- `YYYY-MM` and `YYYY-MM-DD`

### Contact Information Validation
- **Email**: Standard email format validation
- **Phone**: Indian phone numbers (10 digits starting with 6-9, with optional +91 country code)

### Category-Specific Rules
- **Groceries**: Require best before date and unit sale price
- **Personal Care**: Require best before date
- **Medicine**: Require best before date

## Running Tests

Execute the test suite to see various validation scenarios:

```bash
python test_examples.py
```

Run the main demonstration:

```bash
python main.py
```

## Integration with FastAPI

This validator can be easily integrated into a FastAPI backend:

```python
from fastapi import FastAPI
from compliance_validator import ComplianceValidator

app = FastAPI()
validator = ComplianceValidator()

@app.post("/validate-product")
async def validate_product(product_data: dict):
    violations = validator.validate(product_data)
    return {
        "is_compliant": len(violations) == 0,
        "violations": violations,
        "total_violations": len(violations)
    }
```

## Legal Metrology Context

This implementation is based on the Indian Legal Metrology (Packaged Commodities) Rules, 2011, which mandates specific declarations on packaged goods sold in India. The rules ensure consumer protection by requiring clear, accurate product information.

## Contributing

The rule engine is designed to be extensible. To add new rules:

1. Add a new rule tuple to the `self.rules` list in `ComplianceValidator.__init__()`
2. Implement any required helper validation functions
3. Add test cases to verify the new rule

## License

This project is intended for educational and compliance purposes. Ensure you understand current legal requirements before using in production.