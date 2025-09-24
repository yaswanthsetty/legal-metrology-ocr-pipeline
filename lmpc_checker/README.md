# Legal Metrology Compliance Validator# Legal Metrology Compliance Rule Engine



**A comprehensive validation engine for Indian Legal Metrology (Packaged Commodities) Rules, 2011.**A Python-based compliance validation system for e-commerce product data against the Indian Legal Metrology (Packaged Commodities) Rules, 2011.



This module provides automated compliance checking for e-commerce and retail product data, ensuring adherence to legal requirements for packaged goods sold in India.## Overview



## 🎯 OverviewThis project implements a `ComplianceValidator` class that validates product information extracted from e-commerce platforms against legal metrology requirements. The system is designed to integrate with OCR/AI pipelines that extract structured product data from images.



The `ComplianceValidator` class implements a robust validation system that checks product information against 20+ specific Legal Metrology rules. Designed to integrate seamlessly with OCR pipelines and e-commerce platforms.## Features



### Key Capabilities- **Comprehensive Rule Coverage**: Implements 13 specific rules covering all major Legal Metrology requirements

- ✅ **20+ Validation Rules**: Covers all major Legal Metrology requirements- **Flexible Input Handling**: Handles missing or `None` values gracefully (common in OCR scenarios)

- 🔍 **Flexible Input Handling**: Gracefully handles missing or `None` values (common in OCR)- **Severity Classification**: Categorizes violations as Critical, High, Medium severity

- 📊 **Severity Classification**: Categorizes violations as Critical, High, Medium- **Detailed Validation Reports**: Returns structured violation reports with rule IDs, descriptions, and affected fields

- 📋 **Detailed Reports**: Returns structured violation reports with rule IDs and descriptions- **Multiple Validation Patterns**: Supports regex-based format validation, date validation, and conditional logic

- 🛡️ **Production Ready**: Comprehensive error handling and logging

## Project Structure

## 🏗️ Architecture

```

```mermaidd:\lmpc_checker\

graph TD├── compliance_validator.py     # Main ComplianceValidator class

    A[Product Data Input] --> B[Field Validation]├── main.py                     # Demonstration script with examples

    B --> C[Format Validation]├── test_examples.py           # Comprehensive test suite

    C --> D[Content Validation]└── README.md                  # This documentation

    D --> E[Cross-Field Validation]```

    E --> F[Violation Report]

```## Installation



## 📋 Supported Validation Rules1. Ensure Python 3.8+ is installed

2. Install required dependencies:

### Critical Requirements (Must-Have Fields)   ```bash

| Rule ID | Description | Field |   pip install rule-engine

|---------|-------------|-------|   ```

| `LM_RULE_01_MRP_MISSING` | MRP is mandatory | `mrp` |

| `LM_RULE_02_NET_QTY_MISSING` | Net Quantity is mandatory | `net_quantity` |## Usage

| `LM_RULE_03_ORIGIN_MISSING` | Country of Origin is mandatory | `country_of_origin` |

| `LM_RULE_04_MFG_DETAILS_MISSING` | Manufacturer details are mandatory | `manufacturer_details` |### Basic Usage

| `LM_RULE_05_DATE_MISSING` | Manufacturing/Import date is mandatory | `date_of_manufacture` |

| `LM_RULE_06_CARE_INFO_MISSING` | Consumer care details are mandatory | `consumer_care` |```python

from compliance_validator import ComplianceValidator

### Format & Content Validation

| Rule ID | Description | Validation Type |# Create validator instance

|---------|-------------|-----------------|validator = ComplianceValidator()

| `LM_RULE_07_MRP_FORMAT_INVALID` | MRP format validation | Currency format |

| `LM_RULE_08_NET_QTY_FORMAT_INVALID` | Net quantity format validation | Unit format |# Define product data

| `LM_RULE_09_PHONE_FORMAT_INVALID` | Phone number format validation | Phone format |product_data = {

| `LM_RULE_10_DATE_FORMAT_INVALID` | Date format validation | Date format |    "product_id": "SKU-ELEC-001",

| `LM_RULE_11_PIN_CODE_INVALID` | PIN code validation | 6-digit format |    "category": "Electronics", 

| `LM_RULE_12_ORIGIN_INVALID` | Country validation | Valid country names |    "manufacturer_details": "Noise, Gurugram, Haryana, 122001",

| `LM_RULE_13_MRP_ZERO_INVALID` | MRP cannot be zero | Value validation |    "importer_details": None,

    "net_quantity": "1 Unit",

## 🚀 Quick Start    "mrp": "₹1999",

    "unit_sale_price": None,

### Installation    "country_of_origin": "India",

The validator is included with the main OCR pipeline. No separate installation required.    "date_of_manufacture": "08/2024",

    "date_of_import": None,

### Basic Usage    "best_before_date": None,

    "consumer_care": "help@go-noise.com, +91 88821 32132",

```python    "dimensions": "4.6 cm (1.83 inch) HD Display",

from lmpc_checker.compliance_validator import ComplianceValidator    "contents": "1N Smartwatch, 1N Charging Cable, 1N Manual"

}

# Initialize validator

validator = ComplianceValidator()# Validate the product

violations = validator.validate(product_data)

# Product data to validate

product_data = {# Check results

    "product_id": "SKU-ELECTRONICS-001",if violations:

    "category": "Electronics",    print(f"Found {len(violations)} violations:")

    "manufacturer_details": "TechCorp India Pvt Ltd, 123 Tech Park, Bangalore-560001",    for violation in violations:

    "net_quantity": "1 UNIT",        print(f"- {violation['rule_id']}: {violation['description']}")

    "mrp": "₹2999.00",else:

    "country_of_origin": "India",    print("✅ Product is compliant!")

    "date_of_manufacture": "06/2024",```

    "consumer_care": "1800-TECH-HELP, support@techcorp.in"

}### Expected Input Data Structure



# Validate complianceThe validator expects a dictionary with the following fields (any field can be `None`):

violations = validator.validate(product_data)

```python

# Check results{

if violations:    "product_id": str,              # Product identifier

    print(f"❌ Found {len(violations)} compliance violations:")    "category": str,                # Product category (Electronics, Groceries, etc.)

    for violation in violations:    "manufacturer_details": str,    # Manufacturer name and address

        print(f"  • {violation['description']}")    "importer_details": str,        # Importer details (if applicable)

else:    "net_quantity": str,           # Quantity with unit (e.g., "500g", "1 Unit")

    print("✅ Product is fully compliant!")    "mrp": str,                    # Maximum Retail Price (e.g., "₹1999")

```    "unit_sale_price": str,        # Unit sale price (for groceries)

    "country_of_origin": str,      # Country where product was made

## 📊 Input Data Schema    "date_of_manufacture": str,    # Manufacturing date (MM/YYYY or DD/MM/YYYY)

    "date_of_import": str,         # Import date (if imported)

The validator expects a dictionary with the following fields (all fields optional, but required fields will generate violations if missing):    "best_before_date": str,       # Expiry/best before date

    "consumer_care": str,          # Contact information (email/phone)

```python    "dimensions": str,             # Product dimensions

{    "contents": str               # Package contents description

    "product_id": "str",              # Product identifier}

    "category": "str",                # Product category```

    "manufacturer_details": "str",    # Full manufacturer name and address

    "importer_details": "str",        # Importer information (if applicable)### Output Structure

    "net_quantity": "str",            # Net quantity with units (e.g., "500g", "1L")

    "mrp": "str",                     # Maximum Retail Price (e.g., "₹299.00")The `validate()` method returns a list of violation dictionaries:

    "unit_sale_price": "str",         # Unit selling price

    "country_of_origin": "str",       # Country of origin```python

    "date_of_manufacture": "str",     # Manufacturing date (MM/YYYY format)[

    "date_of_import": "str",          # Import date (if applicable)    {

    "best_before_date": "str",        # Expiry/best before date        "rule_id": "LM_RULE_01_MRP_MISSING",

    "consumer_care": "str",           # Customer care contact information        "description": "The Maximum Retail Price (MRP) is a mandatory declaration and is missing.",

    "dimensions": "str",              # Product dimensions        "violating_field": "mrp",

    "contents": "str"                 # Product contents/ingredients        "severity": "critical"

}    }

```]

```

## 📋 Output Format

## Implemented Rules

The `validate()` method returns a list of violation dictionaries:

### Mandatory Field Presence Rules

```python

[| Rule ID | Description | Severity |

    {|---------|-------------|----------|

        "rule_id": "LM_RULE_01_MRP_MISSING",| `LM_RULE_01_MRP_MISSING` | MRP is missing | Critical |

        "severity": "Critical",| `LM_RULE_02_NET_QTY_MISSING` | Net Quantity is missing | Critical |

        "description": "Maximum Retail Price (MRP) is missing",| `LM_RULE_03_ORIGIN_MISSING` | Country of Origin is missing | Critical |

        "field": "mrp",| `LM_RULE_04_MFG_DETAILS_MISSING` | Manufacturer/Importer details missing | Critical |

        "expected": "Valid MRP with currency symbol",| `LM_RULE_05_DATE_MISSING` | Manufacturing/Import date missing | Critical |

        "actual": "None"| `LM_RULE_06_CARE_INFO_MISSING` | Consumer care details missing | Critical |

    }

]### Format and Content Validation Rules

```

| Rule ID | Description | Severity |

### Severity Levels|---------|-------------|----------|

- **Critical**: Must be fixed before product can be sold| `LM_RULE_07_MRP_FORMAT` | Invalid MRP format | High |

- **High**: Important compliance requirements| `LM_RULE_08_NET_QTY_FORMAT` | Invalid net quantity format | High |

- **Medium**: Recommended improvements| `LM_RULE_09_DATE_FUTURE` | Date is in the future | High |

| `LM_RULE_10_CARE_INFO_FORMAT` | Invalid contact format | Medium |

## 🔧 Advanced Usage

### Conditional & Category-Specific Rules

### Custom Validation

```python| Rule ID | Description | Severity |

from lmpc_checker.compliance_validator import ComplianceValidator|---------|-------------|----------|

| `LM_RULE_11_BEST_BEFORE_MISSING` | Missing expiry date for applicable categories | High |

validator = ComplianceValidator()| `LM_RULE_12_IMPORT_DATE_MISSING` | Missing import date for imported products | High |

| `LM_RULE_13_USP_MISSING` | Missing unit sale price for groceries | Medium |

# Validate specific fields only

partial_data = {"mrp": "₹500", "net_quantity": "250g"}## Validation Details

violations = validator.validate(partial_data)

### Supported MRP Formats

# Custom severity filtering- `₹1999`

critical_violations = [v for v in violations if v['severity'] == 'Critical']- `Rs. 1999`

```- `Rs.1999`

- `₹ 1999.50`

### Integration with OCR Pipeline

```python### Supported Net Quantity Units

from live_processor import LiveProcessor- Weight: `g`, `kg`

from data_refiner.refiner import DataRefiner- Volume: `ml`, `L`

from lmpc_checker.compliance_validator import ComplianceValidator- Length: `cm`, `m`

- Count: `N`, `Unit`, `Units`, `Piece`, `Pieces`

# Complete pipeline integration

processor = LiveProcessor()### Supported Date Formats

refiner = DataRefiner()- `MM/YYYY` (e.g., `08/2024`)

validator = ComplianceValidator()- `DD/MM/YYYY` (e.g., `15/08/2024`)

- `MM-YYYY` and `DD-MM-YYYY`

# Process image through pipeline- `YYYY-MM` and `YYYY-MM-DD`

ocr_result = processor.process_single_capture()

clean_data = refiner.refine(ocr_result)### Contact Information Validation

violations = validator.validate(clean_data)- **Email**: Standard email format validation

- **Phone**: Indian phone numbers (10 digits starting with 6-9, with optional +91 country code)

# Generate compliance report

compliance_status = "COMPLIANT" if not violations else "NON_COMPLIANT"### Category-Specific Rules

print(f"Compliance Status: {compliance_status}")- **Groceries**: Require best before date and unit sale price

```- **Personal Care**: Require best before date

- **Medicine**: Require best before date

## 📈 Validation Examples

## Running Tests

### ✅ Compliant Product

```pythonExecute the test suite to see various validation scenarios:

compliant_product = {

    "product_id": "SKU-FOOD-001",```bash

    "category": "Food",python test_examples.py

    "manufacturer_details": "GoodFood Ltd, Food Park, Chennai-600001",```

    "net_quantity": "500g",

    "mrp": "₹150.00",Run the main demonstration:

    "country_of_origin": "India",

    "date_of_manufacture": "08/2024",```bash

    "consumer_care": "1800-GOOD-FOOD, care@goodfood.com"python main.py

}```



violations = validator.validate(compliant_product)## Integration with FastAPI

# Result: [] (no violations)

```This validator can be easily integrated into a FastAPI backend:



### ❌ Non-Compliant Product```python

```pythonfrom fastapi import FastAPI

non_compliant_product = {from compliance_validator import ComplianceValidator

    "product_id": "SKU-INVALID-001",

    "category": "Electronics",app = FastAPI()

    # Missing required fieldsvalidator = ComplianceValidator()

    "mrp": "0",  # Invalid zero MRP

    "net_quantity": "invalid unit",  # Invalid format@app.post("/validate-product")

    "consumer_care": "123"  # Invalid phone formatasync def validate_product(product_data: dict):

}    violations = validator.validate(product_data)

    return {

violations = validator.validate(non_compliant_product)        "is_compliant": len(violations) == 0,

# Result: Multiple violations for missing and invalid fields        "violations": violations,

```        "total_violations": len(violations)

    }

## 🧪 Testing```



### Run Tests## Legal Metrology Context

```python

# Run the demonstration scriptThis implementation is based on the Indian Legal Metrology (Packaged Commodities) Rules, 2011, which mandates specific declarations on packaged goods sold in India. The rules ensure consumer protection by requiring clear, accurate product information.

python lmpc_checker/main.py

## Contributing

# The script includes comprehensive test cases:

# - Compliant product examplesThe rule engine is designed to be extensible. To add new rules:

# - Various violation scenarios  

# - Edge cases and boundary conditions1. Add a new rule tuple to the `self.rules` list in `ComplianceValidator.__init__()`

```2. Implement any required helper validation functions

3. Add test cases to verify the new rule

### Test Cases Included

- ✅ **Fully Compliant Products**: All requirements met## License

- ❌ **Missing Critical Fields**: MRP, net quantity, origin, etc.

- ⚠️ **Format Violations**: Invalid phone numbers, dates, unitsThis project is intended for educational and compliance purposes. Ensure you understand current legal requirements before using in production.
- 🔍 **Edge Cases**: Empty strings, None values, special characters

## 🔗 Integration Points

### With OCR Pipeline
```python
# data_refiner/refiner.py already outputs compatible format
clean_data = refiner.refine(ocr_output)
violations = validator.validate(clean_data)
```

### With E-commerce Platforms
```python
# Product catalog validation
for product in product_catalog:
    violations = validator.validate(product)
    if violations:
        mark_for_review(product, violations)
```

### With Quality Assurance
```python
# Batch validation for quality control
def validate_product_batch(products):
    results = []
    for product in products:
        violations = validator.validate(product)
        results.append({
            'product_id': product.get('product_id'),
            'compliant': len(violations) == 0,
            'violations': violations
        })
    return results
```

## ⚖️ Legal Compliance Notes

This validator implements rules based on:
- **Legal Metrology (Packaged Commodities) Rules, 2011**
- **Consumer Protection Act, 2019**
- **Bureau of Indian Standards (BIS) guidelines**

**Important**: This tool is for compliance assistance only. Always consult with legal experts for authoritative compliance validation.

## 🛠️ Configuration

The validation rules are configurable in `compliance_validator.py`. You can:
- Modify validation patterns
- Add custom rules
- Adjust severity levels
- Configure field requirements

## 📞 Support

For issues specific to the Legal Metrology Compliance Validator:
- Check the main repository issues: [GitHub Issues](https://github.com/yaswanthsetty/legal-metrology-ocr-pipeline/issues)
- Review the validator logic in `compliance_validator.py`
- Run `python main.py` for comprehensive test examples

---

**⚖️ Ensuring Legal Metrology compliance for Indian retail and e-commerce**