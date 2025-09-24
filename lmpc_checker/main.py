"""
Main demonstration script for the Legal Metrology Compliance Validator.

This script demonstrates the usage of the ComplianceValidator class
with the exact example provided in the requirements.
"""

from compliance_validator import ComplianceValidator


def main():
    """Main function demonstrating the ComplianceValidator usage."""
    print("Legal Metrology Compliance Validator")
    print("=" * 40)
    print()
    
    # Create validator instance
    validator = ComplianceValidator()
    
    # Example from the prompt
    print("Example 1: Product from prompt requirements")
    print("-" * 45)
    
    product_data = {
        "product_id": "SKU-ELEC-001",
        "category": "Electronics",
        "manufacturer_details": "Noise, Gurugram, Haryana, 122001",
        "importer_details": None,
        "net_quantity": "1 Unit",
        "mrp": "₹1999",
        "unit_sale_price": None,
        "country_of_origin": "India",
        "date_of_manufacture": "08/2025",  # Future date - will trigger violation
        "date_of_import": None,
        "best_before_date": None,
        "consumer_care": "help@go-noise.com, +91 88821 32132",
        "dimensions": "4.6 cm (1.83 inch) HD Display",
        "contents": "1N Smartwatch, 1N Charging Cable, 1N Manual"
    }
    
    violations = validator.validate(product_data)
    
    print(f"Product ID: {product_data['product_id']}")
    print(f"Category: {product_data['category']}")
    print(f"Violations found: {len(violations)}")
    print()
    
    if violations:
        print("Detected violations:")
        for i, violation in enumerate(violations, 1):
            print(f"{i}. Rule ID: {violation['rule_id']}")
            print(f"   Description: {violation['description']}")
            print(f"   Field: {violation['violating_field']}")
            print(f"   Severity: {violation['severity']}")
            print()
    else:
        print("✅ No violations detected - product is compliant!")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 2: Product with missing MRP (from prompt example)
    print("Example 2: Product with missing MRP")
    print("-" * 35)
    
    product_data_2 = {
        "product_id": "SKU-TEST-002",
        "category": "Electronics",
        "manufacturer_details": "Test Company",
        "importer_details": None,
        "net_quantity": "200g",
        "mrp": None,  # Missing MRP
        "unit_sale_price": None,
        "country_of_origin": "India",
        "date_of_manufacture": "08/2024",
        "date_of_import": None,
        "best_before_date": None,
        "consumer_care": "test@company.com",
        "dimensions": None,
        "contents": None
    }
    
    violations_2 = validator.validate(product_data_2)
    
    print(f"Product ID: {product_data_2['product_id']}")
    print(f"Violations found: {len(violations_2)}")
    print()
    
    if violations_2:
        print("Detected violations:")
        for i, violation in enumerate(violations_2, 1):
            print(f"{i}. {violation}")
            print()
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 3: Grocery product with multiple issues
    print("Example 3: Grocery product with multiple compliance issues")
    print("-" * 55)
    
    grocery_data = {
        "product_id": "SKU-FOOD-003",
        "category": "Groceries",
        "manufacturer_details": None,  # Missing
        "importer_details": None,      # Missing
        "net_quantity": "invalid weight",  # Invalid format
        "mrp": "1000 rupees",          # Invalid format
        "unit_sale_price": None,       # Missing for groceries
        "country_of_origin": None,     # Missing
        "date_of_manufacture": "12/2026",  # Future date
        "date_of_import": None,
        "best_before_date": None,      # Missing for groceries
        "consumer_care": "invalid-contact",  # Invalid format
        "dimensions": None,
        "contents": None
    }
    
    violations_3 = validator.validate(grocery_data)
    
    print(f"Product ID: {grocery_data['product_id']}")
    print(f"Category: {grocery_data['category']}")
    print(f"Violations found: {len(violations_3)}")
    print()
    
    if violations_3:
        print("Detected violations (by severity):")
        
        # Group by severity
        critical = [v for v in violations_3 if v['severity'] == 'critical']
        high = [v for v in violations_3 if v['severity'] == 'high']
        medium = [v for v in violations_3 if v['severity'] == 'medium']
        
        if critical:
            print("\n🔴 CRITICAL violations:")
            for v in critical:
                print(f"   • {v['rule_id']}: {v['description']}")
        
        if high:
            print("\n🟡 HIGH severity violations:")
            for v in high:
                print(f"   • {v['rule_id']}: {v['description']}")
        
        if medium:
            print("\n🟠 MEDIUM severity violations:")
            for v in medium:
                print(f"   • {v['rule_id']}: {v['description']}")
    
    print("\n" + "=" * 60)
    print("✅ Compliance validation complete!")


if __name__ == "__main__":
    main()