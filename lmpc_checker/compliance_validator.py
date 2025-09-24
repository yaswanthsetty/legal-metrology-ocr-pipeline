"""
Compliance Rule Engine for Legal Metrology

This module implements a Python-based rule engine for validating product data
against the Indian Legal Metrology (Packaged Commodities) Rules, 2011.

This implementation uses direct Python validation functions for clarity and reliability,
while maintaining the structure that could be adapted to use rule-engine library.
"""

import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union


class ComplianceValidator:
    """
    A compliance validator that validates product data against Legal Metrology rules.
    """
    
    def __init__(self):
        """
        Initialize the ComplianceValidator with all validation rules.
        """
        # Define all rules as tuples (rule_id, description, field, severity, validation_function)
        self.rules = [
            # Mandatory Field Presence Rules
            (
                "LM_RULE_01_MRP_MISSING",
                "The Maximum Retail Price (MRP) is a mandatory declaration and is missing.",
                "mrp",
                "critical",
                lambda data: self._is_none_or_empty(data.get("mrp"))
            ),
            (
                "LM_RULE_02_NET_QTY_MISSING",
                "Net Quantity is a mandatory declaration and is missing.",
                "net_quantity",
                "critical",
                lambda data: self._is_none_or_empty(data.get("net_quantity"))
            ),
            (
                "LM_RULE_03_ORIGIN_MISSING",
                "Country of Origin is a mandatory declaration and is missing.",
                "country_of_origin",
                "critical",
                lambda data: self._is_none_or_empty(data.get("country_of_origin"))
            ),
            (
                "LM_RULE_04_MFG_DETAILS_MISSING",
                "Manufacturer or Importer details are missing (at least one is required).",
                "manufacturer_details",
                "critical",
                lambda data: self._is_none_or_empty(data.get("manufacturer_details")) and self._is_none_or_empty(data.get("importer_details"))
            ),
            (
                "LM_RULE_05_DATE_MISSING",
                "Manufacturing or Import date is missing (at least one is required).",
                "date_of_manufacture",
                "critical",
                lambda data: self._is_none_or_empty(data.get("date_of_manufacture")) and self._is_none_or_empty(data.get("date_of_import"))
            ),
            (
                "LM_RULE_06_CARE_INFO_MISSING",
                "Consumer care details are a mandatory declaration and are missing.",
                "consumer_care",
                "critical",
                lambda data: self._is_none_or_empty(data.get("consumer_care"))
            ),
            
            # Format and Content Validation Rules
            (
                "LM_RULE_07_MRP_FORMAT",
                "MRP does not follow the correct format (should start with ₹ or Rs. followed by a number).",
                "mrp",
                "high",
                lambda data: not self._is_none_or_empty(data.get("mrp")) and not self._is_valid_mrp_format(data.get("mrp"))
            ),
            (
                "LM_RULE_08_NET_QTY_FORMAT",
                "Net Quantity does not contain a valid unit (g, kg, ml, L, cm, m, N, Unit).",
                "net_quantity",
                "high",
                lambda data: not self._is_none_or_empty(data.get("net_quantity")) and not self._is_valid_net_quantity_format(data.get("net_quantity"))
            ),
            (
                "LM_RULE_09_DATE_FUTURE",
                "Manufacturing or import date cannot be in the future.",
                "date_of_manufacture",
                "high",
                lambda data: (not self._is_none_or_empty(data.get("date_of_manufacture")) and self._is_future_date(data.get("date_of_manufacture"))) or (not self._is_none_or_empty(data.get("date_of_import")) and self._is_future_date(data.get("date_of_import")))
            ),
            (
                "LM_RULE_10_CARE_INFO_FORMAT",
                "Consumer care details must contain a valid email or phone number.",
                "consumer_care",
                "medium",
                lambda data: not self._is_none_or_empty(data.get("consumer_care")) and not self._has_valid_care_info(data.get("consumer_care"))
            ),
            
            # Conditional & Category-Specific Rules
            (
                "LM_RULE_11_BEST_BEFORE_MISSING",
                "Best Before/Expiry date is missing for a category that requires it.",
                "best_before_date",
                "high",
                lambda data: self._is_category_requiring_best_before(data.get("category")) and self._is_none_or_empty(data.get("best_before_date"))
            ),
            (
                "LM_RULE_12_IMPORT_DATE_MISSING",
                "Imported product is missing the date of import.",
                "date_of_import",
                "high",
                lambda data: not self._is_none_or_empty(data.get("importer_details")) and self._is_none_or_empty(data.get("date_of_import"))
            ),
            (
                "LM_RULE_13_USP_MISSING",
                "Unit Sale Price is missing for grocery items where it is required.",
                "unit_sale_price",
                "medium",
                lambda data: data.get("category") == "Groceries" and not self._is_none_or_empty(data.get("net_quantity")) and self._is_none_or_empty(data.get("unit_sale_price"))
            ),
        ]

    def _is_none_or_empty(self, value: Any) -> bool:
        """Check if a value is None or an empty string."""
        return value is None or (isinstance(value, str) and value.strip() == "")

    def _is_valid_email(self, email: str) -> bool:
        """Validate email format using regex."""
        if not email or not isinstance(email, str):
            return False
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.search(email_pattern, email))

    def _is_valid_phone(self, phone: str) -> bool:
        """Validate Indian phone number format using regex."""
        if not phone or not isinstance(phone, str):
            return False
        # Remove spaces, hyphens, and plus signs for validation
        cleaned_phone = re.sub(r'[\s\-\+]', '', phone)
        # Check for 10-digit number or 10-digit with country code (+91)
        phone_patterns = [
            r'^[6-9]\d{9}$',  # 10-digit starting with 6-9
            r'^91[6-9]\d{9}$',  # With country code 91
        ]
        return any(re.match(pattern, cleaned_phone) for pattern in phone_patterns)

    def _is_valid_mrp_format(self, mrp: str) -> bool:
        """Validate MRP format (₹ or Rs. followed by a number)."""
        if not mrp or not isinstance(mrp, str):
            return False
        mrp_pattern = r'^(₹|Rs\.?\s*)\s*\d+(\.\d{1,2})?$'
        return bool(re.match(mrp_pattern, mrp.strip()))

    def _is_valid_net_quantity_format(self, net_quantity: str) -> bool:
        """Validate net quantity format (number + valid unit)."""
        if not net_quantity or not isinstance(net_quantity, str):
            return False
        # Valid units: g, kg, ml, L, cm, m, N, Unit
        quantity_pattern = r'\d+(\.\d+)?\s*(g|kg|ml|L|cm|m|N|Unit|Units|Piece|Pieces)\b'
        return bool(re.search(quantity_pattern, net_quantity, re.IGNORECASE))

    def _is_future_date(self, date_str: str) -> bool:
        """Check if a date string represents a future date."""
        if not date_str or not isinstance(date_str, str):
            return False
        
        try:
            # Try different date formats
            date_formats = [
                '%m/%Y',      # MM/YYYY
                '%d/%m/%Y',   # DD/MM/YYYY
                '%m-%Y',      # MM-YYYY
                '%d-%m-%Y',   # DD-MM-YYYY
                '%Y-%m',      # YYYY-MM
                '%Y-%m-%d',   # YYYY-MM-DD
            ]
            
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str.strip(), fmt)
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                return False  # Could not parse date
            
            # Compare with current date
            current_date = datetime.now()
            return parsed_date > current_date
            
        except Exception:
            return False

    def _is_category_requiring_best_before(self, category: str) -> bool:
        """Check if category requires best before date."""
        if not category or not isinstance(category, str):
            return False
        requiring_categories = ['Groceries', 'Personal Care', 'Medicine']
        return category in requiring_categories

    def _has_valid_care_info(self, care_info: str) -> bool:
        """Check if consumer care info contains valid email or phone."""
        if not care_info or not isinstance(care_info, str):
            return False
        
        # Check if the string contains an email
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        has_email = bool(re.search(email_pattern, care_info))
        
        # Check for phone patterns within the text - more flexible
        phone_patterns = [
            r'\+91[\s\-]?[6-9]\d{9}',  # +91 followed by 10 digits
            r'91[\s\-]?[6-9]\d{9}',    # 91 followed by 10 digits  
            r'[6-9]\d[\s\-]?\d{3}[\s\-]?\d{5}',  # 10 digit number with separators
            r'[6-9]\d{9}',             # Simple 10 digit number
        ]
        
        has_phone = any(re.search(pattern, care_info) for pattern in phone_patterns)
        
        return has_email or has_phone

    def validate(self, product_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Validate product data against all Legal Metrology rules.
        
        Args:
            product_data: Dictionary containing product information
            
        Returns:
            List of violation dictionaries, empty list if no violations
        """
        violations = []
        
        for rule_id, description, field, severity, validation_func in self.rules:
            try:
                # Execute the validation function
                if validation_func(product_data):
                    violation = {
                        "rule_id": rule_id,
                        "description": description,
                        "violating_field": field,
                        "severity": severity
                    }
                    violations.append(violation)
            except Exception as e:
                print(f"Error evaluating rule {rule_id}: {e}")
        
        return violations