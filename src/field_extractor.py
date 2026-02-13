"""
Binary Brain - Field Extraction Engine
Extracts structured fields from OCR results using:
- Fuzzy Matching for Dealer Name (>=90% match)
- Exact Match for Model Name
- Regex/Rule Engine for numeric fields (Horse Power, Asset Cost)
- Key-Value pair detection
"""

import re
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz, process


# Known dealer names database (can be expanded)
KNOWN_DEALERS = [
    "Agri Machinery", "Mahindra Tractors", "Sonalika International",
    "TAFE", "John Deere India", "Escorts Limited", "New Holland",
    "Kubota India", "Force Motors", "Preet Tractors", "Indo Farm",
    "Swaraj Tractors", "Eicher Tractors", "Farmtrac", "Captain Tractors",
    "VST Tillers", "ACE Tractors", "Powertrac", "Digitrac",
    "Massey Ferguson", "Standard Tractors", "HMT Tractors",
    "Punjab Tractors", "Balwan Tractors", "Kartar Tractors",
]

# Known tractor model patterns
KNOWN_MODELS = [
    "Sonali 550", "Sonali 590", "Sonali 750", "Sonali 60",
    "DI 745", "DI 750", "Arjun 605", "Arjun Novo 605",
    "Sarpanch 5900", "Tiger 60", "Yuvo 575", "Jivo 305",
    "RX Series", "Bhoomiputra", "Shaktimaan",
    "Model YS 75", "Model YS 50", "Model RS 40",
    "575 DI", "475 DI", "275 DI", "241 DI",
]

# Labels that indicate field positions
FIELD_LABELS = {
    'dealer_name': [
        'dealer name', 'dealer', 'vendor name', 'vendor', 'seller',
        'seller name', 'supplier', 'dealer/seller', 'sold by',
        'authorized dealer', 'dealership', 'from'
    ],
    'model_name': [
        'model name', 'model', 'tractor model', 'product name',
        'product', 'item', 'description', 'particular', 'vehicle model',
        'model no', 'model number'
    ],
    'horse_power': [
        'horse power', 'hp', 'h.p.', 'horsepower', 'power',
        'engine power', 'rated power', 'engine hp', 'bhp'
    ],
    'asset_cost': [
        'asset cost', 'total amount', 'total', 'amount', 'price',
        'cost', 'net amount', 'grand total', 'invoice amount',
        'total price', 'ex-showroom', 'showroom price', 'rate',
        'quotation amount', 'quoted price', 'basic price'
    ]
}


class FieldExtractor:
    """Extracts structured fields from OCR text results."""

    def __init__(self, known_dealers: list = None, known_models: list = None):
        self.known_dealers = known_dealers or KNOWN_DEALERS
        self.known_models = known_models or KNOWN_MODELS

    def extract_all_fields(self, ocr_results: List[Dict],
                           full_text: str = None) -> Dict:
        """
        Extract all fields from OCR results.

        Args:
            ocr_results: List of OCR result dicts with 'text', 'bbox', 'confidence'
            full_text: Full concatenated text (optional)

        Returns:
            Dict with extracted fields and confidence scores
        """
        if full_text is None:
            full_text = '\n'.join([r['text'] for r in ocr_results])

        result = {
            'dealerName': self._extract_dealer_name(ocr_results, full_text),
            'modelName': self._extract_model_name(ocr_results, full_text),
            'horsePower': self._extract_horse_power(ocr_results, full_text),
            'assetCost': self._extract_asset_cost(ocr_results, full_text),
        }

        return result

    def _extract_dealer_name(self, ocr_results: List[Dict],
                             full_text: str) -> Dict:
        """Extract dealer name using fuzzy matching (>=90%)."""
        # Method 1: Look for text near "dealer" label
        value = self._find_value_near_label(ocr_results, FIELD_LABELS['dealer_name'])

        if value:
            # Fuzzy match against known dealers
            match = process.extractOne(value, self.known_dealers, scorer=fuzz.ratio)
            if match and match[1] >= 70:
                return {
                    'value': match[0],
                    'raw_value': value,
                    'confidence': round(match[1] / 100, 4),
                    'method': 'fuzzy_match_label'
                }
            return {
                'value': value,
                'raw_value': value,
                'confidence': 0.7,
                'method': 'label_extraction'
            }

        # Method 2: Search full text for known dealers
        for dealer in self.known_dealers:
            match_score = fuzz.partial_ratio(dealer.lower(), full_text.lower())
            if match_score >= 90:
                return {
                    'value': dealer,
                    'raw_value': dealer,
                    'confidence': round(match_score / 100, 4),
                    'method': 'fuzzy_search'
                }

        # Method 3: Look for company-like names in text
        company_patterns = [
            r'(?:M/s\.?\s*)?([A-Z][a-zA-Z\s]+(?:Machinery|Tractors?|Motors?|'
            r'Enterprises?|Industries?|Farm|Agri|Ltd\.?|Pvt\.?))',
            r'(?:dealer|vendor|seller)[\s:]+([A-Za-z\s]+?)(?:\n|$)',
        ]
        for pattern in company_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                name = matches[0].strip()
                if len(name) > 3:
                    return {
                        'value': name,
                        'raw_value': name,
                        'confidence': 0.6,
                        'method': 'pattern_match'
                    }

        return {'value': None, 'confidence': 0, 'method': 'not_found'}

    def _extract_model_name(self, ocr_results: List[Dict],
                            full_text: str) -> Dict:
        """Extract model name using exact matching."""
        # Method 1: Look for text near "model" label
        value = self._find_value_near_label(ocr_results, FIELD_LABELS['model_name'])

        if value:
            # Exact match against known models
            for model in self.known_models:
                if model.lower() in value.lower() or value.lower() in model.lower():
                    return {
                        'value': model,
                        'raw_value': value,
                        'confidence': 0.95,
                        'method': 'exact_match_label'
                    }
            # Fuzzy match
            match = process.extractOne(value, self.known_models, scorer=fuzz.ratio)
            if match and match[1] >= 80:
                return {
                    'value': match[0],
                    'raw_value': value,
                    'confidence': round(match[1] / 100, 4),
                    'method': 'fuzzy_match_label'
                }
            return {
                'value': value,
                'raw_value': value,
                'confidence': 0.7,
                'method': 'label_extraction'
            }

        # Method 2: Search full text for known models
        for model in self.known_models:
            if model.lower() in full_text.lower():
                return {
                    'value': model,
                    'raw_value': model,
                    'confidence': 0.95,
                    'method': 'exact_search'
                }

        # Method 3: Pattern matching for model-like strings
        model_patterns = [
            r'(?:model|tractor)[\s:]+([A-Za-z]+\s*\d+\s*[A-Za-z]*)',
            r'\b([A-Z][a-z]+\s+\d{2,4})\b',
            r'\b(\d{3,4}\s*(?:DI|HP|HT))\b',
        ]
        for pattern in model_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                return {
                    'value': matches[0].strip(),
                    'raw_value': matches[0].strip(),
                    'confidence': 0.6,
                    'method': 'pattern_match'
                }

        return {'value': None, 'confidence': 0, 'method': 'not_found'}

    def _extract_horse_power(self, ocr_results: List[Dict],
                             full_text: str) -> Dict:
        """Extract horse power using regex rules."""
        # Method 1: Find value near HP label
        value = self._find_value_near_label(ocr_results, FIELD_LABELS['horse_power'])
        if value:
            hp_val = self._extract_number(value)
            if hp_val and 10 <= hp_val <= 500:
                return {
                    'value': str(int(hp_val)),
                    'raw_value': value,
                    'confidence': 0.9,
                    'method': 'label_extraction'
                }

        # Method 2: Regex patterns for HP
        hp_patterns = [
            r'(\d{2,3})\s*(?:HP|hp|H\.P\.|h\.p\.|horse\s*power|bhp|BHP)',
            r'(?:HP|hp|H\.P\.|horse\s*power|bhp|BHP)[\s:]*(\d{2,3})',
            r'(?:power|engine)[\s:]*(\d{2,3})\s*(?:HP|hp)?',
            r'(\d{2,3})\s*(?:\u0905\u0936\u094d\u0935\u0936\u0915\u094d\u0924\u093f|HP)',  # Hindi
        ]
        for pattern in hp_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                hp_val = int(matches[0])
                if 10 <= hp_val <= 500:
                    return {
                        'value': str(hp_val),
                        'raw_value': matches[0],
                        'confidence': 0.85,
                        'method': 'regex_match'
                    }

        return {'value': None, 'confidence': 0, 'method': 'not_found'}

    def _extract_asset_cost(self, ocr_results: List[Dict],
                            full_text: str) -> Dict:
        """Extract asset cost using regex and numeric validation."""
        # Method 1: Find value near cost/amount label
        value = self._find_value_near_label(ocr_results, FIELD_LABELS['asset_cost'])
        if value:
            amount = self._extract_amount(value)
            if amount and amount >= 10000:
                return {
                    'value': str(int(amount)),
                    'raw_value': value,
                    'confidence': 0.9,
                    'method': 'label_extraction'
                }

        # Method 2: Regex for amounts
        amount_patterns = [
            r'(?:Rs\.?|\u20b9|INR)\s*([\d,]+(?:\.\d{2})?)',
            r'(?:total|amount|cost|price)[\s:]*(?:Rs\.?|\u20b9)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:total|amount|cost|price)[\s:]*(\d[\d,]*(?:\.\d{2})?)',
            r'(\d{4,8}(?:\.\d{2})?)\s*(?:/\-|only)',
        ]
        for pattern in amount_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                for match in matches:
                    amount = self._extract_amount(match)
                    if amount and amount >= 50000:
                        return {
                            'value': str(int(amount)),
                            'raw_value': match,
                            'confidence': 0.85,
                            'method': 'regex_match'
                        }

        # Method 3: Find largest number (likely the total)
        all_numbers = re.findall(r'[\d,]+(?:\.\d{2})?', full_text)
        amounts = []
        for num_str in all_numbers:
            amount = self._extract_amount(num_str)
            if amount and amount >= 50000:
                amounts.append(amount)

        if amounts:
            max_amount = max(amounts)
            return {
                'value': str(int(max_amount)),
                'raw_value': str(max_amount),
                'confidence': 0.6,
                'method': 'max_number'
            }

        return {'value': None, 'confidence': 0, 'method': 'not_found'}

    def _find_value_near_label(self, ocr_results: List[Dict],
                               labels: List[str]) -> Optional[str]:
        """Find the value text near a label in OCR results."""
        for label in labels:
            label_lower = label.lower()
            for i, r in enumerate(ocr_results):
                text_lower = r['text'].lower()
                if label_lower in text_lower or fuzz.partial_ratio(label_lower, text_lower) >= 85:
                    # Check if value is in the same text block
                    remaining = text_lower.replace(label_lower, '').strip()
                    remaining = re.sub(r'^[\s:]+', '', remaining)
                    if remaining and len(remaining) > 1:
                        return r['text'].replace(label, '').strip().strip(':').strip()

                    # Look at next text blocks (right or below)
                    bbox = r['bbox']
                    candidates = []

                    for j, r2 in enumerate(ocr_results):
                        if j == i:
                            continue
                        b2 = r2['bbox']

                        # Right of label (same row)
                        if (b2[0] > bbox[2] - 20 and
                                abs(b2[1] - bbox[1]) < 30 and
                                b2[0] - bbox[2] < 500):
                            dist = b2[0] - bbox[2]
                            candidates.append((r2['text'], dist, 'right'))

                        # Below label
                        if (b2[1] > bbox[3] - 10 and
                                abs(b2[0] - bbox[0]) < 150 and
                                b2[1] - bbox[3] < 100):
                            dist = b2[1] - bbox[3]
                            candidates.append((r2['text'], dist, 'below'))

                    if candidates:
                        candidates.sort(key=lambda c: c[1])
                        return candidates[0][0]

        return None

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a number from text."""
        text = text.replace(',', '').replace(' ', '')
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        return None

    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract monetary amount from text, removing currency symbols."""
        if isinstance(text, (int, float)):
            return float(text)
        text = re.sub(r'[\u20b9$\u20acRs\.INR,\s]', '', str(text))
        match = re.search(r'(\d+(?:\.\d{1,2})?)', text)
        if match:
            return float(match.group(1))
        return None
