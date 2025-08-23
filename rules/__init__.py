"""Rules module for diabetes management application."""
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

def load_rules() -> Dict[str, Any]:
    """Load and return the rules from rules.json.
    
    Returns:
        Dict containing the loaded rules.
    """
    rules_path = Path(__file__).parent / 'rules.json'
    try:
        with open(rules_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Rules file not found at {rules_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing rules file: {e}")

def get_traffic_light_status(metric_name: str, value: float, rules: Dict) -> str:
    """Get traffic light status for a given metric and value.
    
    Args:
        metric_name: Name of the metric (e.g., 'hba1c', 'bp_sys', 'bmi', 'ldl')
        value: Numeric value to evaluate
        rules: Dictionary containing the rules
        
    Returns:
        str: 'red', 'amber', or 'green'
    """
    if not rules or 'traffic' not in rules or metric_name not in rules['traffic']:
        return 'green'  # Default to green if no thresholds defined
        
    thresholds = rules['traffic'].get(metric_name, {})
    
    if value <= thresholds.get('green_max', float('inf')):
        return 'green'
    elif value <= thresholds.get('amber_max', float('inf')):
        return 'amber'
    else:
        return 'red'

def get_traffic_light_emoji(status: str) -> str:
    """Get emoji for a traffic light status.
    
    Args:
        status: Traffic light status ('red', 'amber', or 'green')
        
    Returns:
        str: Emoji representation of the status
    """
    emoji_map = {
        'red': '🔴',
        'amber': '🟠',
        'green': '🟢'
    }
    return emoji_map.get(status.lower(), '⚪')

# Make functions available at the package level
__all__ = ['load_rules', 'get_traffic_light_status', 'get_traffic_light_emoji']
