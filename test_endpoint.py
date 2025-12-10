#!/usr/bin/env python3
"""
Simple script to test the /generate-report-custom endpoint
"""

import requests
import json

# Configuration
API_URL = "http://98.70.41.227:8004/generate-report-custom"

# Test payload
payload = {
    # template_id removed - endpoint now uses general_medical_analysis_001.json by default
    "template_parameters": ["FINDINGS", "IMPRESSION"],
    "region": "general",  # Updated to match the new template location
    "user_findings": "normal",
    "patient_metadata": {
        "PATIENT_SEX": "M",
        "PATIENT_AGE": "34",
        "BODY_PART": "CHEST",
        "MODALITY": "X-Ray",
        "VIEW_POSITION": "AP"
    }
}


print("=" * 80)
print("Testing Custom Report Generation Endpoint")
print("=" * 80)
print(f"\nEndpoint: {API_URL}")
print(f"\nRequest:")
print(json.dumps(payload, indent=2))
print("\n" + "=" * 80)
print("Sending request...\n")

try:
    response = requests.post(API_URL, json=payload, timeout=120)
    
    print(f"Status Code: {response.status_code}\n")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS!")
        print(f"\nStatus: {data.get('status')}")
        print("\n" + "-" * 80)
        print("FULL REPORT:")
        print("-" * 80)
        print(data.get('full_report', ''))
        print("\n" + "-" * 80)
        print("MEDICAL CONTENT:")
        print("-" * 80)
        print(data.get('medical_content', ''))
    else:
        print("❌ ERROR!")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Connection Error: Could not connect to the backend.")
    print("   Make sure the backend is running on http://localhost:8004")
except requests.exceptions.Timeout:
    print("❌ Timeout: Request took too long (model generation)")
except Exception as e:
    print(f"❌ Error: {e}")

