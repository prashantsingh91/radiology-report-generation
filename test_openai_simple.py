#!/usr/bin/env python3
"""
Simple test script for OpenAI/GPT Report Generation API
"""

import requests
import json

# API endpoint - Using public IP
PUBLIC_IP = "98.70.41.227"
URL = f"http://{PUBLIC_IP}:8004/fill-template"

# Sample request data
patient_metadata = {
    'PATIENT_SEX': 'M',
    'PATIENT_AGE': '45',
    'BODY_PART': '',
    'MODALITY': 'XRAY',
    'VIEW_POSITION': 'AP'
}

form_data = {
    'findings': 'Patchy homogeneous opacity obscuring vessels with air bronchogram noted in bilateral upper lung zone and right paracardiac region S/O- consolidation. Impression- Above X-ray Findings are suggestive of consolidation. ADVICE :- Complete Blood Count.',
    'template': '''XRAY CHEST AP VIEW

OBSERVATIONS ;

l The trachea is central .

l Cardiophrenic and costophrenic angles are normal.

l The mediastinal and cardiac silhoutte are normal.

l Both hila are normal.

l Cardiothoracic ratio is normal.

l Bones of the thoracic cage are normal.

l Soft tissues of the chest wall are normal.

IMPRESSION :- Normal study.

ADVICE :- NA.''',
    'model': 'gpt-5.1',
    'patient_metadata': json.dumps(patient_metadata)
}

print("=" * 60)
print("Testing OpenAI Report Generation API")
print("=" * 60)
print(f"\nEndpoint: {URL}")
print(f"\nRequest Data:")
print(f"  Findings: {form_data['findings']}")
print(f"  Model: {form_data['model']}")
print(f"  Patient Metadata: {patient_metadata}")
print(f"\nSending request...\n")

try:
    # Send request
    response = requests.post(URL, data=form_data, timeout=120)
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS!")
        print(f"\nModel Used: {result['openai_model']}")
        print(f"Status: {result['status']}")
        print(f"\nGenerated Report:")
        print("-" * 60)
        print(result['report'])
        print("-" * 60)
    else:
        print(f"❌ ERROR: Status {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
