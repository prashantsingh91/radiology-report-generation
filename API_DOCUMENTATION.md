# MedGemma Report Generation API Documentation

## Endpoint 1: Generate Radiology Report (Full Template)

### URL
```
POST http://localhost:8004/generate-report
```

This endpoint generates a report using all AI-generated sections from the specified template.

---

## Endpoint 2: Generate Radiology Report (Custom Sections)

### URL
```
POST http://localhost:8004/generate-report-custom
```

This endpoint allows you to specify which sections to generate from the template using `template_parameters`.

### Request Headers
```
Content-Type: application/json
```

### Request Body Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `template_id` | string | Yes | The ID of the template to use (e.g., "chest_xray_standard_001") |
| `region` | string | Yes | The body region/category (e.g., "chest", "brain", "abdomen", "spine") |
| `user_findings` | string | Yes | Radiologist observations/findings (e.g., "Bronchitis", "normal", "bilateral lower lobe opacities") |
| `patient_metadata` | object | No | Patient and DICOM metadata (see structure below) |
| `max_length` | integer | No | Maximum tokens for generation (default: 1024, max: 600) |

### Patient Metadata Structure (Optional)

The `patient_metadata` object can contain the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `PATIENT_SEX` | string | Patient's sex (e.g., "M", "F") |
| `PATIENT_AGE` | string | Patient's age (e.g., "34", "45 years") |
| `PATIENT_NAME` | string | Patient's name |
| `PATIENT_ID` | string | Patient's ID |
| `BODY_PART` | string | Body part examined (e.g., "CHEST", "BRAIN") |
| `MODALITY` | string | Imaging modality (e.g., "X-Ray", "CT", "MRI") |
| `VIEW_POSITION` | string | View position (e.g., "AP", "PA", "LATERAL") |
| `STUDY_DATE` | string | Study date (e.g., "2024-11-19") |
| `REFERRING_PHYSICIAN` | string | Referring physician name |

### Response Structure

```json
{
  "medical_content": "string - Post-processed medical content",
  "full_report": "string - Complete formatted report with metadata",
  "status": "string - Status (e.g., 'success')",
  "raw_prompt": "string - The exact prompt sent to the model",
  "raw_output": "string - Raw model output before post-processing"
}
```

### Available Templates

#### Chest Region
- `chest_xray_standard_001` - Standard format with Clinical Details, Findings, Impression, Recommendations
- `chest_xray_compact_001` - Compact format with Clinical Details, Findings, Impression only

#### Other Regions
- `abdomen/` - Abdominal imaging templates
- `brain/` - Brain imaging templates
- `spine/` - Spine imaging templates

### Example Requests

#### Example 1: Basic Request (Minimal)
```json
{
  "template_id": "chest_xray_standard_001",
  "region": "chest",
  "user_findings": "Bronchitis"
}
```

#### Example 2: Request with Patient Metadata
```json
{
  "template_id": "chest_xray_standard_001",
  "region": "chest",
  "user_findings": "Bilateral lower lobe opacities",
  "patient_metadata": {
    "PATIENT_SEX": "M",
    "PATIENT_AGE": "34",
    "PATIENT_NAME": "John Doe",
    "PATIENT_ID": "P12345",
    "BODY_PART": "CHEST",
    "MODALITY": "X-Ray",
    "VIEW_POSITION": "AP",
    "STUDY_DATE": "2024-11-19",
    "REFERRING_PHYSICIAN": "Dr. Smith"
  },
  "max_length": 600
}
```

#### Example 3: Normal Report
```json
{
  "template_id": "chest_xray_compact_001",
  "region": "chest",
  "user_findings": "normal"
}
```

### Example Response

```json
{
  "medical_content": "CLINICAL DETAILS:\nPatient presents with symptoms of bronchitis...\n\nFINDINGS:\nBilateral lower lobe opacities are noted...\n\nIMPRESSION:\nBilateral lower lobe pneumonia...\n\nRECOMMENDATIONS:\nFollow-up chest X-ray recommended in 2 weeks.",
  "full_report": "RADIOLOGY REPORT\n\nPatient Name: John Doe\nPatient ID: P12345\nAge: 34\nSex: M\nStudy Date: 2024-11-19\nModality: X-Ray\nReferring Physician: Dr. Smith\n\nCLINICAL DETAILS:\nPatient presents with symptoms of bronchitis...\n\nFINDINGS:\nBilateral lower lobe opacities are noted...\n\nIMPRESSION:\nBilateral lower lobe pneumonia...\n\nRECOMMENDATIONS:\nFollow-up chest X-ray recommended in 2 weeks.\n\n---\nReport generated on 2024-11-19 10:39:01",
  "status": "success",
  "raw_prompt": "You are an expert radiologist assistant.\n\nPatient Information (for reference only – DO NOT generate this as a section):\nPatient's Sex: M\nPatient's Age: 34\nBody Part Examined: CHEST\nModality: X-Ray\nView Position: AP\n\nRadiologist Observations (this is your only clinical input; base your report ONLY on this):\nBilateral lower lobe opacities\n\nYou must generate the following sections in this exact order:\nCLINICAL DETAILS:, FINDINGS:, IMPRESSION:, RECOMMENDATIONS:\n\nFor the FINDINGS section:\n- Expand on the Radiologist Observations in detail, providing a comprehensive description\n- Use the Radiologist Observations as the foundation and build upon them\n- Do NOT invent new observations beyond what is stated in the Radiologist Observations\n- If the Radiologist Observations indicate 'normal' or no abnormalities, write a detailed normal report describing the normal appearance of all relevant anatomical structures (lungs, heart, mediastinum, pleura, bones, soft tissues)\n\nGenerate ONLY the sections in order.",
  "raw_output": "CLINICAL DETAILS:\nPatient presents with symptoms of bronchitis...\n\nFINDINGS:\nBilateral lower lobe opacities are noted...\n\nIMPRESSION:\nBilateral lower lobe pneumonia...\n\nRECOMMENDATIONS:\nFollow-up chest X-ray recommended in 2 weeks."
}
```

### cURL Examples

#### Basic Request
```bash
curl -X POST http://localhost:8004/generate-report \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "chest_xray_standard_001",
    "region": "chest",
    "user_findings": "Bronchitis"
  }'
```

#### Request with Metadata
```bash
curl -X POST http://localhost:8004/generate-report \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "chest_xray_standard_001",
    "region": "chest",
    "user_findings": "Bilateral lower lobe opacities",
    "patient_metadata": {
      "PATIENT_SEX": "M",
      "PATIENT_AGE": "34",
      "BODY_PART": "CHEST",
      "MODALITY": "X-Ray",
      "VIEW_POSITION": "AP"
    },
    "max_length": 600
  }'
```

### Python Example

```python
import requests

url = "http://localhost:8004/generate-report"

payload = {
    "template_id": "chest_xray_standard_001",
    "region": "chest",
    "user_findings": "Bronchitis",
    "patient_metadata": {
        "PATIENT_SEX": "M",
        "PATIENT_AGE": "34",
        "BODY_PART": "CHEST",
        "MODALITY": "X-Ray",
        "VIEW_POSITION": "AP"
    },
    "max_length": 600
}

response = requests.post(url, json=payload)
data = response.json()

print("Status:", data["status"])
print("Full Report:")
print(data["full_report"])
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const url = 'http://localhost:8004/generate-report';

const payload = {
  template_id: 'chest_xray_standard_001',
  region: 'chest',
  user_findings: 'Bronchitis',
  patient_metadata: {
    PATIENT_SEX: 'M',
    PATIENT_AGE: '34',
    BODY_PART: 'CHEST',
    MODALITY: 'X-Ray',
    VIEW_POSITION: 'AP'
  },
  max_length: 600
};

axios.post(url, payload)
  .then(response => {
    console.log('Status:', response.data.status);
    console.log('Full Report:');
    console.log(response.data.full_report);
  })
  .catch(error => {
    console.error('Error:', error.response?.data || error.message);
  });
```

### Error Responses

#### Model Not Loaded (503)
```json
{
  "detail": "Model not loaded. Please check server logs."
}
```

#### Template Not Found (404)
```json
{
  "detail": "Template 'chest_xray_standard_001' not found"
}
```

#### Validation Error (422)
```json
{
  "detail": [
    {
      "loc": ["body", "template_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Notes

1. **Model Parameters**: The model uses:
   - `max_new_tokens`: Minimum of `max_length` and 600
   - `temperature`: 0.4
   - `do_sample`: True

2. **Template Selection**: Make sure the `region` matches the folder name in the templates directory, and `template_id` matches the JSON filename (without .json extension).

3. **User Findings**: This is the key input - the radiologist's observations. The model will expand on this in the FINDINGS section.

4. **Normal Reports**: If `user_findings` is "normal", the model will generate a detailed normal report describing normal anatomical structures.

5. **Response Fields**:
   - `medical_content`: Post-processed content extracted from model output
   - `full_report`: Complete formatted report with all metadata
   - `raw_prompt`: Useful for debugging - shows exact prompt sent to model
   - `raw_output`: Useful for debugging - shows raw model output before post-processing

---

## Endpoint 2 Details: Custom Section Generation

### Request Body Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `template_id` | string | Yes | The ID of the template to use (e.g., "chest_xray_standard_001") |
| `template_parameters` | array | Yes | List of section names to generate (e.g., ["FINDINGS", "IMPRESSION"]) |
| `region` | string | Yes | The body region/category (e.g., "chest", "brain", "abdomen", "spine") |
| `user_findings` | string | Yes | Radiologist observations/findings |
| `patient_metadata` | object | No | Patient and DICOM metadata (same structure as Endpoint 1) |

### Available Section Names

You can use any of these section names in `template_parameters`:
- `"FINDINGS"` - Imaging findings section
- `"IMPRESSION"` - Clinical impression/diagnosis
- `"CLINICAL_DETAILS"` or `"CLINICAL DETAILS"` - Clinical context
- `"RECOMMENDATIONS"` - Follow-up recommendations
- `"OBSERVATION"` or `"OBSERVATIONS"` - Maps to FINDINGS section

**Note**: Section names are case-insensitive and will be normalized to uppercase.

### Example Request

```json
{
  "template_id": "chest_xray_standard_001",
  "template_parameters": ["FINDINGS", "IMPRESSION"],
  "region": "chest",
  "user_findings": "Bilateral lower lobe opacities",
  "patient_metadata": {
    "PATIENT_SEX": "M",
    "PATIENT_AGE": "34",
    "BODY_PART": "CHEST",
    "MODALITY": "X-Ray",
    "VIEW_POSITION": "AP"
  }
}
```

### Example: Generate Only Findings

```json
{
  "template_id": "chest_xray_standard_001",
  "template_parameters": ["FINDINGS"],
  "region": "chest",
  "user_findings": "normal"
}
```

### Example: Generate Findings and Impression Only

```json
{
  "template_id": "chest_xray_standard_001",
  "template_parameters": ["FINDINGS", "IMPRESSION"],
  "region": "chest",
  "user_findings": "Bilateral lower lobe opacities"
}
```

### cURL Example

```bash
curl -X POST http://localhost:8004/generate-report-custom \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "chest_xray_standard_001",
    "template_parameters": ["FINDINGS", "IMPRESSION"],
    "region": "chest",
    "user_findings": "Bilateral lower lobe opacities",
    "patient_metadata": {
      "PATIENT_SEX": "M",
      "PATIENT_AGE": "34",
      "BODY_PART": "CHEST",
      "MODALITY": "X-Ray",
      "VIEW_POSITION": "AP"
    }
  }'
```

### Python Example

```python
import requests

url = "http://localhost:8004/generate-report-custom"

payload = {
    "template_id": "chest_xray_standard_001",
    "template_parameters": ["FINDINGS", "IMPRESSION"],
    "region": "chest",
    "user_findings": "Bilateral lower lobe opacities",
    "patient_metadata": {
        "PATIENT_SEX": "M",
        "PATIENT_AGE": "34",
        "BODY_PART": "CHEST",
        "MODALITY": "X-Ray",
        "VIEW_POSITION": "AP"
    }
}

response = requests.post(url, json=payload)
data = response.json()

print("Status:", data["status"])
print("Full Report:")
print(data["full_report"])
```

### Error Responses

#### Invalid Section Names (400)
```json
{
  "detail": "None of the requested sections ['INVALID_SECTION'] were found in the template. Available sections: ['CLINICAL_DETAILS', 'FINDINGS', 'IMPRESSION', 'RECOMMENDATIONS']"
}
```

#### No AI-Generated Sections (400)
```json
{
  "detail": "None of the requested sections are AI-generated. Please select sections that can be generated by the model."
}
```

### Notes

1. **Section Mapping**: The endpoint automatically maps common variations:
   - `"OBSERVATION"` or `"OBSERVATIONS"` → `"FINDINGS"`
   - `"CLINICAL DETAILS"` (with space) → `"CLINICAL_DETAILS"`

2. **Case Insensitive**: Section names are case-insensitive (e.g., `"findings"`, `"Findings"`, `"FINDINGS"` all work).

3. **Template Validation**: Only sections that exist in the template and are marked as `ai_generated: true` will be generated.

4. **Report Format**: The full report will still include all sections from the template format, but only the requested sections will have AI-generated content. Other sections will be empty or show "N/A".

5. **Model Parameters**: The endpoint uses fixed generation parameters:
   - `max_new_tokens`: 600 (fixed, not configurable)
   - `temperature`: 0.4
   - `do_sample`: True

---

## Endpoint 3: Generate Report with OpenAI/GPT Models

### URL
```
POST http://localhost:8004/generate-report-openai
```

This endpoint generates a radiology report using OpenAI's GPT models (GPT-5.1, GPT-5, GPT-4o, etc.). It uses a sophisticated template filling system that removes unrelated conditions and maintains medical consistency.

### Request Headers
```
Content-Type: multipart/form-data
```

### Request Body Parameters (Form Data)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dicom_file` | file | No | DICOM file (.dcm, .dicom) - metadata will be extracted automatically |
| `findings` | string | Yes | Radiologist findings/observations (e.g., "bilateral lower lobe opacities, cardiomegaly") |
| `template` | string | Yes | Template text or JSON string. Can be plain text or JSON with `prompt_template` field |
| `model` | string | No | OpenAI model to use (default: "gpt-5.1"). Options: "gpt-5.1", "gpt-5", "gpt-4o", "gpt-4-turbo", "gpt-4", "o1-preview", "o1-mini" |
| `patient_metadata` | string (JSON) | No | Patient metadata as JSON string (see structure below) |

### Patient Metadata Structure (Optional JSON String)

The `patient_metadata` should be a JSON string containing:

```json
{
  "PATIENT_SEX": "M",
  "PATIENT_AGE": "34",
  "PATIENT_NAME": "John Doe",
  "PATIENT_ID": "P12345",
  "BODY_PART": "CHEST",
  "MODALITY": "X-Ray",
  "VIEW_POSITION": "AP",
  "STUDY_DATE": "2024-11-19",
  "REFERRING_PHYSICIAN": "Dr. Smith"
}
```

### Response Structure

```json
{
  "report": "string - The filled template with findings integrated",
  "openai_model": "string - The model used (e.g., 'gpt-5.1')",
  "status": "string - Status (e.g., 'success')"
}
```

### Available Models

- `gpt-5.1` (default) - Latest GPT model with responses API
- `gpt-5` - GPT-5 model
- `gpt-4o` - GPT-4 Optimized
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-4` - GPT-4
- `o1-preview` - O1 Preview (Reasoning model)
- `o1-mini` - O1 Mini (Reasoning model)

**Note**: If the selected model fails or is not available, the system will automatically fallback to `gpt-4o`.

### Template Format

The template can be provided in two formats:

#### 1. Plain Text Template
```
RADIOLOGY REPORT

CLINICAL DETAILS:
[To be filled]

FINDINGS:
[To be filled]

IMPRESSION:
[To be filled]

RECOMMENDATIONS:
[To be filled]
```

#### 2. JSON Template
```json
{
  "prompt_template": "RADIOLOGY REPORT\n\nCLINICAL DETAILS:\n[To be filled]\n\nFINDINGS:\n[To be filled]\n\nIMPRESSION:\n[To be filled]"
}
```

### Key Features

1. **Template Filling**: Uses a sophisticated prompt system that fills templates with provided findings
2. **Condition Removal**: Automatically removes unrelated conditions from templates that are not in the findings
3. **Medical Consistency**: Ensures no contradictions between sections
4. **Format Preservation**: Maintains exact template structure and formatting
5. **No Hallucination**: Only uses information from provided findings

### Example Requests

#### Example 1: Basic Request (Text Template)
```bash
curl -X POST http://localhost:8004/generate-report-openai \
  -F "findings=bilateral lower lobe opacities, cardiomegaly, no pneumothorax" \
  -F "template=RADIOLOGY REPORT

CLINICAL DETAILS:
[To be filled]

FINDINGS:
[To be filled]

IMPRESSION:
[To be filled]

RECOMMENDATIONS:
[To be filled]" \
  -F "model=gpt-5.1"
```

#### Example 2: Request with DICOM File
```bash
curl -X POST http://localhost:8004/generate-report-openai \
  -F "dicom_file=@/path/to/patient.dcm" \
  -F "findings=bilateral lower lobe opacities, cardiomegaly" \
  -F "template=RADIOLOGY REPORT

FINDINGS:
[To be filled]

IMPRESSION:
[To be filled]" \
  -F "model=gpt-5.1"
```

#### Example 3: Request with Patient Metadata
```bash
curl -X POST http://localhost:8004/generate-report-openai \
  -F "findings=bilateral lower lobe opacities, cardiomegaly" \
  -F "template=RADIOLOGY REPORT

FINDINGS:
[To be filled]

IMPRESSION:
[To be filled]" \
  -F "model=gpt-5.1" \
  -F 'patient_metadata={"PATIENT_SEX":"M","PATIENT_AGE":"34","BODY_PART":"CHEST","MODALITY":"X-Ray"}'
```

### Python Example

```python
import requests

url = "http://localhost:8004/generate-report-openai"

# Prepare form data
form_data = {
    'findings': 'bilateral lower lobe opacities, cardiomegaly, no pneumothorax',
    'template': '''RADIOLOGY REPORT

CLINICAL DETAILS:
[To be filled]

FINDINGS:
[To be filled]

IMPRESSION:
[To be filled]

RECOMMENDATIONS:
[To be filled]''',
    'model': 'gpt-5.1'
}

# Optional: Add DICOM file
files = {}
# files = {'dicom_file': open('patient.dcm', 'rb')}

# Optional: Add patient metadata
# import json
# form_data['patient_metadata'] = json.dumps({
#     'PATIENT_SEX': 'M',
#     'PATIENT_AGE': '34',
#     'BODY_PART': 'CHEST',
#     'MODALITY': 'X-Ray'
# })

response = requests.post(url, data=form_data, files=files)
data = response.json()

print("Status:", data["status"])
print("Model Used:", data["openai_model"])
print("Report:")
print(data["report"])
```

### JavaScript/Node.js Example

```javascript
const FormData = require('form-data');
const axios = require('axios');
const fs = require('fs');

const url = 'http://localhost:8004/generate-report-openai';

const formData = new FormData();
formData.append('findings', 'bilateral lower lobe opacities, cardiomegaly, no pneumothorax');
formData.append('template', `RADIOLOGY REPORT

CLINICAL DETAILS:
[To be filled]

FINDINGS:
[To be filled]

IMPRESSION:
[To be filled]

RECOMMENDATIONS:
[To be filled]`);
formData.append('model', 'gpt-5.1');

// Optional: Add DICOM file
// formData.append('dicom_file', fs.createReadStream('patient.dcm'));

// Optional: Add patient metadata
// formData.append('patient_metadata', JSON.stringify({
//   PATIENT_SEX: 'M',
//   PATIENT_AGE: '34',
//   BODY_PART: 'CHEST',
//   MODALITY: 'X-Ray'
// }));

axios.post(url, formData, {
  headers: formData.getHeaders()
})
  .then(response => {
    console.log('Status:', response.data.status);
    console.log('Model Used:', response.data.openai_model);
    console.log('Report:');
    console.log(response.data.report);
  })
  .catch(error => {
    console.error('Error:', error.response?.data || error.message);
  });
```

### Example Response

```json
{
  "report": "RADIOLOGY REPORT\n\nCLINICAL DETAILS:\nNot specified\n\nFINDINGS:\nBilateral lower lobe opacities are noted. Cardiomegaly is present. No pneumothorax is identified.\n\nIMPRESSION:\nCardiomegaly with bilateral lower lobe opacities. No pneumothorax.\n\nRECOMMENDATIONS:\nNot specified",
  "openai_model": "gpt-5.1",
  "status": "success"
}
```

### Error Responses

#### Missing API Key (500)
```json
{
  "detail": "OPENAI_API_KEY environment variable not set"
}
```

#### Missing Required Fields (422)
```json
{
  "detail": [
    {
      "loc": ["body", "findings"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### Model Error (500)
```json
{
  "detail": "Error generating OpenAI report: [error message]"
}
```

### Notes

1. **Template Filling Prompt**: The system uses a sophisticated prompt file located at `prompts/template_filling_prompt.txt` that ensures:
   - Removal of unrelated conditions from templates
   - Medical consistency across sections
   - Preservation of template structure
   - No hallucination (only uses provided findings)

2. **Model Selection**: The selected model is used directly. Fallback to `gpt-4o` only occurs if the API returns an error related to the model (e.g., "model not found").

3. **DICOM Processing**: If a DICOM file is provided, metadata is automatically extracted and included in the prompt context (but not in the final report output).

4. **Template Format**: The template can be plain text or JSON. If JSON, the system looks for a `prompt_template` field.

5. **Output Format**: The output is the filled template with findings integrated, maintaining the exact structure and formatting of the input template.

6. **API Key**: Requires `OPENAI_API_KEY` environment variable to be set.

