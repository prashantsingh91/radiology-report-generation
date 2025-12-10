# OpenAI/GPT API Endpoint - Request/Response Structure

## Endpoint
```
POST http://localhost:8004/generate-report-openai
Content-Type: multipart/form-data
```

---

## Request Structure

### Form Data Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `dicom_file` | File | No | DICOM file (.dcm, .dicom) | `patient.dcm` |
| `findings` | string | **Yes** | Radiologist findings/observations | `"bilateral lower lobe opacities, cardiomegaly"` |
| `template` | string | **Yes** | Template text or JSON string | See examples below |
| `model` | string | No | OpenAI model (default: `"gpt-5.1"`) | `"gpt-5.1"`, `"gpt-4o"`, etc. |
| `patient_metadata` | string (JSON) | No | Patient metadata as JSON string | See structure below |

### Available Models
- `gpt-5.1` (default) - Latest GPT model
- `gpt-5` - GPT-5 model
- `gpt-4o` - GPT-4 Optimized
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-4` - GPT-4
- `o1-preview` - O1 Preview (Reasoning)
- `o1-mini` - O1 Mini (Reasoning)

### Patient Metadata Structure (JSON String)

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

### Template Format

#### Option 1: Plain Text Template
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

#### Option 2: JSON Template
```json
{
  "prompt_template": "RADIOLOGY REPORT\n\nCLINICAL DETAILS:\n[To be filled]\n\nFINDINGS:\n[To be filled]\n\nIMPRESSION:\n[To be filled]"
}
```

---

## Response Structure

### Success Response (200 OK)

```json
{
  "report": "RADIOLOGY REPORT\n\nCLINICAL DETAILS:\nNot specified\n\nFINDINGS:\nBilateral lower lobe opacities are noted. Cardiomegaly is present. No pneumothorax is identified.\n\nIMPRESSION:\nCardiomegaly with bilateral lower lobe opacities. No pneumothorax.\n\nRECOMMENDATIONS:\nNot specified",
  "openai_model": "gpt-5.1",
  "status": "success"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `report` | string | The filled template with findings integrated |
| `openai_model` | string | The model actually used (may differ from requested if fallback occurred) |
| `status` | string | Status message (typically `"success"`) |

---

## Error Responses

### Missing API Key (500 Internal Server Error)

```json
{
  "detail": "OPENAI_API_KEY environment variable not set"
}
```

### Missing Required Field (422 Unprocessable Entity)

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

### Model Error (500 Internal Server Error)

```json
{
  "detail": "Error generating OpenAI report: [error message]"
}
```

### Invalid Template Format (500 Internal Server Error)

```json
{
  "detail": "Error processing template: [error message]"
}
```

---

## Example Requests

### Example 1: Basic Request (cURL)

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

### Example 2: With DICOM File (cURL)

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

### Example 3: With Patient Metadata (cURL)

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

### Example 4: Python Request

```python
import requests

url = "http://localhost:8004/generate-report-openai"

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
import json
form_data['patient_metadata'] = json.dumps({
    'PATIENT_SEX': 'M',
    'PATIENT_AGE': '34',
    'BODY_PART': 'CHEST',
    'MODALITY': 'X-Ray'
})

response = requests.post(url, data=form_data, files=files)
data = response.json()

print("Status:", data["status"])
print("Model Used:", data["openai_model"])
print("Report:")
print(data["report"])
```

### Example 5: JavaScript/Node.js Request

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
formData.append('patient_metadata', JSON.stringify({
  PATIENT_SEX: 'M',
  PATIENT_AGE: '34',
  BODY_PART: 'CHEST',
  MODALITY: 'X-Ray'
}));

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

---

## Response Examples

### Success Response Example

```json
{
  "report": "RADIOLOGY REPORT\n\nCLINICAL DETAILS:\nNot specified\n\nFINDINGS:\nBilateral lower lobe opacities are noted. Cardiomegaly is present. No pneumothorax is identified.\n\nIMPRESSION:\nCardiomegaly with bilateral lower lobe opacities. No pneumothorax.\n\nRECOMMENDATIONS:\nNot specified",
  "openai_model": "gpt-5.1",
  "status": "success"
}
```

### Error Response Example (Missing Field)

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

### Error Response Example (API Key Missing)

```json
{
  "detail": "OPENAI_API_KEY environment variable not set"
}
```

---

## Notes

1. **Template Filling**: The system uses a sophisticated prompt system that:
   - Fills templates with provided findings
   - Removes unrelated conditions from templates
   - Maintains medical consistency
   - Preserves template structure and formatting

2. **Model Fallback**: If the selected model fails or is not available, the system automatically falls back to `gpt-4o`.

3. **DICOM Processing**: If a DICOM file is provided, metadata is automatically extracted and included in the prompt context (but not in the final report output).

4. **Template Format**: The template can be plain text or JSON. If JSON, the system looks for a `prompt_template` field.

5. **Timeout**: Requests may take 30-120 seconds depending on the model and complexity.

6. **API Key**: Requires `OPENAI_API_KEY` environment variable to be set on the server.

---

## Testing

Use the provided test script to test the endpoint:

```bash
python3 test_openai_api.py
```

The test script includes:
- Basic request test
- Request with patient metadata
- Different model testing
- JSON template format
- Template cleanup (removing unrelated conditions)
- Error handling
- Complex template testing

