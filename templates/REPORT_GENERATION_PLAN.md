# Radiology Report Generation Flow - Implementation Plan

## Overview
This document outlines the complete flow for generating radiology reports using JSON templates and the MedGemma AI model.

## System Flow

### Step 1: Template Selection & Loading
**Location**: Frontend UI
- User selects anatomical region (chest, brain, abdomen, spine)
- System loads available templates from `templates/{region}/` folder
- User selects a specific template (e.g., "Standard Chest X-Ray Report")
- System reads the JSON template file which contains:
  - Template structure and sections
  - Field definitions
  - Format template with placeholders
  - Prompt template for AI generation

### Step 2: Patient Information Extraction
**Location**: Backend (if DICOM uploaded) or Frontend (manual entry)
- **If DICOM file uploaded**: Extract patient metadata from DICOM tags
  - Patient Name, ID, Age, Sex, DOB
  - Study Date, Modality, Referring Physician
  - Accession Number, etc.
- **If manual entry**: User fills in patient information form
- Map extracted/entered data to template field placeholders

### Step 3: User Input - Radiologist Findings
**Location**: Frontend UI
- Display a text input area labeled "Radiologist Observations" or "Findings"
- User manually enters raw findings from radiologist
  - Can be one word: "normal"
  - Can be short phrase: "bilateral lower lobe opacities"
  - Can be multiple lines: Detailed observations with multiple findings
- This is the KEY INPUT that drives the report generation
- Store as `user_findings` variable

### Step 4: Prompt Construction
**Location**: Backend
- Combine the following into a structured prompt:
  1. **Template Format**: The report structure from JSON (`template_format`)
  2. **User Findings**: Raw observations entered by user
  3. **Patient Info**: Extracted from DICOM or manual entry
  4. **Instructions**: From JSON `prompt_template` field
  5. **Section Instructions**: Specific guidance for each AI-generated section

- Final prompt structure:
  ```
  [System Instructions from prompt_template]
  
  Template Structure:
  {template_format with placeholders}
  
  Radiologist Observations:
  {user_findings}
  
  Patient Information:
  {patient_info formatted}
  
  [Section-specific instructions]
  
  Generate the complete report:
  ```

### Step 5: AI Model Processing
**Location**: Backend (MedGemma Model)
- Send constructed prompt to MedGemma model
- Model processes the prompt and:
  - Analyzes the user findings
  - Understands the template structure
  - Generates content for each AI-generated section:
    - **CLINICAL_DETAILS**: Extracts/summarizes clinical context from findings
    - **FINDINGS**: Expands and structures the raw observations into detailed, organized findings
    - **IMPRESSION**: Creates diagnostic impression based on findings
    - **RECOMMENDATIONS**: Suggests follow-up if needed
- Model fills in all placeholders in the template
- Returns complete formatted report

### Step 6: Report Generation & Formatting
**Location**: Backend
- Model output is the complete report with all sections filled
- Replace any remaining placeholders (like REPORT_DATE) with actual values
- Clean up formatting:
  - Remove duplicate sections if model repeats
  - Ensure proper spacing and line breaks
  - Validate all required sections are present
- Apply final formatting according to template structure

### Step 7: Display & Output
**Location**: Frontend UI
- Display the generated report in a formatted view
- Show sections clearly separated
- Allow user to:
  - Review the report
  - Edit if needed (manual corrections)
  - Export as PDF/Text
  - Save to database/file system

## Key Components Needed

### Backend Endpoints:
1. **GET /templates/{region}** - List available templates for a region
2. **GET /templates/{region}/{template_id}** - Get specific template JSON
3. **POST /generate-report** - Main endpoint for report generation
   - Input: template_id, user_findings, patient_info (optional), dicom_metadata (optional)
   - Output: Generated formatted report

### Frontend Components:
1. **Template Selector** - Dropdown/List to choose template
2. **Patient Info Form** - If not from DICOM
3. **Findings Input** - Large textarea for radiologist observations
4. **Report Viewer** - Display generated report with formatting
5. **Export Options** - PDF, Text, Print

### Data Flow:
```
JSON Template → Template Structure
     +
User Findings (Manual Input)
     +
Patient Info (DICOM or Manual)
     +
Prompt Template
     ↓
Constructed Prompt
     ↓
MedGemma Model
     ↓
Generated Report
     ↓
Formatted Output
```

## Example Flow Walkthrough

1. **User uploads DICOM file** → System extracts metadata
2. **User selects "Chest" region** → System shows available chest templates
3. **User selects "Standard Chest X-Ray Report"** → Template loaded
4. **System auto-fills patient info** from DICOM metadata
5. **User enters findings**: "Bilateral lower lobe opacities, cardiomegaly, no pneumothorax"
6. **User clicks "Generate Report"**
7. **Backend constructs prompt**:
   - Template format with placeholders
   - User findings: "Bilateral lower lobe opacities..."
   - Patient info: "Name: John Doe, Age: 45..."
   - Instructions for AI
8. **MedGemma processes** and generates:
   - Clinical Details: "45-year-old male with chest pain..."
   - Findings: "● Lungs: Bilateral lower lobe opacities... ● Heart: Cardiomegaly... ● Pleura: No pneumothorax..."
   - Impression: "Bilateral lower lobe pneumonia, cardiomegaly"
   - Recommendations: "Follow-up chest X-ray in 2 weeks..."
9. **Report displayed** in formatted view
10. **User can export** or make edits

## Benefits of This Approach

1. **Flexible Templates**: Easy to add new templates by creating JSON files
2. **Structured Output**: Consistent report format across all reports
3. **AI Enhancement**: Model expands brief observations into detailed reports
4. **DICOM Integration**: Automatic patient info extraction
5. **User Control**: Radiologist provides key findings, AI handles formatting
6. **Scalable**: Can support multiple anatomical regions and template types

## Next Steps for Implementation

1. Create backend endpoint to load templates
2. Create backend endpoint for report generation
3. Build frontend template selector component
4. Build findings input component
5. Build report viewer component
6. Integrate with existing DICOM upload functionality
7. Add export functionality (PDF generation)

