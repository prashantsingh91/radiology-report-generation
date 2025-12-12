from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import torch
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
import logging
import pydicom
import numpy as np
from PIL import Image
import io
import base64
import json
import re
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import subprocess
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedGemma API", version="1.0.0")

# CORS middleware to allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for external access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None

def _serialize_pydicom_value(value: Any) -> Any:
    """Recursively serialize pydicom values (Sequence/Dataset) with no truncation.
    Returns Python primitives/lists/dicts suitable for JSON dumping."""
    try:
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence
        import pydicom
    except Exception:
        # Fallback plain stringify
        return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return str(value)
    if isinstance(value, Sequence):
        return [_serialize_pydicom_value(item) for item in value]
    if isinstance(value, Dataset):
        result: Dict[str, Any] = {}
        for elem in value:
            try:
                # Skip raw PixelData in nested datasets if present
                if elem.tag == pydicom.tag.Tag('PixelData'):
                    continue
                key = elem.name
                result[key] = _serialize_pydicom_value(elem.value)
            except Exception:
                continue
        return result
    if isinstance(value, (list, tuple)):
        return [ _serialize_pydicom_value(v) for v in value ]
    # Numeric or other simple types
    try:
        import pydicom
        if isinstance(value, pydicom.valuerep.DSfloat):
            return float(value)
        if isinstance(value, pydicom.valuerep.IS):
            return str(value)
    except Exception:
        pass
    return value if isinstance(value, (int, float, bool)) else str(value)

# Request/Response models
class PromptRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 512

class PromptResponse(BaseModel):
    output: str
    status: str

class DICOMResponse(BaseModel):
    image_base64: str
    metadata: Dict[str, Any]
    status: str

class ReportGenerationRequest(BaseModel):
    # template_id removed; endpoint now uses the generalized template by default
    region: str
    user_findings: str
    patient_metadata: Optional[Dict[str, Any]] = None
    max_length: Optional[int] = 1024
    
    # Allow clients that still send template_id to pass without error
    model_config = {"extra": "ignore"}

class ReportGenerationResponse(BaseModel):
    medical_content: str
    full_report: str
    status: str
    raw_prompt: Optional[str] = None
    raw_output: Optional[str] = None

class CustomReportGenerationRequest(BaseModel):
    # template_id removed; endpoint now uses the generalized template by default
    template_parameters: List[str]  # List of section names like ["FINDINGS", "IMPRESSION"]
    region: str
    user_findings: str
    patient_metadata: Optional[Dict[str, Any]] = None
    
    # Allow extra fields from older clients
    model_config = {"extra": "ignore"}

class PreviewRequest(BaseModel):
    region: str
    # template_id removed; endpoint now previews the generalized template by default
    patient_metadata: Optional[Dict[str, Any]] = None
    
    # Allow extra fields from older clients
    model_config = {"extra": "ignore"}

class PreviewResponse(BaseModel):
    preview_html: str
    status: str

class OpenAIReportRequest(BaseModel):
    dicom_file: Optional[str] = None  # Base64 encoded DICOM file (optional)
    findings: str
    # template removed - endpoint now uses generalized template and prompt from prompts/template_filling_prompt.txt
    patient_metadata: Optional[Dict[str, Any]] = None
    model: Optional[str] = "gpt-5.1"  # Default to gpt-5.1, can use gpt-4o, o1-preview or o1-mini

class OpenAIReportResponse(BaseModel):
    report: str
    status: str
    openai_model: str  # Model name used for generation
    
    model_config = {"protected_namespaces": ()}

class SrInterpretRequest(BaseModel):
    sr_json: Dict[str, Any]
    model: Optional[str] = "gpt-5.1"

class SrInterpretResponse(BaseModel):
    findings: str
    status: str
    openai_model: str

def load_model():
    """Load MedGemma model and processor from cache"""
    global model, processor
    
    if model is not None and processor is not None:
        logger.info("Model already loaded")
        return
    
    try:
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("⚠️  GPU not available, will use CPU (slower)")
        
        model_id = "google/medgemma-4b-it"
        
        # Check default Hugging Face cache location first
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        hf_model_cache = os.path.join(hf_cache_dir, "models--google--medgemma-4b-it")
        
        # Also check local project cache
        project_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
        project_model_cache = os.path.join(project_cache_dir, "models--google--medgemma-4b-it")
        
        # Priority: 1. HF default cache, 2. Project cache, 3. Download
        model_cache_path = None
        if os.path.exists(hf_model_cache):
            model_cache_path = hf_model_cache
            logger.info(f"Found model in Hugging Face cache: {hf_model_cache}")
        elif os.path.exists(project_model_cache):
            model_cache_path = project_model_cache
            logger.info(f"Found model in project cache: {project_model_cache}")
        
        if model_cache_path:
            # Find snapshot directory
            snapshots_path = os.path.join(model_cache_path, "snapshots")
            if os.path.exists(snapshots_path):
                snapshot_dirs = [d for d in os.listdir(snapshots_path) 
                               if os.path.isdir(os.path.join(snapshots_path, d))]
                if snapshot_dirs:
                    snapshot_dir = os.path.join(snapshots_path, snapshot_dirs[0])
                    logger.info(f"Loading from snapshot: {snapshot_dir}")
                    model = AutoModelForImageTextToText.from_pretrained(
                        snapshot_dir,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    processor = AutoProcessor.from_pretrained(snapshot_dir, trust_remote_code=True)
                    
                    # Verify model device
                    if hasattr(model, 'device'):
                        logger.info(f"Model device: {model.device}")
                    elif hasattr(model, 'hf_device_map'):
                        logger.info(f"Model device map: {model.hf_device_map}")
                    else:
                        # Check first parameter device
                        first_param = next(model.parameters())
                        logger.info(f"Model on device: {first_param.device}")
                    
                    logger.info("Model loaded successfully from cache")
                    return
                else:
                    logger.warning("No snapshot directory found in cache")
            else:
                logger.warning("No snapshots directory found in cache")
        
        # If not in cache, try to load from Hugging Face (will use default cache)
        logger.info("Model not found in local cache, loading from Hugging Face...")
        logger.info("This will use the default Hugging Face cache if available")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=False
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # Verify model device
        if hasattr(model, 'device'):
            logger.info(f"Model device: {model.device}")
        elif hasattr(model, 'hf_device_map'):
            logger.info(f"Model device map: {model.hf_device_map}")
        else:
            # Check first parameter device
            first_param = next(model.parameters())
            logger.info(f"Model on device: {first_param.device}")
        
        logger.info("Model loaded successfully from Hugging Face")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up... Loading model...")
    try:
        load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        logger.error("Please ensure:")
        logger.error("1. You have internet connection")
        logger.error("2. You are authenticated with Hugging Face (run: huggingface-cli login)")
        logger.error("3. You have access to the gated repository: google/medgemma-4b-it")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "MedGemma API is running",
        "model_loaded": model is not None and processor is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    device_info = "unknown"
    gpu_available = torch.cuda.is_available()
    
    if model is not None:
        try:
            if hasattr(model, 'device'):
                device_info = str(model.device)
            elif hasattr(model, 'hf_device_map'):
                device_info = str(model.hf_device_map)
            else:
                first_param = next(model.parameters())
                device_info = str(first_param.device)
        except:
            device_info = "unknown"
    
    return {
        "status": "healthy",
        "model_loaded": model is not None and processor is not None,
        "gpu_available": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "model_device": device_info
    }

@app.post("/generate", response_model=PromptResponse)
async def generate_text(request: PromptRequest):
    """Generate text based on prompt using MedGemma model"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Format prompt as a chat message for MedGemma
        messages = [
            {
                "role": "user",
                # Use structured content format expected by MedGemma tokenizer
                "content": [
                    {"type": "text", "text": request.prompt}
                ]
            }
        ]
        
        # Get model device (handle different model types)
        if hasattr(model, 'device'):
            model_device = model.device
        else:
            # Get device from first parameter
            model_device = next(model.parameters()).device
        
        # Apply chat template and tokenize
        # Following the exact pattern from working medgemma_demo.py
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Check if inputs has .to() method (BatchEncoding) or is a dict
        if hasattr(inputs, 'to'):
            # BatchEncoding object - use .to() method directly
            try:
                inputs = inputs.to(model_device, dtype=torch.bfloat16)
            except Exception as e:
                logger.error(f"Error calling .to() on BatchEncoding: {e}")
                # Fallback: convert to dict and move tensors manually
                inputs = {k: v.to(model_device, dtype=torch.bfloat16) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
        elif isinstance(inputs, dict):
            # Plain dictionary - move each tensor to device
            inputs = {k: v.to(model_device, dtype=torch.bfloat16) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
        else:
            logger.error(f"Unexpected input type: {type(inputs)}")
            raise ValueError(f"apply_chat_template returned unexpected type: {type(inputs)}")
        
        # Get input_ids - handle both dict and BatchEncoding access patterns
        try:
            if isinstance(inputs, dict):
                input_ids = inputs["input_ids"]
            else:
                # BatchEncoding - access as attribute
                input_ids = inputs.input_ids
        except (KeyError, AttributeError) as e:
            logger.error(f"Could not access input_ids: {e}")
            logger.error(f"Inputs type: {type(inputs)}, keys/attrs: {dir(inputs) if not isinstance(inputs, dict) else list(inputs.keys())}")
            raise ValueError("Could not access input_ids from processor output")
        
        input_len = input_ids.shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=min(request.max_length, 600),
                do_sample=False,
                temperature=0.3
            )
            # Extract only the generated tokens (exclude input)
            generation = generation[0][input_len:]
        
        # Decode the output
        generated_text = processor.decode(generation, skip_special_tokens=True)
        
        return PromptResponse(
            output=generated_text.strip(),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

def process_dicom_file(file_content: bytes) -> tuple[Image.Image, Dict[str, Any]]:
    """Process DICOM file and extract image (if present) and metadata.
    Handles non-image DICOMs (e.g., SR) by returning a placeholder image with full metadata."""
    try:
        # Read DICOM file from bytes
        dcm = pydicom.dcmread(io.BytesIO(file_content), force=True)
        
        # Extract metadata first
        metadata: Dict[str, Any] = {}
        try:
            for elem in dcm:
                try:
                    # Skip pixel data and other large elements
                    if elem.tag == pydicom.tag.Tag('PixelData'):
                        continue
                    # Get element name and fully serialized value (handles SR Content Sequence)
                    name = elem.name
                    serialized_value = _serialize_pydicom_value(elem.value)
                    # For complex (dict/list) values, dump to pretty JSON string
                    if isinstance(serialized_value, (dict, list)):
                        metadata[name] = json.dumps(serialized_value, ensure_ascii=False, indent=2)
                    else:
                        metadata[name] = serialized_value
                except Exception:
                    # Skip problematic elements
                    continue
        except Exception as e:
            metadata['Error'] = f'Metadata extraction failed: {str(e)}'
        
        # If no pixel data (e.g., Structured Report), return placeholder image with metadata
        if 'PixelData' not in dcm:
            # Create a neutral placeholder image
            placeholder = Image.new('L', (512, 512), color=200)
            # Annotate minimal info in metadata to clarify
            try:
                sop_class = getattr(dcm.file_meta, 'MediaStorageSOPClassUID', None) or getattr(dcm, 'SOPClassUID', None)
                metadata['Note'] = "No PixelData found; non-image DICOM (e.g., Structured Report)."
                if sop_class:
                    metadata['SOPClassUID'] = str(sop_class)
            except Exception:
                pass
            return placeholder, metadata

        # Otherwise, extract pixel data (with decompression attempts)
        try:
            pixel_array = dcm.pixel_array
        except Exception as compression_error:
            logger.warning(f"PixelData decode failed: {compression_error}")
            # Try to enable/load common handlers and decompress, then retry
            try:
                # Attempt to import handlers so pydicom registers them
                try:
                    import pylibjpeg  # noqa: F401
                    import pylibjpeg_libjpeg  # noqa: F401
                    import pylibjpeg_openjpeg  # noqa: F401
                    import pylibjpeg_rle  # noqa: F401
                except Exception:
                    pass
                try:
                    import imagecodecs  # noqa: F401
                except Exception:
                    pass
                # If compressed, try in-place decompression using available handlers
                try:
                    dcm.decompress()
                except Exception:
                    # Not fatal—some handlers decompress on first pixel_array access
                    pass
                # Retry pixel array
                pixel_array = dcm.pixel_array
            except Exception as retry_error:
                logger.error(f"DICOM decompression failed: {retry_error}")
                # Fallback: try external gdcmconv if available to convert to uncompressed
                try:
                    gdcmconv_path = shutil.which("gdcmconv")
                    if not gdcmconv_path:
                        raise RuntimeError("gdcmconv not found")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as in_f:
                        in_path = in_f.name
                        in_f.write(file_content)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as out_f:
                        out_path = out_f.name
                    try:
                        # --raw writes uncompressed
                        subprocess.run(
                            [gdcmconv_path, "--raw", in_path, out_path],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                        # Read the uncompressed file and retry
                        dcm_uncompressed = pydicom.dcmread(out_path, force=True)
                        if 'PixelData' not in dcm_uncompressed:
                            raise RuntimeError("No PixelData in converted DICOM")
                        pixel_array = dcm_uncompressed.pixel_array
                        dcm = dcm_uncompressed  # use the converted dataset for metadata extraction
                    finally:
                        try:
                            os.remove(in_path)
                        except Exception:
                            pass
                        try:
                            os.remove(out_path)
                        except Exception:
                            pass
                except Exception as fallback_err:
                    raise ValueError(
                        "Unable to decode PixelData. If this is a compressed image DICOM, ensure decoders "
                        "(pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg, pylibjpeg-rle, imagecodecs) are installed; "
                        "optionally install GDCM (sudo apt-get install -y gdcm) for auto-conversion. "
                        f"Details: {fallback_err}"
                    )

        # Normalize pixel values for display
        if pixel_array.dtype != np.uint8:
            pixel_array = ((pixel_array - pixel_array.min()) /
                          (pixel_array.max() - pixel_array.min() + 1e-8) * 255).astype(np.uint8)

        # Ensure array shape is suitable for PIL
        arr = pixel_array
        try:
            if arr.ndim == 3:
                # If channels-last 3 or 4, leave as-is; otherwise assume multi-frame and pick middle frame
                if arr.shape[-1] not in (3, 4):
                    arr = arr[arr.shape[0] // 2]  # pick middle frame
            elif arr.ndim > 3:
                # Reduce higher dimensions by squeezing and averaging until 2D
                arr = np.squeeze(arr)
                while arr.ndim > 2:
                    arr = arr.mean(axis=0)
            # Final squeeze to remove singleton dims like (1, N) or (N, 1, 1)
            arr = np.squeeze(arr)
            # If still 3D with unexpected channels, collapse by mean across first axis
            if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
                arr = arr.mean(axis=0)
            # Ensure dtype uint8
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        except Exception as shape_err:
            logger.warning(f"Failed to reshape pixel data for display: {shape_err}")
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)

        # Convert to PIL Image
        image = Image.fromarray(arr)
        return image, metadata
        
    except Exception as e:
        logger.error(f"Error processing DICOM file: {str(e)}")
        raise

@app.post("/upload-dicom", response_model=DICOMResponse)
async def upload_dicom(file: UploadFile = File(...)):
    """Upload and process DICOM file"""
    try:
        # Check file extension
        if not file.filename.lower().endswith(('.dcm', '.dicom')):
            raise HTTPException(status_code=400, detail="File must be a DICOM file (.dcm or .dicom)")
        
        # Read file content
        file_content = await file.read()
        
        # Process DICOM file
        image, metadata = process_dicom_file(file_content)
        
        # Convert image to base64 for frontend
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        return DICOMResponse(
            image_base64=img_data_url,
            metadata=metadata,
            status="success"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing DICOM file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing DICOM file: {str(e)}")

def extract_sections_from_content(content: str, medical_sections: list) -> dict:
    """Extract section content from the generated report (robust to case/colon variations)"""
    sections: Dict[str, str] = {}
    # Build ordered list of canonical section names and map to field_ids
    section_names: list[str] = []
    section_map: Dict[str, str] = {}
    for section in medical_sections:
        if section.get("ai_generated", False):
            field_id = section.get("field_id")
            section_name = section.get("section_name", "").strip()
            if not section_name:
                continue
            canonical = section_name.upper()
            section_names.append(canonical)
            section_map[canonical] = field_id
            sections[field_id] = ""  # init
    if not section_names:
        return sections
    # Build regex to find headers with optional colon, case-insensitive, start of line
    # Pattern example: ^\s*(CLINICAL DETAILS)\s*:?\s*$
    header_patterns = {
        name: re.compile(rf"^\s*{re.escape(name)}\s*:?\s*$", flags=re.IGNORECASE | re.MULTILINE)
        for name in section_names
    }
    # Find all header positions
    header_spans: list[tuple[int, int, str]] = []
    for name, pattern in header_patterns.items():
        for m in pattern.finditer(content):
            header_spans.append((m.start(), m.end(), name))
    # If no headers matched, try lenient inline header (word followed by colon)
    if not header_spans:
        for name in section_names:
            idx = content.upper().find(f"{name}:")
            if idx != -1:
                header_spans.append((idx, idx + len(name) + 1, name))
    # Nothing found, return empty sections
    if not header_spans:
        return sections
    # Sort by position in text
    header_spans.sort(key=lambda x: x[0])
    # Walk headers and slice content until next header
    for i, (_, end_pos, name) in enumerate(header_spans):
        start_content = end_pos
        if i + 1 < len(header_spans):
            next_start, _, _ = header_spans[i + 1]
            section_text = content[start_content:next_start].strip()
        else:
            section_text = content[start_content:].strip()
        # Assign to corresponding field_id
        field_id = section_map.get(name)
        if field_id:
            sections[field_id] = section_text
    return sections

def clean_report_output(content: str, section_headers: list) -> str:
    """Clean the generated report output by removing duplicates, numbering, etc."""
    if not content:
        return content
    
    # Remove any text before the first section header
    first_section = section_headers[0] if section_headers else "CLINICAL DETAILS"
    first_header = f"{first_section}:"
    
    if first_header in content:
        # Find the first occurrence of the first section
        idx = content.find(first_header)
        if idx > 0:
            content = content[idx:]
    
    # Remove numbering patterns (1., 2., 3., etc.) at start of lines
    # Remove numbering like "1. ", "2. ", "3. " at the start of lines
    content = re.sub(r'^\d+\.\s+', '', content, flags=re.MULTILINE)
    
    # Remove duplicate section headers (keep only first occurrence of each)
    seen_headers = set()
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_upper = line.upper().strip()
        is_header = False
        
        # Check if this line is a section header
        for header in section_headers:
            header_with_colon = f"{header}:"
            if line_upper.startswith(header_with_colon) or line_upper == header:
                if header in seen_headers:
                    # Skip duplicate header
                    is_header = True
                    break
                else:
                    # First occurrence, keep it
                    seen_headers.add(header)
                    cleaned_lines.append(f"{header}:")
                    is_header = True
                    break
        
        if not is_header:
            cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    # Remove any text after the last section (if there's trailing content)
    last_section = section_headers[-1] if section_headers else "RECOMMENDATIONS"
    last_header = f"{last_section}:"
    
    if last_header in content:
        # Find all occurrences
        parts = content.split(last_header)
        if len(parts) > 1:
            # Keep everything up to and including the last section
            # But remove any trailing content that looks like it's after the report
            last_section_content = parts[-1].strip()
            # If there's content after recommendations that looks like footer/extra, remove it
            if last_section == "RECOMMENDATIONS" and len(last_section_content) > 200:
                # Might have extra content, try to find where it should end
                lines_after = last_section_content.split('\n')
                # Keep reasonable content, but stop if we see patterns like "---" or "Report generated"
                cleaned_after = []
                for line in lines_after:
                    if line.strip().startswith('---') or 'Report generated' in line:
                        break
                    cleaned_after.append(line)
                last_section_content = '\n'.join(cleaned_after).strip()
            
            content = last_header.join(parts[:-1]) + last_header + "\n" + last_section_content
    
    # Clean up extra whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 consecutive newlines
    content = content.strip()
    
    return content

def validate_and_expand_impression(impression_text: str, findings_text: str = "") -> str:
    """Validate IMPRESSION section and expand if it's too short (single word/phrase)"""
    if not impression_text:
        return impression_text
    
    # Remove trailing period and whitespace for checking
    cleaned = impression_text.strip().rstrip('.')
    
    # Check if it's just a single word or very short phrase (less than 20 chars or no sentence structure)
    # Count sentences by looking for sentence-ending punctuation
    sentences = re.split(r'[.!?]+', cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If it's a single word/short phrase (no proper sentence structure)
    if len(cleaned) < 30 or len(sentences) == 0 or (len(sentences) == 1 and len(cleaned.split()) <= 3):
        logger.warning(f"IMPRESSION is too short: '{impression_text}'. Attempting to expand.")
        
        # Try to create a proper sentence from the short impression
        # Remove any trailing punctuation
        base_term = cleaned.strip().rstrip('.,!?')
        
        # If it's just a medical term, expand it into a concise sentence
        if len(base_term.split()) <= 2:
            # Create a concise diagnostic statement (1 sentence only)
            expanded = f"{base_term} is present."
            
            logger.info(f"Expanded IMPRESSION from '{impression_text}' to '{expanded}'")
            return expanded
    
    return impression_text

def get_templates_dir():
    """Get the templates directory path"""
    # Get the directory where this script is located
    backend_dir = Path(__file__).resolve().parent
    # Go up one level to project root, then into templates
    return backend_dir.parent / "templates"

@app.get("/templates/{region}")
async def list_templates(region: str):
    """List all available templates for a region"""
    try:
        templates_dir = get_templates_dir() / region
        if not templates_dir.exists():
            raise HTTPException(status_code=404, detail=f"Region '{region}' not found")
        
        templates = []
        for json_file in templates_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    template_data = json.load(f)
                    templates.append({
                        "template_id": template_data.get("template_id", json_file.stem),
                        "name": template_data.get("name", json_file.stem),
                        "category": template_data.get("category", region),
                        "modality": template_data.get("modality", ""),
                        "description": template_data.get("description", "")
                    })
            except Exception as e:
                logger.warning(f"Error reading template {json_file}: {e}")
                continue
        
        return {"region": region, "templates": templates}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing templates: {str(e)}")

@app.get("/templates/{region}/{template_id}")
async def get_template(region: str, template_id: str):
    """Get a specific template by ID"""
    try:
        templates_dir = get_templates_dir() / region
        if not templates_dir.exists():
            raise HTTPException(status_code=404, detail=f"Region '{region}' not found")
        
        # Try to find template file
        template_file = templates_dir / f"{template_id}.json"
        if not template_file.exists():
            # Try without extension
            template_file = templates_dir / template_id
            if not template_file.exists():
                raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
        
        with open(template_file, 'r') as f:
            template_data = json.load(f)
        
        return template_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading template: {str(e)}")

@app.post("/generate-report", response_model=ReportGenerationResponse)
async def generate_report(request: ReportGenerationRequest):
    """Generate radiology report from template and user findings"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Load generalized template by default
        templates_dir = get_templates_dir() / "general"
        template_file = templates_dir / "general_medical_analysis_001.json"
        if not template_file.exists():
            raise HTTPException(status_code=404, detail="Default template 'general_medical_analysis_001' not found")
        
        with open(template_file, 'r') as f:
            template = json.load(f)
        
        # Build DICOM metadata example (for reference only)
        patient_metadata = request.patient_metadata or {}
        dicom_metadata_example = ""
        if patient_metadata:
            dicom_metadata_example = f"Patient's Sex: {patient_metadata.get('PATIENT_SEX', 'N/A')}\n"
            dicom_metadata_example += f"Patient's Age: {patient_metadata.get('PATIENT_AGE', 'N/A')}\n"
            dicom_metadata_example += f"Body Part Examined: {patient_metadata.get('BODY_PART', 'CHEST')}\n"
            dicom_metadata_example += f"Modality: {patient_metadata.get('MODALITY', template.get('modality', 'N/A'))}\n"
            if patient_metadata.get('VIEW_POSITION'):
                dicom_metadata_example += f"View Position: {patient_metadata.get('VIEW_POSITION')}"
        else:
            dicom_metadata_example = "Patient's Sex: N/A\nPatient's Age: N/A\nBody Part Examined: CHEST\nModality: X-Ray"
        
        # Build section list and headers
        medical_sections = template.get("medical_sections", [])
        section_list = ""
        section_headers = []
        for section in medical_sections:
            if section.get("ai_generated", False):
                section_name = section.get("section_name", "").upper()
                section_headers.append(section_name)
                instructions = section.get("instructions", "")
                section_list += f"- {section_name}: {instructions}\n"
        
        section_headers_str = ", ".join([f"{h}:" for h in section_headers])
        
        # Construct prompt with all placeholders
        prompt_template = template.get("prompt_template", "")
        
        # Get modality type and study context from template or patient metadata
        modality_type = "medical imaging"  # default
        study_context = "This is a general medical imaging study. Provide a comprehensive analysis based on the clinical observations."
        
        # Check if template has modality_specific_prompts
        modality_specific = template.get("modality_specific_prompts", {})
        patient_metadata_for_prompt = request.patient_metadata or {}
        
        # Determine modality from patient metadata or use default
        modality_from_metadata = patient_metadata_for_prompt.get("MODALITY", "").lower()
        if modality_from_metadata:
            # Map common modality names to keys
            modality_key = "general"
            if "x-ray" in modality_from_metadata or "xray" in modality_from_metadata or "radiology" in modality_from_metadata:
                modality_key = "radiology"
            elif "pathology" in modality_from_metadata or "histo" in modality_from_metadata:
                modality_key = "pathology"
            elif "dermatology" in modality_from_metadata or "skin" in modality_from_metadata:
                modality_key = "dermatology"
            elif "ophthalmology" in modality_from_metadata or "eye" in modality_from_metadata or "retina" in modality_from_metadata:
                modality_key = "ophthalmology"
            
            if modality_key in modality_specific:
                modality_type = modality_specific[modality_key].get("modality_type", modality_type)
                study_context = modality_specific[modality_key].get("study_context", study_context)
        
        # Replace all placeholders
        prompt = prompt_template.replace("{dicom_metadata_example}", dicom_metadata_example)
        prompt = prompt.replace("{section_list}", section_list.strip())
        prompt = prompt.replace("{section_headers}", section_headers_str)
        prompt = prompt.replace("{user_findings}", request.user_findings)
        prompt = prompt.replace("{modality_type}", modality_type)
        prompt = prompt.replace("{study_context}", study_context)
        
        logger.info("Generating report with template: general_medical_analysis_001")
        logger.info(f"User findings length: {len(request.user_findings)} characters")
        logger.info(f"Sections to generate: {', '.join(section_headers)}")
        
        # Log the full prompt
        logger.info("=" * 80)
        logger.info("PROMPT SENT TO MODEL:")
        logger.info("=" * 80)
        logger.info(prompt)
        logger.info("=" * 80)
        
        # Format prompt as chat message for MedGemma
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Get model device
        if hasattr(model, 'device'):
            model_device = model.device
        else:
            model_device = next(model.parameters()).device
        
        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        if hasattr(inputs, 'to'):
            try:
                inputs = inputs.to(model_device, dtype=torch.bfloat16)
            except Exception as e:
                logger.error(f"Error calling .to() on BatchEncoding: {e}")
                inputs = {k: v.to(model_device, dtype=torch.bfloat16) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
        elif isinstance(inputs, dict):
            inputs = {k: v.to(model_device, dtype=torch.bfloat16) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
        else:
            raise ValueError(f"apply_chat_template returned unexpected type: {type(inputs)}")
        
        # Get input_ids
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs.input_ids
        
        input_len = input_ids.shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=min(request.max_length, 600),
                do_sample=True,
                temperature=0.4
            )
            generation = generation[0][input_len:]
        
        # Decode the output
        raw_output = processor.decode(generation, skip_special_tokens=True).strip()
        
        # Log the raw output
        logger.info("=" * 80)
        logger.info("RAW MODEL OUTPUT (BEFORE POST-PROCESSING):")
        logger.info("=" * 80)
        logger.info(raw_output)
        logger.info("=" * 80)
        
        # Post-process: Clean the output
        medical_content = clean_report_output(raw_output, section_headers)
        
        # Extract sections from medical content (improved extraction)
        sections = extract_sections_from_content(medical_content, medical_sections)
        
        # Validate and expand IMPRESSION if it's too short
        if "IMPRESSION" in sections and sections["IMPRESSION"]:
            findings_text = sections.get("FINDINGS", "")
            sections["IMPRESSION"] = validate_and_expand_impression(sections["IMPRESSION"], findings_text)
        
        # If sections not extracted, provide defaults
        if not any(sections.values()):
            logger.warning("Failed to extract sections, using fallback")
            sections = {
                "CLINICAL_DETAILS": "",
                "FINDINGS": request.user_findings,
                "IMPRESSION": "",
                "RECOMMENDATIONS": ""
            }
        
        # Build full report with metadata if provided
        full_template = template.get("full_template_format", "")
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Default metadata if not provided
        patient_metadata = request.patient_metadata or {}
        default_metadata = {
            "PATIENT_NAME": patient_metadata.get("PATIENT_NAME", "N/A"),
            "PATIENT_ID": patient_metadata.get("PATIENT_ID", "N/A"),
            "PATIENT_AGE": patient_metadata.get("PATIENT_AGE", "N/A"),
            "PATIENT_SEX": patient_metadata.get("PATIENT_SEX", "N/A"),
            "STUDY_DATE": patient_metadata.get("STUDY_DATE", datetime.now().strftime("%Y-%m-%d")),
            "MODALITY": patient_metadata.get("MODALITY", template.get("modality", "N/A")),
            "REFERRING_PHYSICIAN": patient_metadata.get("REFERRING_PHYSICIAN", "N/A"),
            "REPORT_DATE": report_date
        }
        
        # Replace all placeholders
        full_report = full_template
        for key, value in {**default_metadata, **sections}.items():
            full_report = full_report.replace(f"{{{key}}}", str(value))
        
        return ReportGenerationResponse(
            medical_content=medical_content,
            full_report=full_report,
            status="success",
            raw_prompt=prompt,
            raw_output=raw_output
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post("/generate-report-custom", response_model=ReportGenerationResponse)
async def generate_report_custom(request: CustomReportGenerationRequest):
    """Generate radiology report with custom template parameters (specific sections only)"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Load generalized template by default
        templates_dir = get_templates_dir() / "general"
        template_file = templates_dir / "general_medical_analysis_001.json"
        if not template_file.exists():
            raise HTTPException(status_code=404, detail="Default template 'general_medical_analysis_001' not found")
        
        with open(template_file, 'r') as f:
            template = json.load(f)
        
        # Normalize template_parameters to uppercase for matching
        requested_sections = [s.upper().strip() for s in request.template_parameters]
        
        # Map user-provided section names to template field_ids
        # Handle common variations
        section_name_mapping = {
            "FINDINGS": "FINDINGS",
            "IMPRESSION": "IMPRESSION",
            "CLINICAL_DETAILS": "CLINICAL_DETAILS",
            "CLINICAL DETAILS": "CLINICAL_DETAILS",
            "RECOMMENDATIONS": "RECOMMENDATIONS",
            "OBSERVATION": "FINDINGS",  # Map OBSERVATION to FINDINGS
            "OBSERVATIONS": "FINDINGS",
            "OBESERVATION": "FINDINGS",  # Handle typo
        }
        
        # Get all medical sections from template
        medical_sections = template.get("medical_sections", [])
        
        # Create a mapping of field_id to section info
        section_map = {}
        for section in medical_sections:
            field_id = section.get("field_id", "").upper()
            section_map[field_id] = section
        
        # Map requested sections to field_ids
        requested_field_ids = []
        for req_section in requested_sections:
            # Try direct match first
            if req_section in section_map:
                requested_field_ids.append(req_section)
            # Try mapping
            elif req_section in section_name_mapping:
                mapped_id = section_name_mapping[req_section]
                if mapped_id in section_map:
                    requested_field_ids.append(mapped_id)
                else:
                    logger.warning(f"Section '{req_section}' mapped to '{mapped_id}' but not found in template")
            else:
                logger.warning(f"Section '{req_section}' not found in template or mapping")
        
        if not requested_field_ids:
            raise HTTPException(
                status_code=400, 
                detail=f"None of the requested sections {request.template_parameters} were found in the template. Available sections: {list(section_map.keys())}"
            )
        
        # Filter sections to only include requested ones
        filtered_sections = []
        for field_id in requested_field_ids:
            if field_id in section_map:
                section = section_map[field_id]
                if section.get("ai_generated", False):
                    filtered_sections.append(section)
                else:
                    logger.warning(f"Section '{field_id}' is not AI-generated, skipping")
        
        if not filtered_sections:
            raise HTTPException(
                status_code=400,
                detail="None of the requested sections are AI-generated. Please select sections that can be generated by the model."
            )
        
        # Build DICOM metadata example (for reference only)
        patient_metadata = request.patient_metadata or {}
        dicom_metadata_example = ""
        if patient_metadata:
            dicom_metadata_example = f"Patient's Sex: {patient_metadata.get('PATIENT_SEX', 'N/A')}\n"
            dicom_metadata_example += f"Patient's Age: {patient_metadata.get('PATIENT_AGE', 'N/A')}\n"
            dicom_metadata_example += f"Body Part Examined: {patient_metadata.get('BODY_PART', 'CHEST')}\n"
            dicom_metadata_example += f"Modality: {patient_metadata.get('MODALITY', template.get('modality', 'N/A'))}\n"
            if patient_metadata.get('VIEW_POSITION'):
                dicom_metadata_example += f"View Position: {patient_metadata.get('VIEW_POSITION')}"
        else:
            dicom_metadata_example = "Patient's Sex: N/A\nPatient's Age: N/A\nBody Part Examined: CHEST\nModality: X-Ray"
        
        # Build section list and headers from filtered sections
        section_list = ""
        section_headers = []
        for section in filtered_sections:
            section_name = section.get("section_name", "").upper()
            section_headers.append(section_name)
            instructions = section.get("instructions", "")
            section_list += f"- {section_name}: {instructions}\n"
        
        section_headers_str = ", ".join([f"{h}:" for h in section_headers])
        
        # Construct prompt with all placeholders
        prompt_template = template.get("prompt_template", "")
        
        # Get modality type and study context from template or patient metadata
        modality_type = "medical imaging"  # default
        study_context = "This is a general medical imaging study. Provide a comprehensive analysis based on the clinical observations."
        
        # Check if template has modality_specific_prompts
        modality_specific = template.get("modality_specific_prompts", {})
        patient_metadata_for_prompt = request.patient_metadata or {}
        
        # Determine modality from patient metadata or use default
        modality_from_metadata = patient_metadata_for_prompt.get("MODALITY", "").lower()
        if modality_from_metadata:
            # Map common modality names to keys
            modality_key = "general"
            if "x-ray" in modality_from_metadata or "xray" in modality_from_metadata or "radiology" in modality_from_metadata:
                modality_key = "radiology"
            elif "pathology" in modality_from_metadata or "histo" in modality_from_metadata:
                modality_key = "pathology"
            elif "dermatology" in modality_from_metadata or "skin" in modality_from_metadata:
                modality_key = "dermatology"
            elif "ophthalmology" in modality_from_metadata or "eye" in modality_from_metadata or "retina" in modality_from_metadata:
                modality_key = "ophthalmology"
            
            if modality_key in modality_specific:
                modality_type = modality_specific[modality_key].get("modality_type", modality_type)
                study_context = modality_specific[modality_key].get("study_context", study_context)
        
        # Replace all placeholders
        prompt = prompt_template.replace("{dicom_metadata_example}", dicom_metadata_example)
        prompt = prompt.replace("{section_list}", section_list.strip())
        prompt = prompt.replace("{section_headers}", section_headers_str)
        prompt = prompt.replace("{user_findings}", request.user_findings)
        prompt = prompt.replace("{modality_type}", modality_type)
        prompt = prompt.replace("{study_context}", study_context)
        
        logger.info("Generating custom report with template: general_medical_analysis_001")
        logger.info(f"Requested sections: {request.template_parameters}")
        logger.info(f"Mapped sections: {section_headers}")
        logger.info(f"User findings length: {len(request.user_findings)} characters")
        
        # Log the full prompt
        logger.info("=" * 80)
        logger.info("PROMPT SENT TO MODEL:")
        logger.info("=" * 80)
        logger.info(prompt)
        logger.info("=" * 80)
        
        # Format prompt as chat message for MedGemma
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Get model device
        if hasattr(model, 'device'):
            model_device = model.device
        else:
            model_device = next(model.parameters()).device
        
        # Apply chat template and tokenize
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        if hasattr(inputs, 'to'):
            try:
                inputs = inputs.to(model_device, dtype=torch.bfloat16)
            except Exception as e:
                logger.error(f"Error calling .to() on BatchEncoding: {e}")
                inputs = {k: v.to(model_device, dtype=torch.bfloat16) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
        elif isinstance(inputs, dict):
            inputs = {k: v.to(model_device, dtype=torch.bfloat16) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
        else:
            raise ValueError(f"apply_chat_template returned unexpected type: {type(inputs)}")
        
        # Get input_ids
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs.input_ids
        
        input_len = input_ids.shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.4
            )
            generation = generation[0][input_len:]
        
        # Decode the output
        raw_output = processor.decode(generation, skip_special_tokens=True)
        
        # Log the raw output
        logger.info("=" * 80)
        logger.info("RAW MODEL OUTPUT (BEFORE POST-PROCESSING):")
        logger.info("=" * 80)
        logger.info(raw_output)
        logger.info("=" * 80)
        
        # Post-process: Clean the output
        medical_content = clean_report_output(raw_output, section_headers)
        
        # Extract sections from medical content (only for requested sections)
        sections = extract_sections_from_content(medical_content, filtered_sections)
        
        # Validate and expand IMPRESSION if it's too short
        if "IMPRESSION" in sections and sections["IMPRESSION"]:
            findings_text = sections.get("FINDINGS", "")
            sections["IMPRESSION"] = validate_and_expand_impression(sections["IMPRESSION"], findings_text)
        
        # If sections not extracted, provide defaults
        if not any(sections.values()):
            logger.warning("Failed to extract sections, using fallback")
            sections = {}
            for section in filtered_sections:
                field_id = section.get("field_id", "")
                if field_id == "FINDINGS":
                    sections[field_id] = request.user_findings
                else:
                    sections[field_id] = ""
        
        # Build full report with only requested sections
        # Use medical_sections_format instead of full_template_format to avoid including all sections
        report_template = template.get("medical_sections_format", "")
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Default metadata if not provided
        patient_metadata = request.patient_metadata or {}
        default_metadata = {
            "PATIENT_NAME": patient_metadata.get("PATIENT_NAME", "N/A"),
            "PATIENT_ID": patient_metadata.get("PATIENT_ID", "N/A"),
            "PATIENT_AGE": patient_metadata.get("PATIENT_AGE", "N/A"),
            "PATIENT_SEX": patient_metadata.get("PATIENT_SEX", "N/A"),
            "STUDY_DATE": patient_metadata.get("STUDY_DATE", datetime.now().strftime("%Y-%m-%d")),
            "MODALITY": patient_metadata.get("MODALITY", template.get("modality", "N/A")),
            "REFERRING_PHYSICIAN": patient_metadata.get("REFERRING_PHYSICIAN", "N/A"),
            "REPORT_DATE": report_date
        }
        
        # Build report format with only requested sections
        # Create a custom format string with only the requested sections
        requested_section_names = [s.get("section_name", "").upper() for s in filtered_sections]
        report_format_parts = []
        for section in filtered_sections:
            field_id = section.get("field_id", "")
            section_name = section.get("section_name", "").upper()
            report_format_parts.append(f"{section_name}:\n{{{field_id}}}\n")
        
        report_template = "\n".join(report_format_parts)
        
        # Only include sections that were requested and generated
        requested_sections_dict = {}
        for section in filtered_sections:
            field_id = section.get("field_id", "")
            requested_sections_dict[field_id] = sections.get(field_id, "")
        
        # Replace all placeholders
        full_report = report_template
        for key, value in {**default_metadata, **requested_sections_dict}.items():
            full_report = full_report.replace(f"{{{key}}}", str(value))
        
        return ReportGenerationResponse(
            medical_content=medical_content,
            full_report=full_report,
            status="success",
            raw_prompt=prompt,
            raw_output=raw_output
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating custom report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating custom report: {str(e)}")

def generate_preview_html(template: dict, patient_metadata: dict = None) -> str:
    """Generate HTML preview of the report structure"""
    
    # Get metadata values
    metadata = patient_metadata or {}
    default_metadata = {
        "PATIENT_NAME": metadata.get("PATIENT_NAME", "[From DICOM]"),
        "PATIENT_ID": metadata.get("PATIENT_ID", "[From DICOM]"),
        "PATIENT_AGE": metadata.get("PATIENT_AGE", "[From DICOM]"),
        "PATIENT_SEX": metadata.get("PATIENT_SEX", "[From DICOM]"),
        "STUDY_DATE": metadata.get("STUDY_DATE", datetime.now().strftime("%Y-%m-%d")),
        "MODALITY": metadata.get("MODALITY", template.get("modality", "X-Ray")),
        "BODY_PART": metadata.get("BODY_PART", "CHEST"),
        "VIEW_POSITION": metadata.get("VIEW_POSITION", "[From DICOM]"),
        "REFERRING_PHYSICIAN": metadata.get("REFERRING_PHYSICIAN", "[From DICOM]")
    }
    
    # Build metadata section HTML
    metadata_html = f"""
    <div class="preview-metadata">
        <div class="metadata-row"><strong>Patient Name:</strong> {default_metadata['PATIENT_NAME']}</div>
        <div class="metadata-row"><strong>Patient ID:</strong> {default_metadata['PATIENT_ID']}</div>
        <div class="metadata-row"><strong>Age:</strong> {default_metadata['PATIENT_AGE']}</div>
        <div class="metadata-row"><strong>Sex:</strong> {default_metadata['PATIENT_SEX']}</div>
        <div class="metadata-row"><strong>Study Date:</strong> {default_metadata['STUDY_DATE']}</div>
        <div class="metadata-row"><strong>Modality:</strong> {default_metadata['MODALITY']}</div>
        <div class="metadata-row"><strong>Body Part:</strong> {default_metadata['BODY_PART']}</div>
    """
    
    if default_metadata['VIEW_POSITION'] and default_metadata['VIEW_POSITION'] != "[From DICOM]":
        metadata_html += f"<div class=\"metadata-row\"><strong>View Position:</strong> {default_metadata['VIEW_POSITION']}</div>"
    
    if default_metadata['REFERRING_PHYSICIAN'] and default_metadata['REFERRING_PHYSICIAN'] != "[From DICOM]":
        metadata_html += f"<div class=\"metadata-row\"><strong>Referring Physician:</strong> {default_metadata['REFERRING_PHYSICIAN']}</div>"
    
    metadata_html += "</div>"
    
    # Build medical sections HTML
    medical_sections = template.get("medical_sections", [])
    sections_html = ""
    
    for section in medical_sections:
        if section.get("ai_generated", False):
            section_name = section.get("section_name", "").upper()
            instructions = section.get("instructions", "")
            field_id = section.get("field_id", "")
            
            # Create placeholder text based on section type
            placeholder_text = ""
            if "CLINICAL" in section_name:
                placeholder_text = "[AI will generate: Clinical context and indication for study]"
            elif "FINDINGS" in section_name:
                placeholder_text = "[AI will generate: Detailed findings organized by anatomical region based on radiologist observations]"
            elif "IMPRESSION" in section_name:
                placeholder_text = "[AI will generate: Diagnostic impression summarizing key diagnoses]"
            elif "RECOMMENDATIONS" in section_name:
                placeholder_text = "[AI will generate: Follow-up recommendations if clinically indicated, otherwise 'None']"
            else:
                placeholder_text = f"[AI will generate: {instructions}]"
            
            sections_html += f"""
            <div class="preview-section">
                <div class="section-header">{section_name}:</div>
                <div class="section-placeholder">{placeholder_text}</div>
            </div>
            """
    
    # Complete HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Report Preview</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 40px 20px;
                background: #f5f5f5;
            }}
            .preview-container {{
                background: white;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                position: relative;
            }}
            .preview-watermark {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%) rotate(-45deg);
                font-size: 48px;
                color: rgba(0,0,0,0.05);
                font-weight: bold;
                pointer-events: none;
                z-index: 0;
            }}
            .preview-content {{
                position: relative;
                z-index: 1;
            }}
            .report-title {{
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 30px;
                border-bottom: 2px solid #333;
                padding-bottom: 10px;
            }}
            .preview-metadata {{
                margin-bottom: 30px;
                padding: 15px;
                background: #f9f9f9;
                border-left: 4px solid #667eea;
            }}
            .metadata-row {{
                margin: 8px 0;
                font-size: 14px;
            }}
            .metadata-row strong {{
                display: inline-block;
                width: 150px;
                color: #333;
            }}
            .preview-section {{
                margin: 25px 0;
                padding: 15px;
                background: #fafafa;
                border-left: 4px solid #4CAF50;
            }}
            .section-header {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            .section-placeholder {{
                font-style: italic;
                color: #666;
                padding: 10px;
                background: #fff;
                border: 1px dashed #ccc;
                border-radius: 4px;
                min-height: 40px;
            }}
            .preview-badge {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: #ff9800;
                color: white;
                padding: 5px 15px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="preview-container">
            <div class="preview-badge">PREVIEW</div>
            <div class="preview-watermark">PREVIEW</div>
            <div class="preview-content">
                <div class="report-title">RADIOLOGY REPORT</div>
                {metadata_html}
                {sections_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

@app.post("/preview-report", response_model=PreviewResponse)
async def preview_report(request: PreviewRequest):
    """Generate HTML preview of report structure"""
    try:
        # Load generalized template by default
        templates_dir = get_templates_dir() / "general"
        template_file = templates_dir / "general_medical_analysis_001.json"
        if not template_file.exists():
            raise HTTPException(status_code=404, detail="Default template 'general_medical_analysis_001' not found")
        
        with open(template_file, 'r') as f:
            template = json.load(f)
        
        # Generate preview HTML
        preview_html = generate_preview_html(template, request.patient_metadata)
        
        return PreviewResponse(
            preview_html=preview_html,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")

@app.post("/fill-template", response_model=OpenAIReportResponse)
async def generate_report_openai(
    findings: str = Form(...),
    template: str = Form(...),  # Template comes from form data (UI)
    model: str = Form("gpt-5.1"),  # Default to gpt-5.1
    patient_metadata: Optional[str] = Form(None),
    specialty: Optional[str] = Form(None)  # Optional specialty override
):
    """
    Generate radiology report using OpenAI's latest model.
    Accepts template from form data and uses prompt from prompts/template_filling_prompt.txt.
    """
    try:
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set"
            )
        
        client = OpenAI(api_key=api_key)
        
        # Parse patient metadata if provided
        patient_metadata_dict = {}
        dicom_metadata_text = ""
        normalized_modality = None
        normalized_body_part = None
        
        if patient_metadata:
            try:
                patient_metadata_dict = json.loads(patient_metadata)
                # Build metadata text for prompt
                if patient_metadata_dict:
                    dicom_metadata_text = f"Patient's Sex: {patient_metadata_dict.get('PATIENT_SEX', 'N/A')}\n"
                    dicom_metadata_text += f"Patient's Age: {patient_metadata_dict.get('PATIENT_AGE', 'N/A')}\n"
                    dicom_metadata_text += f"Body Part Examined: {patient_metadata_dict.get('BODY_PART', 'N/A')}\n"
                    dicom_metadata_text += f"Modality: {patient_metadata_dict.get('MODALITY', 'N/A')}\n"
                    if patient_metadata_dict.get('VIEW_POSITION'):
                        dicom_metadata_text += f"View Position: {patient_metadata_dict.get('VIEW_POSITION')}\n"

                    # Normalize modality/body part for routing
                    raw_modality = str(patient_metadata_dict.get('MODALITY', '') or '').strip().upper()
                    raw_body_part = str(patient_metadata_dict.get('BODY_PART', '') or '').strip().lower()
                    normalized_modality = raw_modality if raw_modality else None
                    normalized_body_part = raw_body_part if raw_body_part else None
            except:
                pass
        
        # Validate modality - only allow MRI, XRAY, CT, ULTRASOUND
        allowed_modalities = {"MRI", "XRAY", "CT", "ULTRASOUND"}
        if normalized_modality:
            if normalized_modality not in allowed_modalities:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported modality. Supported modalities: MRI, XRAY, CT, Ultrasound"
                )
        # If no modality provided, we still continue with a generic specialty
        
        # Determine specialty label (override > mapping > fallback)
        specialty_label = None
        if specialty and str(specialty).strip():
            specialty_label = str(specialty).strip()
        else:
            # Map (modality, body part) to specialty label
            bp = (normalized_body_part or "").lower()
            mod = (normalized_modality or "").upper()

            def is_match(value: str, candidates):
                v = value.lower()
                return any(c in v for c in candidates)

            if mod == "MRI":
                if is_match(bp, ["brain", "cerebr", "head", "pituitary", "spine"]):
                    specialty_label = "neuro-radiology"
                elif is_match(bp, ["cardiac", "heart"]):
                    specialty_label = "cardiac imaging"
                elif is_match(bp, ["abdomen", "pelvis", "liver", "renal", "kidney", "pancreas", "gi"]):
                    specialty_label = "abdominal radiology"
                else:
                    specialty_label = "radiology (MRI)"
            elif mod == "XRAY":
                if is_match(bp, ["chest", "lung", "thorax"]):
                    specialty_label = "chest radiology"
                elif is_match(bp, ["spine", "extremity", "knee", "hip", "shoulder", "humerus", "femur", "hand", "wrist", "ankle", "foot"]):
                    specialty_label = "MSK radiology"
                else:
                    specialty_label = "radiology (X-ray)"
            elif mod == "CT":
                if is_match(bp, ["chest", "lung", "thorax"]):
                    specialty_label = "chest radiology"
                elif is_match(bp, ["abdomen", "pelvis", "liver", "renal", "kidney", "pancreas", "gi"]):
                    specialty_label = "abdominal radiology"
                elif is_match(bp, ["brain", "head", "cerebr", "spine"]):
                    specialty_label = "neuro-radiology"
                else:
                    specialty_label = "radiology (CT)"
            elif mod == "ULTRASOUND":
                if is_match(bp, ["abdomen", "pelvis", "liver", "renal", "kidney", "pancreas", "gi", "obstetric", "ob"]):
                    specialty_label = "abdominal radiology"
                elif is_match(bp, ["cardiac", "heart", "echo"]):
                    specialty_label = "cardiac imaging"
                else:
                    specialty_label = "ultrasound imaging"
            else:
                # Fallback for missing/unknown modality
                specialty_label = "medical imaging"
        
        # Load the template filling prompt from file
        prompt_file_path = Path(__file__).parent.parent / "prompts" / "template_filling_prompt.txt"
        template_filling_prompt = ""
        if prompt_file_path.exists():
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                template_filling_prompt = f.read()
        else:
            # Fallback prompt if file doesn't exist
            template_filling_prompt = """You are an expert medical report assistant. Fill the provided template with the given findings while maintaining strict medical accuracy.

**CRITICAL FIRST RULE - REMOVE UNRELATED CONDITIONS:**
If the template contains any conditions, findings, or medical statements that are NOT mentioned in the provided findings, REMOVE them completely. Only keep conditions and findings that are explicitly stated in your provided findings.

**INSTRUCTIONS:**
1. Use ONLY information from the provided findings
2. Remove unrelated conditions from template
3. Preserve exact template structure and formatting
4. Maintain medical consistency - no contradictions
5. Return the complete filled template"""
        
        # Inject specialty label into the prompt template (if placeholder present)
        if "{specialty_label}" in template_filling_prompt:
            template_filling_prompt = template_filling_prompt.replace("{specialty_label}", specialty_label)
        
        # Parse template from form data (can be JSON string or plain text)
        template_text = template
        try:
            # Try to parse as JSON
            template_json = json.loads(template)
            if isinstance(template_json, dict) and "medical_sections_format" in template_json:
                template_text = template_json["medical_sections_format"]
            elif isinstance(template_json, dict) and "prompt_template" in template_json:
                template_text = template_json["prompt_template"]
        except:
            # If not JSON, use as-is
            pass
        
        # Build the final prompt using the template filling prompt
        prompt = f"""{template_filling_prompt}

**TEMPLATE TO FILL:**
{template_text}

**FINDINGS/OBSERVATIONS:**
{findings}

**PATIENT INFORMATION (for reference only):**
{dicom_metadata_text or "N/A"}

**TASK:**
Fill the template above with the provided findings, following all the rules. Return ONLY the filled template - do not include these instructions, the findings list, or patient information in your output. Return just the completed template."""
        
        # Call OpenAI API - use the model selected by the user
        model_name = model.strip() if model else "gpt-5.1"
        
        # Log which model is being used
        logger.info(f"Using OpenAI model: {model_name}")
        
        # For GPT-5.1, use the new responses API structure
        if model_name == "gpt-5.1" or model_name == "gpt-5":
            try:
                # GPT-5.1 uses responses.create() with different structure
                response = client.responses.create(
                    model="gpt-5",  # API uses "gpt-5" for GPT-5.1
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                report = response.output_text
            except Exception as model_error:
                # If GPT-5.1 API fails, try fallback to gpt-4o
                logger.warning(f"GPT-5.1 API failed, falling back to gpt-4o: {str(model_error)}")
                try:
                    # Fallback to standard chat API
                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.3,
                        max_tokens=2000
                    )
                    report = response.choices[0].message.content
                    model_name = "gpt-4o"
                except Exception as fallback_error:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Both GPT-5.1 and fallback failed: {str(fallback_error)}"
                    )
        # For o1 models, use different API structure
        elif model_name.startswith("o1"):
            messages = [{"role": "user", "content": prompt}]
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                report = response.choices[0].message.content
            except Exception as model_error:
                # If o1 model fails, fallback to gpt-4o
                if "model" in str(model_error).lower():
                    logger.warning(f"Model {model_name} failed, falling back to gpt-4o: {str(model_error)}")
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    report = response.choices[0].message.content
                    model_name = "gpt-4o"
                else:
                    raise
        else:
            # For regular models (gpt-4o, gpt-4-turbo, etc.)
            messages = [{"role": "user", "content": prompt}]
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                report = response.choices[0].message.content
            except Exception as model_error:
                # If model doesn't exist or fails, try with gpt-4o as fallback
                if "model" in str(model_error).lower() and ("not found" in str(model_error).lower() or "invalid" in str(model_error).lower() or "does not exist" in str(model_error).lower()):
                    logger.warning(f"Model {model_name} not available, falling back to gpt-4o: {str(model_error)}")
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    report = response.choices[0].message.content
                    model_name = "gpt-4o"  # Update model name for response
                else:
                    raise
        
        # Log successful generation with model used
        logger.info(f"Successfully generated report using model: {model_name}")
        
        return OpenAIReportResponse(
            report=report,
            status="success",
            openai_model=model_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating OpenAI report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating OpenAI report: {str(e)}")

@app.post("/interpret-sr", response_model=SrInterpretResponse)
async def interpret_sr(request: SrInterpretRequest):
    """
    Interpret DICOM Structured Report (SR) JSON using OpenAI and return a concise, clinically meaningful Findings text.
    Expects full SR metadata JSON (including nested 'Content Sequence' if present).
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)

        # Load SR interpretation prompt from file
        prompt_file_path = Path(__file__).parent.parent / "prompts" / "sr_interpretation_prompt.txt"
        if prompt_file_path.exists():
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                base_prompt = f.read()
        else:
            # Fallback minimal prompt
            base_prompt = (
                "You are an expert in interpreting DICOM Structured Reports (SR). "
                "Given the SR JSON content, extract ONLY clinically meaningful information and present it as a clear, concise set of findings. "
                "Ignore technical, display, or acquisition details that are not clinically relevant."
            )

        # Prepare SR JSON text (ensure it's JSON string, not Python str(dict))
        try:
            sr_json_text = json.dumps(request.sr_json, ensure_ascii=False, indent=2)
        except Exception:
            # As a fallback, stringify
            sr_json_text = str(request.sr_json)

        # Compose final prompt
        final_prompt = (
            f"{base_prompt}\n\n"
            "SR JSON:\n"
            "```\n"
            f"{sr_json_text}\n"
            "```\n\n"
            "Return a single section titled 'FINDINGS' containing the clinically meaningful summary only."
        )

        model_name = (request.model or "gpt-5.1").strip()
        logger.info(f"Interpreting SR with OpenAI model: {model_name}")

        findings_text = ""
        if model_name in ("gpt-5.1", "gpt-5"):
            try:
                response = client.responses.create(
                    model="gpt-5",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": final_prompt}
                            ]
                        }
                    ]
                )
                findings_text = response.output_text
            except Exception as model_error:
                logger.warning(f"GPT-5 API failed, falling back to gpt-4o: {model_error}")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.2,
                    max_tokens=1200
                )
                findings_text = response.choices[0].message.content
                model_name = "gpt-4o"
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,
                max_tokens=1200
            )
            findings_text = response.choices[0].message.content

        # Normalize to ensure it starts with FINDINGS header
        findings_text = findings_text.strip()
        if not findings_text.upper().startswith("FINDINGS"):
            findings_text = f"FINDINGS:\n{findings_text}"

        return SrInterpretResponse(findings=findings_text, status="success", openai_model=model_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error interpreting SR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interpreting SR: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

