# MedGemma App

A full-stack application featuring a FastAPI backend and React frontend for interacting with the MedGemma medical AI model.

## Features

- 🏥 **MedGemma Model Integration**: Uses the Google MedGemma-4B model for medical text generation
- 🚀 **FastAPI Backend**: High-performance Python API server
- ⚛️ **React Frontend**: Modern, responsive user interface
- 💾 **Model Caching**: Models are downloaded and cached in the `.cache` folder
- 🔄 **Real-time Generation**: Interactive prompt input and response generation

## Project Structure

```
medgemma-app/
├── backend/              # FastAPI application
│   ├── main.py          # Main API server
│   └── requirements.txt # Python dependencies
├── frontend/            # React application
│   ├── src/
│   │   ├── App.jsx     # Main React component
│   │   ├── App.css     # Styles
│   │   └── main.jsx    # Entry point
│   ├── package.json    # Node dependencies
│   └── vite.config.js  # Vite configuration
├── .cache/              # Model cache directory (created automatically)
├── venv/                # Python virtual environment (created by setup)
├── setup.sh             # Setup script
├── start.sh             # Start script
└── README.md           # This file
```

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher and npm
- Hugging Face account with access to the MedGemma model (gated repository)
- CUDA-capable GPU (recommended) or sufficient RAM for CPU inference

## Setup Instructions

### 1. Run the Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all backend dependencies
- Install all frontend dependencies

### 2. Authenticate with Hugging Face

The MedGemma model is a gated repository, so you need to authenticate:

```bash
source venv/bin/activate
huggingface-cli login
```

Enter your Hugging Face access token when prompted. You can get a token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### 3. Request Access to MedGemma

Visit the [MedGemma model page](https://huggingface.co/google/medgemma-4b-it) and request access if you haven't already.

## Running the Application

### Option 1: Use the Start Script (Recommended)

```bash
chmod +x start.sh
./start.sh
```

This starts both the backend and frontend services.

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
source venv/bin/activate
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Accessing the Application

- **Frontend**: http://localhost:3006
- **Backend API**: http://localhost:8004
- **API Docs**: http://localhost:8004/docs (Swagger UI)

## API Endpoints

### Health Check
```
GET /health
```

### Generate Text
```
POST /generate
Body: {
  "prompt": "Your medical question here",
  "max_length": 512
}
```

## Model Caching

The application will automatically detect and use the model from the following locations (in priority order):

1. **Default Hugging Face cache**: `~/.cache/huggingface/hub/models--google--medgemma-4b-it`
2. **Project cache**: `./.cache/models--google--medgemma-4b-it`
3. **Auto-download**: If not found, will download to the default Hugging Face cache

The cache structure follows Hugging Face's standard format:
```
~/.cache/huggingface/hub/models--google--medgemma-4b-it/
└── snapshots/
    └── [snapshot-hash]/
        └── [model files]
```

**Note**: If you already have the model downloaded at `~/.cache/huggingface/hub/models--google--medgemma-4b-it`, the application will use it automatically without re-downloading.

## Troubleshooting

### Model Not Loading

1. **Check Hugging Face Authentication:**
   ```bash
   huggingface-cli whoami
   ```

2. **Verify Access to MedGemma:**
   - Ensure you've requested and received access to the gated repository
   - Visit: https://huggingface.co/google/medgemma-4b-it

3. **Check Cache Directory:**
   - The model should be in `.cache/models--google--medgemma-4b-it/`
   - If corrupted, delete the cache and let it re-download

### Backend Errors

- Check that port 8004 is not already in use
- Verify all Python dependencies are installed: `pip install -r backend/requirements.txt`
- Check GPU availability if using CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Frontend Errors

- Ensure Node.js and npm are installed: `node --version && npm --version`
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check that the backend is running before starting the frontend

## Development

### Backend Development

The FastAPI backend uses hot-reload when running with uvicorn. Modify `main.py` and changes will be reflected automatically.

### Frontend Development

The React frontend uses Vite for fast hot module replacement. Changes to React components will update in real-time.

## Notes

- **First Run**: The first time you run the application, the model will be downloaded (several GB). This may take some time depending on your internet connection.
- **GPU Recommended**: While the model can run on CPU, GPU acceleration significantly improves inference speed.
- **Memory Requirements**: The model requires approximately 8-16GB of RAM/VRAM depending on your configuration.

## License

This project uses the MedGemma model, which is subject to Google's license terms. Please review the model's license on Hugging Face before use.

