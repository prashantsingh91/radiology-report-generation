# Quick Start Guide

## 🚀 Quick Setup (3 Steps)

### Step 1: Run Setup Script
```bash
cd /home/psingh/medgemma/medgemma-app
chmod +x setup.sh
./setup.sh
```

This will:
- ✅ Create Python virtual environment
- ✅ Install all backend dependencies
- ✅ Install all frontend dependencies

### Step 2: Authenticate with Hugging Face
```bash
source venv/bin/activate
huggingface-cli login
```

**Important**: You need access to the gated MedGemma repository:
- Visit: https://huggingface.co/google/medgemma-4b-it
- Request access if you haven't already

### Step 3: Start the Application

**Option A - Use start script (both services):**
```bash
./start.sh
```

**Option B - Manual start (two terminals):**

Terminal 1 (Backend):
```bash
source venv/bin/activate
cd backend
python main.py
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

## 🌐 Access the App

- **Frontend UI**: http://localhost:3006
- **Backend API**: http://localhost:8004
- **API Docs**: http://localhost:8004/docs

## 📝 Usage

1. Open http://localhost:3006 in your browser
2. Enter a medical prompt (e.g., "What are the symptoms of pneumonia?")
3. Click "Generate Response"
4. View the AI-generated response

## 📁 Project Structure

```
medgemma-app/
├── backend/           # FastAPI server
│   ├── main.py       # API endpoints
│   └── requirements.txt
├── frontend/         # React app
│   ├── src/
│   │   ├── App.jsx   # Main component
│   │   └── App.css   # Styles
│   └── package.json
├── .cache/           # Model cache (auto-created)
├── venv/             # Python virtual env (auto-created)
├── setup.sh          # Setup script
└── start.sh          # Start script
```

## ⚠️ First Run Notes

- **First time**: Model will download (~4GB) to `.cache/` folder
- **Subsequent runs**: Uses cached model (much faster)
- **GPU recommended**: Faster inference, but CPU works too

## 🐛 Troubleshooting

**Model not loading?**
- Check: `huggingface-cli whoami` (should show your username)
- Verify access to: https://huggingface.co/google/medgemma-4b-it

**Port already in use?**
- Backend uses port 8004
- Frontend uses port 3006
- Change ports in `backend/main.py` and `frontend/vite.config.js` if needed

**Dependencies issues?**
- Backend: `pip install -r backend/requirements.txt`
- Frontend: `cd frontend && npm install`
