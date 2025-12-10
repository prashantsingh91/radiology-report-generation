# Disk Space Management

## Current Status
- **Available Space**: ~13GB (after cleanup)
- **Usage**: 96% of 248GB

## What Was Done
- Cleaned pip cache (~8.5GB freed)
- Updated setup script to use `--no-cache-dir` flag to prevent cache buildup during installation

## Setup Script Improvements
The `setup.sh` script now:
1. ✅ Checks available disk space before installation
2. ✅ Automatically cleans pip cache if space is low (<5GB)
3. ✅ Uses `--no-cache-dir` flag to avoid filling cache during package installation

## If You Still Run Out of Space

### Option 1: Clean More Cache (Safe)
```bash
# Clean pip cache
pip cache purge

# Clean Hugging Face cache (if you have models elsewhere)
# Be careful - this will delete downloaded models
du -sh ~/.cache/huggingface/*
# Then selectively remove if needed
```

### Option 2: Use /mnt for Virtual Environment
If you need more space, you can create the venv on the /mnt mount (1.4TB available):

```bash
cd /home/psingh/medgemma/medgemma-app
python3 -m venv /mnt/medgemma-venv
source /mnt/medgemma-venv/bin/activate
cd backend
pip install --no-cache-dir -r requirements.txt
```

Then update `start.sh` to use `/mnt/medgemma-venv` instead of `./venv`.

### Option 3: Clean Up Large Directories
Check what's taking space:
```bash
# Check cache sizes
du -sh ~/.cache/* | sort -hr

# Check medgemma directory
du -sh /home/psingh/medgemma/* | sort -hr | head -10
```

## Recommended: Run Setup Now
With 13GB available, you should be able to complete the setup:

```bash
cd /home/psingh/medgemma/medgemma-app
./setup.sh
```

The script will now install packages without caching, saving disk space.

