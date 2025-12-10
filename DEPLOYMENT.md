# MedGemma App Deployment Guide

This guide explains how to deploy the MedGemma application using systemd services for external access.

## Prerequisites

- Backend and frontend dependencies installed
- Virtual environment set up
- Node.js and npm installed
- sudo/root access for systemd service installation

## Deployment Steps

### 1. Update Frontend Configuration

The frontend has been configured to:
- Listen on `0.0.0.0` (all interfaces) for external access
- Automatically detect the API URL based on the hostname

### 2. Install Systemd Services

Run the deployment script with sudo:

```bash
cd /home/psingh/medgemma/medgemma-app
sudo ./deploy.sh
```

This will:
- Copy service files to `/etc/systemd/system/`
- Enable services to start on boot
- Reload systemd daemon

### 3. Start Services

```bash
# Start backend
sudo systemctl start medgemma-backend

# Start frontend
sudo systemctl start medgemma-frontend
```

### 4. Check Service Status

```bash
# Check backend status
sudo systemctl status medgemma-backend

# Check frontend status
sudo systemctl status medgemma-frontend
```

### 5. View Logs

```bash
# Backend logs
sudo journalctl -u medgemma-backend -f

# Frontend logs
sudo journalctl -u medgemma-frontend -f

# View last 100 lines
sudo journalctl -u medgemma-backend -n 100
```

## Service Management

### Start Service (Starts Both Backend & Frontend)
```bash
sudo systemctl start medgemma-app
```

### Stop Service
```bash
sudo systemctl stop medgemma-app
```

### Restart Service
```bash
sudo systemctl restart medgemma-app
```

### Enable/Disable Auto-start on Boot
```bash
# Enable
sudo systemctl enable medgemma-app

# Disable
sudo systemctl disable medgemma-app
```

## Accessing the Application

### Local Access
- **Frontend**: http://localhost:3006
- **Backend API**: http://localhost:8004
- **API Docs**: http://localhost:8004/docs

### External Access

Replace `YOUR_SERVER_IP` with your server's IP address:

- **Frontend**: http://YOUR_SERVER_IP:3006
- **Backend API**: http://YOUR_SERVER_IP:8004
- **API Docs**: http://YOUR_SERVER_IP:8004/docs

To find your server IP:
```bash
hostname -I
```

## Firewall Configuration

If you have a firewall enabled, you may need to open ports 3006 and 8004:

### UFW (Ubuntu)
```bash
sudo ufw allow 3006/tcp
sudo ufw allow 8004/tcp
sudo ufw reload
```

### firewalld (CentOS/RHEL)
```bash
sudo firewall-cmd --permanent --add-port=3006/tcp
sudo firewall-cmd --permanent --add-port=8004/tcp
sudo firewall-cmd --reload
```

## Troubleshooting

### Service Won't Start

1. Check service status:
   ```bash
   sudo systemctl status medgemma-app
   ```

2. Check logs:
   ```bash
   sudo journalctl -u medgemma-app -n 50
   ```

3. Check individual process logs:
   ```bash
   # Backend logs
   tail -f /home/psingh/medgemma/medgemma-app/backend.log
   
   # Frontend logs
   tail -f /home/psingh/medgemma/medgemma-app/frontend.log
   ```

4. Verify paths in service file:
   ```bash
   sudo cat /etc/systemd/system/medgemma-app.service
   ```

### Port Already in Use

If ports 3006 or 8004 are already in use:

1. Find the process:
   ```bash
   sudo lsof -i :8004
   sudo lsof -i :3006
   ```

2. Kill the process or update service files to use different ports

### Frontend Can't Connect to Backend

1. Verify backend is running:
   ```bash
   curl http://localhost:8004/health
   ```

2. Check if backend is listening on correct interface:
   ```bash
   sudo netstat -tlnp | grep 8004
   ```

3. Update frontend API URL if needed (check `frontend/src/App.jsx`)

## Manual Service File Location

Service file is located at:
- `/etc/systemd/system/medgemma-app.service`

To edit it:
```bash
sudo systemctl edit --full medgemma-app
```

After editing, reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart medgemma-app
```

## Port Information

**MedGemma App Ports:**
- Backend: **8004** (no conflict - other apps use 8000, 8001)
- Frontend: **3006** (no conflict - other apps use 3000)

**Other Apps on System:**
- Port 8000: face-alert-app backend
- Port 8001: Another app backend
- Port 3000: face-alert-app frontend

## Notes

- Services run as user `psingh` - update in service files if different
- Backend uses virtual environment at `/home/psingh/medgemma/medgemma-app/venv`
- Frontend runs in development mode (`npm run dev`) - for production, consider building and serving static files
- Logs are available via `journalctl` and also written to systemd journal
