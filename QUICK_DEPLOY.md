# Quick Deployment Guide

## Deploy Service (Single Command for Both Backend & Frontend)

```bash
cd /home/psingh/medgemma/medgemma-app
sudo ./deploy.sh
```

## Start Service

```bash
sudo systemctl start medgemma-app
```

## Check Status

```bash
sudo systemctl status medgemma-app
```

## View Logs

```bash
# Combined logs (backend + frontend)
sudo journalctl -u medgemma-app -f

# View last 100 lines
sudo journalctl -u medgemma-app -n 100
```

## Stop Service

```bash
sudo systemctl stop medgemma-app
```

## Restart Service

```bash
sudo systemctl restart medgemma-app
```

## Access URLs

**Local:**
- Frontend: http://localhost:3006
- Backend: http://localhost:8004

**External (replace YOUR_IP):**
- Frontend: http://YOUR_IP:3006
- Backend: http://YOUR_IP:8004

Find your IP:
```bash
hostname -I
```

## Firewall (if needed)

```bash
sudo ufw allow 3006/tcp
sudo ufw allow 8004/tcp
```

