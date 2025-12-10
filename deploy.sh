#!/bin/bash

# Deployment script for MedGemma App
# This script sets up systemd services for backend and frontend

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Deploying MedGemma App with systemd..."

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  This script needs sudo privileges to install systemd services"
    echo "   Please run: sudo $0"
    exit 1
fi

# Copy service file to systemd directory
echo "📋 Installing systemd service file..."
cp medgemma-app.service /etc/systemd/system/

# Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

# Enable service to start on boot
echo "✅ Enabling service to start on boot..."
systemctl enable medgemma-app.service

echo ""
echo "✅ Deployment complete!"
echo ""
echo "To start the service (both backend and frontend):"
echo "  sudo systemctl start medgemma-app"
echo ""
echo "To check status:"
echo "  sudo systemctl status medgemma-app"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u medgemma-app -f"
echo ""
echo "To stop service:"
echo "  sudo systemctl stop medgemma-app"
echo ""
echo "Ports used:"
echo "  • Backend: 8004 (no conflict with existing apps)"
echo "  • Frontend: 3006 (no conflict with existing apps)"
echo ""

