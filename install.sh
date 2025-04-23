#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "\033[0m"
cat << "EOF"
     ...#####...##.....##..######..##.....##.##.....##.####.########.########...#######.
     ..##...##...##...##..##....##.##.....##.###...###..##.....##....##.....##.##.....##
     .##.....##...##.##...##.......##.....##.####.####..##.....##....##.....##.##.....##
     .##.....##....###.....######..##.....##.##.###.##..##.....##....########..##.....##
     .##.....##...##.##.........##.##.....##.##.....##..##.....##....##...##...##.....##
     ..##...##...##...##..##....##.##.....##.##.....##..##.....##....##....##..##.....##
     ...#####...##.....##..######...#######..##.....##.####....##....##.....##..#######.
     
     x.com/0xsumitro   

EOF

echo -e "${GREEN}Starting GPU Monitoring Bot installation...${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo).${NC}"
    exit 1
fi

# Install system dependencies
echo -e "${GREEN}Installing system dependencies...${NC}"
apt update
# apt install -y python3 python3-pip screen nvidia-driver-535 nvidia-utils-535

# Create project directory
INSTALL_DIR="/root/monitoring"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone the GitHub repository
REPO_URL="https://github.com/xmmitro/gpu-monitoring"
echo -e "${GREEN}Cloning repository from $REPO_URL...${NC}"
git clone "$REPO_URL" . || {
    echo -e "${RED}Failed to clone repository. Check the URL or network.${NC}"
    exit 1
}

# Create virtual environment
echo -e "${GREEN}Setting up Python virtual environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Get Telegram bot token
echo -e "${GREEN}Please enter your Telegram Bot Token (obtained from @BotFather):${NC}"
echo -e "${GREEN}Example: 1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890${NC}"
while true; do
    read -r BOT_TOKEN
    if [ -z "$BOT_TOKEN" ]; then
        echo -e "${RED}Bot token is required! Please enter a valid token.${NC}"
    else
        echo -e "${GREEN}Token received. Proceeding...${NC}"
        break
    fi
done

# Update bot.py with the token
sed -i "s/YOUR_BOT_TOKEN/$BOT_TOKEN/" bot.py

# Make bot.py executable
chmod +x bot.py

# Ensure log file permissions
touch /var/log/gpu_monitor.log
chmod 666 /var/log/gpu_monitor.log

# Start bot in a screen session
echo -e "${GREEN}Starting bot in a screen session named 'gpubot'...${NC}"
screen -dmS gpubot bash -c "./bot.py; exec bash"

# Add to crontab for auto-start
# echo -e "${GREEN}Adding bot to crontab for auto-start on reboot...${NC}"
# (crontab -l 2>/dev/null; echo "@reboot /root/live-monitoring/.venv/bin/python3 /root/live-monitoring/bot.py") | crontab -

echo -e "${GREEN}Installation complete!${NC}"
echo -e "The bot is running in a screen session named 'gpubot'."
echo -e "To check the bot, reattach with: ${GREEN}screen -r gpubot${NC}"
echo -e "To start interacting, open Telegram and send /start to your bot."
echo -e "Logs are saved to /var/log/gpu_monitor.log"
