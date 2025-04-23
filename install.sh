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
     
     X: x.com/0xsumitro   

EOF

echo -e "${GREEN}Starting GPU Monitoring Bot installation...${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo).${NC}"
    exit 1
fi

# Remove old files
echo -e "${GREEN}Cleaning old files...${NC}"
rm -rf monitoring
rm -rf /var/log/system_monitor.log


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
pip install "python-telegram-bot[job-queue]"
pip install -r requirements.txt

# Get Telegram bot token
echo -e "${GREEN}Please enter your Telegram Bot Token (obtained from @BotFather):${NC}"
echo -e "${GREEN}Example: 1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890${NC}"
echo -e "${GREEN}You have 3 attempts to enter a valid token.${NC}"
attempts=3
while [ $attempts -gt 0 ]; do
    read -r BOT_TOKEN </dev/tty || {
        echo -e "${RED}Error: Cannot read input. Run the script locally: curl -s $REPO_URL/main/install.sh > install.sh; sudo bash install.sh${NC}"
        exit 1
    }
    if [ -z "$BOT_TOKEN" ]; then
        attempts=$((attempts-1))
        echo -e "${RED}Bot token is required! $attempts attempts remaining.${NC}"
        continue
    fi
    # Basic token format validation (e.g., number:alphanumeric)
    if echo "$BOT_TOKEN" | grep -qE '^[0-9]+:[A-Za-z0-9_-]+$'; then
        echo -e "${GREEN}Token received. Proceeding...${NC}"
        break
    else
        attempts=$((attempts-1))
        echo -e "${RED}Invalid token format! Must be like '1234567890:XYZ...'. $attempts attempts remaining.${NC}"
    fi
done
if [ $attempts -eq 0 ]; then
    echo -e "${RED}Failed to provide a valid token after 3 attempts. Exiting.${NC}"
    exit 1
fi

# Update bot.py with the token
sed -i "s/YOUR_BOT_TOKEN/$BOT_TOKEN/" bot.py

# Make bot.py executable
chmod +x bot.py

# Ensure log file permissions
touch /var/log/system_monitor.log
chmod 666 /var/log/system_monitor.log

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
echo -e "Logs are saved to /var/log/system_monitor.log"
