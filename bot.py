#!/usr/bin/env python3

import logging
import subprocess
import psutil
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters

# Configuration
BOT_TOKEN = "YOUR_BOT_TOKEN"  # Replaced by install.sh
LOG_FILE = "/var/log/system_monitor.log"
THRESHOLDS = {
    "gpu_util": 80,  # Alert if GPU utilization > 80%
    "mem_util": 80,  # Alert if memory utilization > 80%
    "vram_usage": 80,  # Alert if VRAM usage > 80%
    "cpu_usage": 80,  # Alert if CPU usage > 80%
    "ram_usage": 80,  # Alert if RAM usage > 80%
    "temp": 85,  # Alert if GPU temperature > 85°C
}
ALERT_CONFIG = {
    "vram_alert_threshold": 95,  # Alert if VRAM usage > 95%
    "snooze_duration": 300,  # Snooze alerts for 5 minutes
}

# Global state (thread-safe with dictionary access)
monitoring_jobs: Dict[int, Dict] = {}  # {chat_id: {job, message_id, interval}}
last_message_ids: Dict[int, int] = {}  # {chat_id: message_id}
alert_snooze: Dict[int, Dict[str, float]] = {}  # {chat_id: {metric: timestamp}}
PROCESS_ALERTS: Dict[int, Dict[str, Dict]] = {}  # {chat_id: {pid: {name, vram_threshold, start_time}}}
gpu_count: int = 0  # Number of GPUs detected
gpu_util_history: List[float] = []  # GPU utilization history for charts
cpu_usage_history: List[float] = []  # CPU usage history for charts

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inline keyboard with enhanced options
def get_keyboard(is_monitoring: bool = False, gpu_count: int = 1, chat_id: Optional[int] = None) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("📊 GPU Metrics", callback_data="gpu"),
            InlineKeyboardButton("💻 System Metrics", callback_data="system")
        ],
        [
            InlineKeyboardButton("🔄 GPU Processes", callback_data="processes"),
            InlineKeyboardButton("⚠️ Watch Process", callback_data="watch_process")
        ],
        [
            InlineKeyboardButton("📈 Status", callback_data="status"),
            InlineKeyboardButton("❓ Help", callback_data="help")
        ]
    ]
    if gpu_count > 1:
        gpu_buttons = [
            InlineKeyboardButton(f"GPU {i}", callback_data=f"gpu_{i}") for i in range(gpu_count)
        ]
        keyboard.insert(1, gpu_buttons)
    monitoring_buttons = []
    if is_monitoring:
        monitoring_buttons.append(InlineKeyboardButton("⏹️ Stop Monitoring", callback_data="stop_monitoring"))
    else:
        monitoring_buttons.append(InlineKeyboardButton("▶️ Start Monitoring", callback_data="start_monitoring"))
    if chat_id in PROCESS_ALERTS and PROCESS_ALERTS[chat_id]:
        monitoring_buttons.append(InlineKeyboardButton("🛑 Stop Watching", callback_data="stop_watching"))
    keyboard.append(monitoring_buttons)
    return InlineKeyboardMarkup(keyboard)

# ASCII chart generator with improved scaling
def generate_ascii_chart(values: List[float], width: int = 20, height: int = 5, label: str = "") -> str:
    if not values:
        return f"📉 *{label}*: No data available."
    max_val = max(values, default=100)
    min_val = min(values, default=0)
    if max_val == min_val:
        max_val += 1
    chart = []
    for i in range(height, -1, -1):
        line = []
        threshold = min_val + (i / height) * (max_val - min_val)
        for val in values:
            if val >= threshold:
                line.append("█")
            else:
                line.append(" ")
        chart.append("".join(line))
    return f"📉 *{label}* (Min: {min_val:.1f}, Max: {max_val:.1f}):\n```\n" + "\n".join(chart) + "\n```"

# Log messages with timestamp
def log_message(message: str) -> None:
    logger.info(message)

# Delete previous message to keep chat clean
async def delete_previous_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    if chat_id in last_message_ids:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=last_message_ids[chat_id])
            del last_message_ids[chat_id]
        except Exception as e:
            log_message(f"Failed to delete message {last_message_ids.get(chat_id, 'unknown')}: {e}")

# Detect GPU count with error handling
def detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        return max(1, int(result.stdout.strip()))
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
        log_message(f"Error detecting GPU count: {e}")
        return 1

# Get NVIDIA GPU metrics with caching and fan speed
async def get_gpu_metrics(gpu_index: int = 0, cache_duration: float = 2.0) -> str:
    global gpu_util_history
    cache_key = f"gpu_{gpu_index}_{int(time.time() // cache_duration)}"
    if cache_key in globals().get('_cache', {}):
        return globals()['_cache'][cache_key]

    try:
        result = subprocess.run(
            ['nvidia-smi', f'--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,fan.speed', f'--format=csv,noheader,nounits', f'--id={gpu_index}'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        index, gpu_name, gpu_util, mem_util, vram_total, vram_used, vram_free, temp, power, fan_speed = output.split(', ')
        
        gpu_name = gpu_name.strip()
        gpu_util = int(gpu_util.replace('%', '')) if '%' in gpu_util else 0
        mem_util = int(mem_util.replace('%', '')) if '%' in mem_util else 0
        vram_total = int(vram_total)
        vram_used = int(vram_used)
        vram_free = int(vram_free)
        temp = int(temp)
        power = float(power) if power != 'N/A' else 0.0
        fan_speed = int(fan_speed.replace('%', '')) if '%' in fan_speed else 0
        
        vram_usage = (vram_used / vram_total) * 100 if vram_total > 0 else 0
        
        gpu_util_history.append(gpu_util)
        if len(gpu_util_history) > 20:
            gpu_util_history.pop(0)
        chart = generate_ascii_chart(gpu_util_history, width=20, height=5, label="GPU Utilization (%)")
        
        message = f"📊 *GPU {index} Monitoring Report* - {subprocess.getoutput('hostname')} 🖥️\n"
        message += f"*GPU*: {gpu_name}\n"
        message += f"*GPU Utilization*: {gpu_util}% {'⚠️ High' if gpu_util > THRESHOLDS['gpu_util'] else ''}\n"
        message += f"*Memory Utilization*: {mem_util}% {'⚠️ High' if mem_util > THRESHOLDS['mem_util'] else ''}\n"
        message += f"*VRAM Usage*: {vram_used} MB / {vram_total} MB ({vram_usage:.2f}%) {'⚠️ High' if vram_usage > THRESHOLDS['vram_usage'] else ''}\n"
        message += f"*Temperature*: {temp}°C {'⚠️ High' if temp > THRESHOLDS['temp'] else ''}\n"
        message += f"*Power Draw*: {power:.1f} W\n"
        message += f"*Fan Speed*: {fan_speed}%\n"
        message += f"{chart}"
        
        globals().setdefault('_cache', {})[cache_key] = message
        log_message(f"GPU {index} Metrics: {gpu_name}, Util: {gpu_util}%, Mem: {mem_util}%, VRAM: {vram_used}/{vram_total} MB, Temp: {temp}°C, Power: {power} W, Fan: {fan_speed}%")
        return message
    except FileNotFoundError:
        message = "🚨 *Error*: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    except subprocess.CalledProcessError as e:
        message = f"🚨 *Error*: Failed to fetch GPU {gpu_index} metrics.\n*Details*: {e.stderr}"
        log_message(f"Error fetching GPU {gpu_index} metrics: {e}")
    return message

# Get system metrics with enhanced details
async def get_system_metrics() -> str:
    global cpu_usage_history
    try:
        cpu_usage = psutil.cpu_percent(interval=0.5)
        cpu_per_core = psutil.cpu_percent(interval=0.5, percpu=True)
        
        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024 ** 3)
        ram_used = ram.used / (1024 ** 3)
        ram_usage = ram.percent
        
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes / (1024 ** 3) if disk_io else 0
        disk_write = disk_io.write_bytes / (1024 ** 3) if disk_io else 0
        
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent / (1024 ** 2) if net_io else 0
        net_recv = net_io.bytes_recv / (1024 ** 2) if net_io else 0
        
        uptime_seconds = time.time() - psutil.boot_time()
        uptime = str(timedelta(seconds=int(uptime_seconds)))
        load_avg = psutil.getloadavg()
        
        cpu_usage_history.append(cpu_usage)
        if len(cpu_usage_history) > 20:
            cpu_usage_history.pop(0)
        chart = generate_ascii_chart(cpu_usage_history, width=20, height=5, label="CPU Usage (%)")
        
        message = f"💻 *System Monitoring Report* - {subprocess.getoutput('hostname')} 🖥️\n"
        message += f"*Uptime*: {uptime}\n"
        message += f"*Load Average*: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f} (1m, 5m, 15m)\n"
        message += f"*CPU Usage*: {cpu_usage:.1f}% {'⚠️ High' if cpu_usage > THRESHOLDS['cpu_usage'] else ''}\n"
        message += f"*CPU Per Core*: {', '.join([f'Core {i}: {p:.1f}%' for i, p in enumerate(cpu_per_core)])}\n"
        message += f"*RAM Usage*: {ram_used:.2f} GB / {ram_total:.2f} GB ({ram_usage:.1f}%) {'⚠️ High' if ram_usage > THRESHOLDS['ram_usage'] else ''}\n"
        message += f"*Disk I/O*: Read: {disk_read:.2f} GB, Write: {disk_write:.2f} GB\n"
        message += f"*Network I/O*: Sent: {net_sent:.2f} MB, Received: {net_recv:.2f} MB\n"
        message += f"{chart}"
        
        log_message(f"System Metrics: CPU: {cpu_usage}%, RAM: {ram_usage}%, Disk R/W: {disk_read}/{disk_write} GB, Net S/R: {net_sent}/{net_recv} MB")
        return message
    except Exception as e:
        log_message(f"Error fetching system metrics: {e}")
        return f"🚨 *Error*: Failed to fetch system metrics.\n*Details*: {str(e)}"

# Get GPU-related processes with runtime
async def get_gpu_processes() -> str:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory,gpu_uuid', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        processes = result.stdout.strip()
        if not processes:
            return "ℹ️ *No processes* are currently using any GPU."
        
        message = f"🔄 *GPU Processes* - {subprocess.getoutput('hostname')} 🖥️\n"
        for line in processes.split('\n'):
            pid, proc_name, mem, gpu_uuid = line.split(', ')
            gpu_index = get_gpu_index_from_uuid(gpu_uuid)
            try:
                proc = psutil.Process(int(pid))
                runtime = timedelta(seconds=int(time.time() - proc.create_time()))
                message += f"*GPU {gpu_index}* - *PID*: {pid}, *Name*: {proc_name}, *VRAM*: {mem}, *Runtime*: {runtime}\n"
            except psutil.NoSuchProcess:
                message += f"*GPU {gpu_index}* - *PID*: {pid}, *Name*: {proc_name}, *VRAM*: {mem}, *Runtime*: Unknown\n"
        
        log_message(f"GPU Processes: {processes}")
        return message
    except FileNotFoundError:
        return "🚨 *Error*: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    except subprocess.CalledProcessError as e:
        log_message(f"Error fetching GPU processes: {e}")
        return f"🚨 *Error*: Failed to fetch GPU processes.\n*Details*: {e.stderr}"

# Get GPU index from UUID with caching
def get_gpu_index_from_uuid(uuid: str) -> str:
    cache = globals().setdefault('_uuid_cache', {})
    if uuid in cache:
        return cache[uuid]
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.strip().split('\n'):
            index, gpu_uuid = line.split(', ')
            cache[gpu_uuid] = index
            if gpu_uuid == uuid:
                return index
        return "Unknown"
    except Exception as e:
        log_message(f"Error mapping GPU UUID: {e}")
        return "Unknown"

# List processes for watching with interactive buttons
async def list_processes_for_watching(chat_id: int) -> tuple[str, InlineKeyboardMarkup]:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        processes = result.stdout.strip()
        if not processes:
            return "ℹ️ *No processes* are currently using the GPU.", get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        
        message = "⚠️ *GPU Processes Available for Monitoring* 🔍\n"
        message += "Click a button to monitor a process or use `/watch <pid> [vram_threshold]` or `/watch <process_name> [vram_threshold]`.\n\n"
        buttons = []
        for line in processes.split('\n'):
            pid, proc_name, mem = line.split(', ')
            message += f"*PID*: {pid}, *Name*: {proc_name}, *VRAM*: {mem}\n"
            buttons.append([InlineKeyboardButton(f"Monitor {proc_name} (PID {pid})", callback_data=f"watch_pid_{pid}")])
        
        log_message(f"Listed processes for watching: {processes}")
        keyboard = InlineKeyboardMarkup(buttons + [[InlineKeyboardButton("⬅️ Back", callback_data="back")]])
        return message, keyboard
    except FileNotFoundError:
        return "🚨 *Error*: nvidia-smi not found. Ensure NVIDIA drivers are installed.", get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
    except subprocess.CalledProcessError as e:
        log_message(f"Error listing GPU processes: {e}")
        return f"🚨 *Error*: Failed to list GPU processes.\n*Details*: {e.stderr}", get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)

# Check process alerts with improved tracking
async def check_process_alerts(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    if chat_id not in PROCESS_ALERTS:
        return
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        processes = {pid: (name, mem) for line in result.stdout.strip().split('\n') for pid, name, mem in [line.split(', ')]} if result.stdout.strip() else {}
        
        for pid, config in list(PROCESS_ALERTS[chat_id].items()):
            if pid not in processes:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"ℹ️ *Process Stopped*: {config['name']} (PID {pid}) is no longer running.",
                    parse_mode='Markdown'
                )
                del PROCESS_ALERTS[chat_id][pid]
                continue
            
            proc_name, mem = processes[pid]
            mem_mb = int(re.findall(r'\d+', mem)[0]) if re.findall(r'\d+', mem) else 0
            if mem_mb > config["vram_threshold"]:
                runtime = timedelta(seconds=int(time.time() - config["start_time"]))
                await send_alert(
                    context,
                    chat_id,
                    f"🚨 *Process Alert*: {proc_name} (PID {pid}) is using {mem_mb} MB VRAM, exceeding threshold of {config['vram_threshold']} MB.\n*Runtime*: {runtime}"
                )
        
        if not PROCESS_ALERTS[chat_id]:
            del PROCESS_ALERTS[chat_id]
    except Exception as e:
        log_message(f"Error checking process alerts: {e}")

# Send alerts with improved formatting
async def send_alert(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message: str) -> None:
    await context.bot.send_message(
        chat_id=chat_id,
        text=message,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔇 Snooze Alerts (5m)", callback_data="snooze_alert")]
        ])
    )

# Check if alert is snoozed
def is_snoozed(chat_id: int, metric: str) -> bool:
    return chat_id in alert_snooze and metric in alert_snooze[chat_id] and time.time() < alert_snooze[chat_id][metric]

# Telegram command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global gpu_count, gpu_util_history, cpu_usage_history
    gpu_count = detect_gpu_count()
    gpu_util_history = []
    cpu_usage_history = []
    
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        f"👋 *Welcome to GPU & System Monitoring Bot* on {subprocess.getoutput('hostname')} 🚀\n"
        f"Detected {gpu_count} GPU(s). Ready to monitor your server!\n\n"
        "🔍 *What can I do?*\n"
        "- 📊 Check GPU usage, VRAM, and temperature\n"
        "- 💻 Monitor CPU, RAM, disk, and network\n"
        "- 🔄 List GPU processes\n"
        "- ⚠️ Watch specific processes\n"
        "- ▶️ Start continuous monitoring\n\n"
        "Use the buttons below or type /help for more!",
        reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count, chat_id=chat_id),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id
    log_message(f"Bot started for chat {chat_id}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        "❓ *GPU & System Monitoring Bot Help* 🛠️\n\n"
        "*Commands:*\n"
        "- /start: Initialize the bot\n"
        "- /gpu: Show GPU metrics\n"
        "- /system: Show system metrics\n"
        "- /processes: List GPU processes\n"
        "- /watch_process: List processes to monitor\n"
        "- /watch <pid/name> [vram_threshold]: Monitor a process\n"
        "- /status: Show monitoring and watched processes\n"
        "- /start_monitoring [interval]: Start continuous monitoring\n"
        "- /stop_monitoring: Stop continuous monitoring\n"
        "- /help: Show this help\n\n"
        "*Buttons:*\n"
        "- 📊 GPU Metrics\n"
        "- 💻 System Metrics\n"
        "- 🔄 GPU Processes\n"
        "- ⚠️ Watch Process\n"
        "- 📈 Status\n"
        "- ▶️ Start/⏹️ Stop Monitoring\n\n"
        "Click a button to explore!",
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id

async def gpu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await get_gpu_metrics(gpu_index=0),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
    )
    last_message_ids[chat_id] = message.message_id

async def system(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await get_system_metrics(),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
    )
    last_message_ids[chat_id] = message.message_id

async def processes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await get_gpu_processes(),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
    )
    last_message_ids[chat_id] = message.message_id

async def watch_process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message_text, keyboard = await list_processes_for_watching(chat_id)
    message = await update.message.reply_text(
        message_text,
        parse_mode='Markdown',
        reply_markup=keyboard
    )
    last_message_ids[chat_id] = message.message_id

async def watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    if not context.args:
        message = await update.message.reply_text(
            "⚠️ *Usage*: `/watch <pid/name> [vram_threshold]`\n"
            "Example: `/watch 1234 1000` or `/watch python 1000`\n"
            "Use /watch_process to list available processes.",
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
        last_message_ids[chat_id] = message.message_id
        return
    
    identifier = context.args[0]
    vram_threshold = 1000
    if len(context.args) > 1:
        try:
            vram_threshold = int(context.args[1])
            if vram_threshold < 0:
                raise ValueError("Threshold must be non-negative")
        except ValueError:
            message = await update.message.reply_text(
                "🚨 *Error*: VRAM threshold must be a non-negative number.\n"
                "Example: `/watch 1234 1000`",
                parse_mode='Markdown',
                reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
            )
            last_message_ids[chat_id] = message.message_id
            return
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        processes = result.stdout.strip().split('\n') if result.stdout.strip() else []
        for line in processes:
            pid, proc_name, mem = line.split(', ')
            if pid == identifier or proc_name.lower() == identifier.lower():
                if chat_id not in PROCESS_ALERTS:
                    PROCESS_ALERTS[chat_id] = {}
                PROCESS_ALERTS[chat_id][pid] = {
                    "name": proc_name,
                    "vram_threshold": vram_threshold,
                    "start_time": time.time()
                }
                message = await update.message.reply_text(
                    f"✅ *Monitoring Process*: {proc_name} (PID {pid}) for VRAM > {vram_threshold} MB.\n"
                    "Use /status to view all watched processes.",
                    parse_mode='Markdown',
                    reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
                )
                last_message_ids[chat_id] = message.message_id
                log_message(f"Started watching {proc_name} (PID {pid}) for chat {chat_id}")
                return
        message = await update.message.reply_text(
            f"🚨 *Error*: Process with PID or name '{identifier}' not found.\n"
            "Use /watch_process to list available processes.",
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
    except Exception as e:
        message = await update.message.reply_text(
            f"🚨 *Error*: Failed to fetch processes.\n*Details*: {str(e)}",
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
    last_message_ids[chat_id] = message.message_id

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = f"📈 *Monitoring Status* - {subprocess.getoutput('hostname')} 🖥️\n\n"
    if chat_id in monitoring_jobs:
        interval = monitoring_jobs[chat_id]["interval"]
        message += f"▶️ *Continuous Monitoring*: Active (every {interval} seconds)\n"
    else:
        message += "⏹️ *Continuous Monitoring*: Not active\n"
    
    if chat_id in PROCESS_ALERTS and PROCESS_ALERTS[chat_id]:
        message += "\n⚠️ *Watched Processes*:\n"
        for pid, config in PROCESS_ALERTS[chat_id].items():
            runtime = timedelta(seconds=int(time.time() - config["start_time"]))
            message += f"- *{config['name']}* (PID {pid}): VRAM > {config['vram_threshold']} MB, *Runtime*: {runtime}\n"
    else:
        message += "\n⚠️ *Watched Processes*: None\n"
    
    message += "\nUse buttons to start/stop monitoring or watch processes."
    message = await update.message.reply_text(
        message,
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
    )
    last_message_ids[chat_id] = message.message_id

async def start_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    if chat_id in monitoring_jobs:
        interval = monitoring_jobs[chat_id]["interval"]
        message = await update.message.reply_text(
            f"ℹ️ *Continuous monitoring* is already running (every {interval} seconds).\n"
            "Use the Stop Monitoring button to stop.",
            reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count, chat_id=chat_id),
            parse_mode='Markdown'
        )
        last_message_ids[chat_id] = message.message_id
        return
    
    interval = 60
    if context.args:
        try:
            interval = int(context.args[0])
            if interval < 10 or interval > 3600:
                raise ValueError("Interval must be between 10 and 3600 seconds")
        except ValueError:
            message = await update.message.reply_text(
                "🚨 *Error*: Interval must be a number between 10 and 3600 seconds.\n"
                "Example: `/start_monitoring 30`",
                parse_mode='Markdown',
                reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count, chat_id=chat_id)
            )
            last_message_ids[chat_id] = message.message_id
            return
    
    job = context.job_queue.run_repeating(
        monitor_callback,
        interval=interval,
        first=0,
        data=chat_id,
        name=str(chat_id)
    )
    monitoring_jobs[chat_id] = {"job": job, "message_id": None, "interval": interval}
    
    message = await update.message.reply_text(
        f"✅ *Started continuous monitoring* (every {interval} seconds).\n"
        f"Monitoring GPU and system metrics... 📊💻",
        reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count, chat_id=chat_id),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id
    monitoring_jobs[chat_id]["message_id"] = message.message_id
    log_message(f"Started monitoring for chat {chat_id} with interval {interval}s")

async def monitor_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = context.job.data
    if chat_id not in monitoring_jobs:
        return
    
    message_content = f"{await get_gpu_metrics(gpu_index=0)}\n\n{await get_system_metrics()}"
    await check_process_alerts(context, chat_id)
    
    try:
        if monitoring_jobs[chat_id]["message_id"]:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=monitoring_jobs[chat_id]["message_id"],
                text=message_content,
                parse_mode='Markdown',
                reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count, chat_id=chat_id)
            )
    except Exception as e:
        log_message(f"Error editing monitoring message: {e}")
        await delete_previous_message(context, chat_id)
        message = await context.bot.send_message(
            chat_id=chat_id,
            text=message_content,
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count, chat_id=chat_id)
        )
        monitoring_jobs[chat_id]["message_id"] = message.message_id
        last_message_ids[chat_id] = message.message_id

async def stop_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    if chat_id not in monitoring_jobs:
        message = await update.message.reply_text(
            "ℹ️ *No continuous monitoring* is running.",
            reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count, chat_id=chat_id),
            parse_mode='Markdown'
        )
        last_message_ids[chat_id] = message.message_id
        return
    
    monitoring_jobs[chat_id]["job"].schedule_removal()
    del monitoring_jobs[chat_id]
    
    message = await update.message.reply_text(
        "🛑 *Stopped continuous monitoring*.",
        reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count, chat_id=chat_id),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id
    log_message(f"Stopped monitoring for chat {chat_id}")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    await delete_previous_message(context, chat_id)
    
    if query.data.startswith("gpu_"):
        gpu_index = int(query.data.split("_")[1])
        message = await query.message.reply_text(
            await get_gpu_metrics(gpu_index=gpu_index),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
    elif query.data == "gpu":
        message = await query.message.reply_text(
            await get_gpu_metrics(gpu_index=0),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
    elif query.data == "system":
        message = await query.message.reply_text(
            await get_system_metrics(),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
    elif query.data == "processes":
        message = await query.message.reply_text(
            await get_gpu_processes(),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
    elif query.data == "watch_process":
        message_text, keyboard = await list_processes_for_watching(chat_id)
        message = await query.message.reply_text(
            message_text,
            parse_mode='Markdown',
            reply_markup=keyboard
        )
    elif query.data.startswith("watch_pid_"):
        pid = query.data.split("_")[2]
        vram_threshold = 1000
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            processes = result.stdout.strip().split('\n') if result.stdout.strip() else []
            for line in processes:
                proc_pid, proc_name, _ = line.split(', ')
                if proc_pid == pid:
                    if chat_id not in PROCESS_ALERTS:
                        PROCESS_ALERTS[chat_id] = {}
                    PROCESS_ALERTS[chat_id][pid] = {
                        "name": proc_name,
                        "vram_threshold": vram_threshold,
                        "start_time": time.time()
                    }
                    message = await query.message.reply_text(
                        f"✅ *Monitoring Process*: {proc_name} (PID {pid}) for VRAM > {vram_threshold} MB.\n"
                        "Use /status to view all watched processes.",
                        parse_mode='Markdown',
                        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
                    )
                    log_message(f"Started watching {proc_name} (PID {pid}) for chat {chat_id} via button")
                    break
            else:
                message = await query.message.reply_text(
                    f"🚨 *Error*: Process with PID {pid} not found.",
                    parse_mode='Markdown',
                    reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
                )
        except Exception as e:
            message = await query.message.reply_text(
                f"🚨 *Error*: Failed to fetch processes.\n*Details*: {str(e)}",
                parse_mode='Markdown',
                reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
            )
    elif query.data == "stop_watching":
        if chat_id not in PROCESS_ALERTS or not PROCESS_ALERTS[chat_id]:
            message = await query.message.reply_text(
                "ℹ️ *No processes* are being watched.",
                parse_mode='Markdown',
                reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
            )
        else:
            buttons = [
                [InlineKeyboardButton(f"Stop {config['name']} (PID {pid})", callback_data=f"stop_pid_{pid}")]
                for pid, config in PROCESS_ALERTS[chat_id].items()
            ]
            buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="back")])
            message = await query.message.reply_text(
                "🛑 *Select a process to stop watching*:",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(buttons)
            )
    elif query.data.startswith("stop_pid_"):
        pid = query.data.split("_")[2]
        if chat_id in PROCESS_ALERTS and pid in PROCESS_ALERTS[chat_id]:
            proc_name = PROCESS_ALERTS[chat_id][pid]["name"]
            del PROCESS_ALERTS[chat_id][pid]
            if not PROCESS_ALERTS[chat_id]:
                del PROCESS_ALERTS[chat_id]
            message = await query.message.reply_text(
                f"✅ *Stopped watching*: {proc_name} (PID {pid}).",
                parse_mode='Markdown',
                reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
            )
            log_message(f"Stopped watching {proc_name} (PID {pid}) for chat {chat_id}")
        else:
            message = await query.message.reply_text(
                f"ℹ️ *Process* with PID {pid} is not being watched.",
                parse_mode='Markdown',
                reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
            )
    elif query.data == "status":
        message = await query.message.reply_text(
            await status(None, context),  # Reuse status handler
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
    elif query.data == "help":
        message = await query.message.reply_text(
            await help_command(None, context),  # Reuse help handler
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
    elif query.data == "start_monitoring":
        if chat_id in monitoring_jobs:
            interval = monitoring_jobs[chat_id]["interval"]
            message = await query.message.reply_text(
                f"ℹ️ *Continuous monitoring* is already running (every {interval} seconds).\n"
                "Use the Stop Monitoring button to stop.",
                reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count, chat_id=chat_id),
                parse_mode='Markdown'
            )
        else:
            interval = 60
            job = context.job_queue.run_repeating(
                monitor_callback,
                interval=interval,
                first=0,
                data=chat_id,
                name=str(chat_id)
            )
            monitoring_jobs[chat_id] = {"job": job, "message_id": None, "interval": interval}
            message = await query.message.reply_text(
                f"✅ *Started continuous monitoring* (every {interval} seconds).\n"
                f"Monitoring GPU and system metrics... 📊💻",
                reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count, chat_id=chat_id),
                parse_mode='Markdown'
            )
            monitoring_jobs[chat_id]["message_id"] = message.message_id
            log_message(f"Started monitoring for chat {chat_id} with interval {interval}s")
    elif query.data == "stop_monitoring":
        if chat_id not in monitoring_jobs:
            message = await query.message.reply_text(
                "ℹ️ *No continuous monitoring* is running.",
                reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count, chat_id=chat_id),
                parse_mode='Markdown'
            )
        else:
            monitoring_jobs[chat_id]["job"].schedule_removal()
            del monitoring_jobs[chat_id]
            message = await query.message.reply_text(
                "🛑 *Stopped continuous monitoring*.",
                reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count, chat_id=chat_id),
                parse_mode='Markdown'
            )
            log_message(f"Stopped monitoring for chat {chat_id}")
    elif query.data == "snooze_alert":
        alert_snooze[chat_id] = {
            "vram": time.time() + ALERT_CONFIG["snooze_duration"],
            "temp": time.time() + ALERT_CONFIG["snooze_duration"]
        }
        message = await query.message.reply_text(
            "🔇 *Alerts snoozed* for 5 minutes.",
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id),
            parse_mode='Markdown'
        )
    elif query.data == "back":
        message = await query.message.reply_text(
            "🔙 *Back to main menu*",
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
        )
    
    last_message_ids[chat_id] = message.message_id

# Handle unknown commands
async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    message = await update.message.reply_text(
        "🤔 *Unknown command*.\nUse /help to see available commands or try the buttons below.",
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count, chat_id=chat_id)
    )
    last_message_ids[chat_id] = message.message_id

def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("gpu", gpu))
    app.add_handler(CommandHandler("system", system))
    app.add_handler(CommandHandler("processes", processes))
    app.add_handler(CommandHandler("watch_process", watch_process))
    app.add_handler(CommandHandler("watch", watch))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("start_monitoring", start_monitoring))
    app.add_handler(CommandHandler("stop_monitoring", stop_monitoring))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    app.run_polling()

if __name__ == '__main__':
    main()
