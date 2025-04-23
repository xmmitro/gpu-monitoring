#!/usr/bin/env python3

import logging
import subprocess
import psutil
import time
import re
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Configuration
BOT_TOKEN = "8091891864:AAEfUJ97iZ77bKEq5Ysn1XIu6rKA6F6I1hQ"  # Replace with your Telegram Bot Token
LOG_FILE = "/var/log/gpu_monitor.log"
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
PROCESS_ALERTS = {}  # Store process-specific alerts {chat_id: {pid: {name, vram_threshold}}}

# Global state
monitoring_jobs = {}  # {chat_id: {job, message_id}}
last_message_ids = {}  # {chat_id: message_id}
alert_snooze = {}  # {chat_id: {metric: timestamp}}
gpu_count = 0  # Number of GPUs detected

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

# Inline keyboard
def get_keyboard(is_monitoring: bool = False, gpu_count: int = 1) -> InlineKeyboardMarkup:
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
            InlineKeyboardButton("❓ Help", callback_data="help")
        ]
    ]
    if gpu_count > 1:
        gpu_buttons = [
            InlineKeyboardButton(f"GPU {i}", callback_data=f"gpu_{i}") for i in range(gpu_count)
        ]
        keyboard.insert(1, gpu_buttons)
    if is_monitoring:
        keyboard.append([InlineKeyboardButton("⏹️ Stop Monitoring", callback_data="stop_monitoring")])
    else:
        keyboard.append([InlineKeyboardButton("▶️ Start Monitoring", callback_data="start_monitoring")])
    return InlineKeyboardMarkup(keyboard)

# ASCII chart generator
def generate_ascii_chart(values: list, width: int = 20, height: int = 5) -> str:
    if not values:
        return "No data for chart."
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
    return "```\n" + "\n".join(chart) + "\n```"

# Function to log messages
def log_message(message: str) -> None:
    logger.info(message)

# Function to delete previous message
async def delete_previous_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    if chat_id in last_message_ids:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=last_message_ids[chat_id])
        except Exception as e:
            log_message(f"Failed to delete message {last_message_ids[chat_id]}: {e}")
        del last_message_ids[chat_id]

# Function to detect GPU count
def detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 1  # Assume single GPU if detection fails

# Function to get NVIDIA GPU metrics
async def get_gpu_metrics(gpu_index: int = 0) -> str:
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw', f'--format=csv,noheader,nounits', f'--id={gpu_index}'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        index, gpu_name, gpu_util, mem_util, vram_total, vram_used, vram_free, temp, power = output.split(', ')
        
        gpu_name = gpu_name.strip()
        gpu_util = int(gpu_util.replace('%', ''))
        mem_util = int(mem_util.replace('%', ''))
        vram_total = int(vram_total)
        vram_used = int(vram_used)
        vram_free = int(vram_free)
        temp = int(temp)
        power = float(power)
        
        vram_usage = (vram_used / vram_total) * 100
        
        # Generate ASCII chart for GPU utilization
        gpu_util_history.append(gpu_util)
        if len(gpu_util_history) > 20:
            gpu_util_history.pop(0)
        chart = generate_ascii_chart(gpu_util_history, width=20, height=5)
        
        message = f"📊 *GPU {index} Monitoring Report* - {subprocess.getoutput('hostname')}\n"
        message += f"*GPU*: {gpu_name}\n"
        message += f"*GPU Utilization*: {gpu_util}% {'⚠️ High' if gpu_util > THRESHOLDS['gpu_util'] else ''}\n"
        message += f"*Memory Utilization*: {mem_util}% {'⚠️ High' if mem_util > THRESHOLDS['mem_util'] else ''}\n"
        message += f"*VRAM Usage*: {vram_used} MB / {vram_total} MB ({vram_usage:.2f}%) {'⚠️ High' if vram_usage > THRESHOLDS['vram_usage'] else ''}\n"
        message += f"*Temperature*: {temp}°C {'⚠️ High' if temp > THRESHOLDS['temp'] else ''}\n"
        message += f"*Power Draw*: {power} W\n"
        message += f"*Utilization Chart*:\n{chart}"
        
        # Check for critical alerts
        if vram_usage > ALERT_CONFIG["vram_alert_threshold"] and not is_snoozed(chat_id, "vram"):
            await send_alert(context, chat_id, f"🚨 *Critical VRAM Usage* on GPU {index}: {vram_usage:.2f}%")
        if temp > THRESHOLDS["temp"] and not is_snoozed(chat_id, "temp"):
            await send_alert(context, chat_id, f"🚨 *High Temperature* on GPU {index}: {temp}°C")
        
        log_message(f"GPU {index} Metrics: {gpu_name}, Util: {gpu_util}%, Mem: {mem_util}%, VRAM: {vram_used}/{vram_total} MB, Temp: {temp}°C, Power: {power} W")
        return message
    except FileNotFoundError:
        return "🚨 *Error*: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    except subprocess.CalledProcessError as e:
        log_message(f"Error fetching GPU {gpu_index} metrics: {e}")
        return f"🚨 *Error*: Failed to fetch GPU {gpu_index} metrics.\n*Details*: {e.stderr}"

# Function to get system metrics
async def get_system_metrics() -> str:
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        
        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024 ** 3)
        ram_used = ram.used / (1024 ** 3)
        ram_usage = ram.percent
        
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes / (1024 ** 3)
        disk_write = disk_io.write_bytes / (1024 ** 3)
        
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent / (1024 ** 2)
        net_recv = net_io.bytes_recv / (1024 ** 2)
        
        # System uptime and load
        uptime_seconds = time.time() - psutil.boot_time()
        uptime = f"{int(uptime_seconds // 86400)}d {int((uptime_seconds % 86400) // 3600)}h {int((uptime_seconds % 3600) // 60)}m"
        load_avg = psutil.getloadavg()
        
        # ASCII chart for CPU usage
        cpu_usage_history.append(cpu_usage)
        if len(cpu_usage_history) > 20:
            cpu_usage_history.pop(0)
        chart = generate_ascii_chart(cpu_usage_history, width=20, height=5)
        
        message = f"💻 *System Monitoring Report* - {subprocess.getoutput('hostname')}\n"
        message += f"*Uptime*: {uptime}\n"
        message += f"*Load Average*: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f} (1m, 5m, 15m)\n"
        message += f"*CPU Usage*: {cpu_usage}% {'⚠️ High' if cpu_usage > THRESHOLDS['cpu_usage'] else ''}\n"
        message += f"*CPU Per Core*: {', '.join([f'{i}: {p}%' for i, p in enumerate(cpu_per_core)])}\n"
        message += f"*RAM Usage*: {ram_used:.2f} GB / {ram_total:.2f} GB ({ram_usage:.2f}%) {'⚠️ High' if ram_usage > THRESHOLDS['ram_usage'] else ''}\n"
        message += f"*Disk I/O*: Read: {disk_read:.2f} GB, Write: {disk_write:.2f} GB\n"
        message += f"*Network I/O*: Sent: {net_sent:.2f} MB, Received: {net_recv:.2f} MB\n"
        message += f"*CPU Usage Chart*:\n{chart}"
        
        log_message(f"System Metrics: CPU: {cpu_usage}%, RAM: {ram_usage}%, Disk R/W: {disk_read}/{disk_write} GB, Net S/R: {net_sent}/{net_recv} MB")
        return message
    except Exception as e:
        log_message(f"Error fetching system metrics: {e}")
        return f"🚨 *Error*: Failed to fetch system metrics.\n*Details*: {str(e)}"

# Function to get GPU-related processes
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
        
        message = f"🔄 *GPU Processes* - {subprocess.getoutput('hostname')}\n"
        for line in processes.split('\n'):
            pid, proc_name, mem, gpu_uuid = line.split(', ')
            gpu_index = get_gpu_index_from_uuid(gpu_uuid)
            message += f"*GPU {gpu_index}* - *PID*: {pid}, *Name*: {proc_name}, *VRAM*: {mem}\n"
        
        log_message(f"GPU Processes: {processes}")
        return message
    except FileNotFoundError:
        return "🚨 *Error*: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    except subprocess.CalledProcessError as e:
        log_message(f"Error fetching GPU processes: {e}")
        return f"🚨 *Error*: Failed to fetch GPU processes.\n*Details*: {e.stderr}"

# Function to get GPU index from UUID
def get_gpu_index_from_uuid(uuid: str) -> str:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.strip().split('\n'):
            index, gpu_uuid = line.split(', ')
            if gpu_uuid == uuid:
                return index
        return "Unknown"
    except:
        return "Unknown"

# Function to list processes for watching
async def list_processes_for_watching() -> str:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        processes = result.stdout.strip()
        if not processes:
            return "ℹ️ *No processes* are currently using the GPU."
        
        message = "⚠️ *GPU Processes Available for Monitoring*\n"
        message += "Reply with `/watch <pid>` or `/watch <process_name>` to monitor a process.\n\n"
        for line in processes.split('\n'):
            pid, proc_name, mem = line.split(', ')
            message += f"*PID*: {pid}, *Name*: {proc_name}, *VRAM*: {mem}\n"
        
        log_message(f"Listed processes for watching: {processes}")
        return message
    except FileNotFoundError:
        return "🚨 *Error*: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    except subprocess.CalledProcessError as e:
        log_message(f"Error listing GPU processes: {e}")
        return f"🚨 *Error*: Failed to list GPU processes.\n*Details*: {e.stderr}"

# Function to check process alerts
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
        processes = result.stdout.strip().split('\n')
        for pid, config in PROCESS_ALERTS[chat_id].items():
            for line in processes:
                proc_pid, proc_name, mem = line.split(', ')
                if proc_pid == pid or proc_name == config["name"]:
                    mem_mb = int(re.findall(r'\d+', mem)[0])
                    if mem_mb > config["vram_threshold"]:
                        await send_alert(
                            context,
                            chat_id,
                            f"🚨 *Process Alert*: {proc_name} (PID {proc_pid}) is using {mem_mb} MB VRAM, exceeding threshold of {config['vram_threshold']} MB."
                        )
                    break
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"ℹ️ *Process Alert*: {config['name']} (PID {pid}) is no longer running.",
                    parse_mode='Markdown'
                )
                del PROCESS_ALERTS[chat_id][pid]
                if not PROCESS_ALERTS[chat_id]:
                    del PROCESS_ALERTS[chat_id]
    except Exception as e:
        log_message(f"Error checking process alerts: {e}")

# Function to send alerts
async def send_alert(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message: str) -> None:
    await context.bot.send_message(
        chat_id=chat_id,
        text=message,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔇 Snooze Alerts (5m)", callback_data="snooze_alert")]
        ])
    )

# Function to check if alert is snoozed
def is_snoozed(chat_id: int, metric: str) -> bool:
    if chat_id in alert_snooze and metric in alert_snooze[chat_id]:
        return time.time() < alert_snooze[chat_id][metric]
    return False

# Telegram command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global gpu_count, gpu_util_history, cpu_usage_history
    gpu_count = detect_gpu_count()
    gpu_util_history = []  # Reset history for ASCII charts
    cpu_usage_history = []
    
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        f"👋 *Welcome to the GPU & System Monitoring Bot!*\n"
        f"Detected {gpu_count} GPU(s) on {subprocess.getoutput('hostname')}.\n"
        "Monitor your server with these options:\n"
        "- 📊 *GPU Metrics*: GPU usage, VRAM, temperature\n"
        "- 💻 *System Metrics*: CPU, RAM, disk, network\n"
        "- 🔄 *GPU Processes*: Processes using GPUs\n"
        "- ⚠️ *Watch Process*: Monitor specific processes\n"
        "- ▶️ *Start Monitoring*: Continuous updates\n"
        "- ❓ *Help*: Show this message\n\n"
        "Use the buttons below to get started!",
        reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        "❓ *Help - GPU & System Monitoring Bot*\n"
        "Available commands and buttons:\n"
        "- 📊 *GPU Metrics*: Fetch GPU utilization, VRAM, temperature, power draw\n"
        "- 💻 *System Metrics*: Fetch CPU, RAM, disk I/O, network I/O, uptime\n"
        "- 🔄 *GPU Processes*: List processes using GPUs\n"
        "- ⚠️ *Watch Process*: Monitor a process by PID or name\n"
        "- ▶️ *Start Monitoring*: Start continuous updates (every 60s)\n"
        "- ⏹️ *Stop Monitoring*: Stop continuous updates\n"
        "- ❓ *Help*: Show this message\n\n"
        "Click a button to proceed!",
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id

async def gpu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await get_gpu_metrics(gpu_index=0),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
    )
    last_message_ids[chat_id] = message.message_id

async def system(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await get_system_metrics(),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
    )
    last_message_ids[chat_id] = message.message_id

async def processes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await get_gpu_processes(),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
    )
    last_message_ids[chat_id] = message.message_id

async def watch_process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await list_processes_for_watching(),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
    )
    last_message_ids[chat_id] = message.message_id

async def watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    if not context.args:
        message = await update.message.reply_text(
            "⚠️ *Usage*: `/watch <pid>` or `/watch <process_name>`\n"
            "Use the Watch Process button to list available processes.",
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
        )
        last_message_ids[chat_id] = message.message_id
        return
    
    identifier = context.args[0]
    vram_threshold = 1000  # Default VRAM threshold (1GB)
    if len(context.args) > 1:
        try:
            vram_threshold = int(context.args[1])
        except ValueError:
            vram_threshold = 1000
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        processes = result.stdout.strip().split('\n')
        for line in processes:
            pid, proc_name, mem = line.split(', ')
            if pid == identifier or proc_name.lower() == identifier.lower():
                if chat_id not in PROCESS_ALERTS:
                    PROCESS_ALERTS[chat_id] = {}
                PROCESS_ALERTS[chat_id][pid] = {"name": proc_name, "vram_threshold": vram_threshold}
                message = await update.message.reply_text(
                    f"✅ *Monitoring Process*: {proc_name} (PID {pid}) for VRAM > {vram_threshold} MB.",
                    parse_mode='Markdown',
                    reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
                )
                last_message_ids[chat_id] = message.message_id
                return
        message = await update.message.reply_text(
            f"🚨 *Error*: Process with PID or name '{identifier}' not found.",
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
        )
    except Exception as e:
        message = await update.message.reply_text(
            f"🚨 *Error*: Failed to fetch processes.\n*Details*: {str(e)}",
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
        )
    last_message_ids[chat_id] = message.message_id

async def start_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    if chat_id in monitoring_jobs:
        message = await update.message.reply_text(
            "ℹ️ *Continuous monitoring* is already running. Use the Stop Monitoring button.",
            reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count),
            parse_mode='Markdown'
        )
        last_message_ids[chat_id] = message.message_id
        return
    
    interval = 60
    job = context.job_queue.run_repeating(
        monitor_callback,
        interval=interval,
        context=chat_id,
        name=str(chat_id)
    )
    monitoring_jobs[chat_id] = {"job": job, "message_id": None}
    
    message = await update.message.reply_text(
        f"✅ *Started continuous monitoring* (every {interval} seconds).\n"
        f"Monitoring GPU and system metrics...",
        reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id
    monitoring_jobs[chat_id]["message_id"] = message.message_id
    log_message(f"Started monitoring for chat {chat_id}")

async def monitor_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = context.job.context
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
                reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count)
            )
    except Exception as e:
        log_message(f"Error editing monitoring message: {e}")
        await delete_previous_message(context, chat_id)
        message = await context.bot.send_message(
            chat_id=chat_id,
            text=message_content,
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count)
        )
        monitoring_jobs[chat_id]["message_id"] = message.message_id
        last_message_ids[chat_id] = message.message_id

async def stop_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    if chat_id not in monitoring_jobs:
        message = await update.message.reply_text(
            "ℹ️ *No continuous monitoring* is running.",
            reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count),
            parse_mode='Markdown'
        )
        last_message_ids[chat_id] = message.message_id
        return
    
    monitoring_jobs[chat_id]["job"].schedule_removal()
    del monitoring_jobs[chat_id]
    
    message = await update.message.reply_text(
        "🛑 *Stopped continuous monitoring*.",
        reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count),
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
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
        )
    elif query.data == "gpu":
        message = await query.message.reply_text(
            await get_gpu_metrics(gpu_index=0),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
        )
    elif query.data == "system":
        message = await query.message.reply_text(
            await get_system_metrics(),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
        )
    elif query.data == "processes":
        message = await query.message.reply_text(
            await get_gpu_processes(),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
        )
    elif query.data == "watch_process":
        message = await query.message.reply_text(
            await list_processes_for_watching(),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
        )
    elif query.data == "help":
        message = await query.message.reply_text(
            "❓ *Help - GPU & System Monitoring Bot*\n"
            "Available commands and buttons:\n"
            "- 📊 *GPU Metrics*: Fetch GPU utilization, VRAM, temperature, power draw\n"
            "- 💻 *System Metrics*: Fetch CPU, RAM, disk I/O, network I/O, uptime\n"
            "- 🔄 *GPU Processes*: List processes using GPUs\n"
            "- ⚠️ *Watch Process*: Monitor a process by PID or name\n"
            "- ▶️ *Start Monitoring*: Start continuous updates (every 60s)\n"
            "- ⏹️ *Stop Monitoring*: Stop continuous updates\n"
            "- ❓ *Help*: Show this message\n\n"
            "Click a button to proceed!",
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count)
        )
    elif query.data == "start_monitoring":
        if chat_id in monitoring_jobs:
            message = await query.message.reply_text(
                "ℹ️ *Continuous monitoring* is already running. Use the Stop Monitoring button.",
                reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count),
                parse_mode='Markdown'
            )
        else:
            interval = 60
            job = context.job_queue.run_repeating(
                monitor_callback,
                interval=interval,
                context=chat_id,
                name=str(chat_id)
            )
            monitoring_jobs[chat_id] = {"job": job, "message_id": None}
            message = await query.message.reply_text(
                f"✅ *Started continuous monitoring* (every {interval} seconds).\n"
                f"Monitoring GPU and system metrics...",
                reply_markup=get_keyboard(is_monitoring=True, gpu_count=gpu_count),
                parse_mode='Markdown'
            )
            monitoring_jobs[chat_id]["message_id"] = message.message_id
            log_message(f"Started monitoring for chat {chat_id}")
    elif query.data == "stop_monitoring":
        if chat_id not in monitoring_jobs:
            message = await query.message.reply_text(
                "ℹ️ *No continuous monitoring* is running.",
                reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count),
                parse_mode='Markdown'
            )
        else:
            monitoring_jobs[chat_id]["job"].schedule_removal()
            del monitoring_jobs[chat_id]
            message = await query.message.reply_text(
                "🛑 *Stopped continuous monitoring*.",
                reply_markup=get_keyboard(is_monitoring=False, gpu_count=gpu_count),
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
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs, gpu_count=gpu_count),
            parse_mode='Markdown'
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
    app.add_handler(CommandHandler("start_monitoring", start_monitoring))
    app.add_handler(CommandHandler("stop_monitoring", stop_monitoring))
    app.add_handler(CallbackQueryHandler(button_callback))
    
    app.run_polling()

if __name__ == '__main__':
    main()
