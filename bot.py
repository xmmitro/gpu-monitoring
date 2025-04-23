#!/usr/bin/env python3

import logging
import subprocess
import psutil
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
}

# Global state for continuous monitoring
monitoring_jobs = {}  # Dictionary to store monitoring jobs per chat
last_message_ids = {}  # Track last message ID per chat for deletion

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

# Inline keyboard based on monitoring state
def get_keyboard(is_monitoring: bool = False) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("📊 GPU Metrics", callback_data="gpu"),
            InlineKeyboardButton("💻 System Metrics", callback_data="system")
        ],
        [
            InlineKeyboardButton("🔄 GPU Processes", callback_data="processes"),
            InlineKeyboardButton("❓ Help", callback_data="help")
        ]
    ]
    # Add Start/Stop Monitoring button based on state
    if is_monitoring:
        keyboard.append([InlineKeyboardButton("⏹️ Stop Monitoring", callback_data="stop_monitoring")])
    else:
        keyboard.append([InlineKeyboardButton("▶️ Start Monitoring", callback_data="start_monitoring")])
    return InlineKeyboardMarkup(keyboard)

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

# Function to get NVIDIA GPU metrics
async def get_gpu_metrics() -> str:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        gpu_name, gpu_util, mem_util, vram_total, vram_used, vram_free, temp, power = output.split(', ')
        
        gpu_name = gpu_name.strip()
        gpu_util = int(gpu_util.replace('%', ''))
        mem_util = int(mem_util.replace('%', ''))
        vram_total = int(vram_total)
        vram_used = int(vram_used)
        vram_free = int(vram_free)
        temp = int(temp)
        power = float(power)
        
        vram_usage = (vram_used / vram_total) * 100
        
        message = f"📊 *GPU Monitoring Report* - {subprocess.getoutput('hostname')}\n"
        message += f"*GPU*: {gpu_name}\n"
        message += f"*GPU Utilization*: {gpu_util}% {'⚠️ High' if gpu_util > THRESHOLDS['gpu_util'] else ''}\n"
        message += f"*Memory Utilization*: {mem_util}% {'⚠️ High' if mem_util > THRESHOLDS['mem_util'] else ''}\n"
        message += f"*VRAM Usage*: {vram_used} MB / {vram_total} MB ({vram_usage:.2f}%) {'⚠️ High' if vram_usage > THRESHOLDS['vram_usage'] else ''}\n"
        message += f"*Temperature*: {temp}°C\n"
        message += f"*Power Draw*: {power} W"
        
        log_message(f"GPU Metrics: {gpu_name}, Util: {gpu_util}%, Mem: {mem_util}%, VRAM: {vram_used}/{vram_total} MB, Temp: {temp}°C, Power: {power} W")
        return message
    except FileNotFoundError:
        return "🚨 *Error*: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    except subprocess.CalledProcessError as e:
        log_message(f"Error fetching GPU metrics: {e}")
        return f"🚨 *Error*: Failed to fetch GPU metrics.\n*Details*: {e.stderr}"

# Function to get system metrics (CPU, RAM, Disk, Network)
async def get_system_metrics() -> str:
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        
        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024 ** 3)
        ram_used = ram.used / (1024 ** 3)
        ram_usage = ram.percent
        
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes / (1024 ** 3)  # Convert to GB
        disk_write = disk_io.write_bytes / (1024 ** 3)  # Convert to GB
        
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent / (1024 ** 2)
        net_recv = net_io.bytes_recv / (1024 ** 2)
        
        message = f"💻 *System Monitoring Report* - {subprocess.getoutput('hostname')}\n"
        message += f"*CPU Usage*: {cpu_usage}% {'⚠️ High' if cpu_usage > THRESHOLDS['cpu_usage'] else ''}\n"
        message += f"*CPU Per Core*: {', '.join([f'{i}: {p}%' for i, p in enumerate(cpu_per_core)])}\n"
        message += f"*RAM Usage*: {ram_used:.2f} GB / {ram_total:.2f} GB ({ram_usage:.2f}%) {'⚠️ High' if ram_usage > THRESHOLDS['ram_usage'] else ''}\n"
        message += f"*Disk I/O*: Read: {disk_read:.2f} GB, Write: {disk_write:.2f} GB\n"
        message += f"*Network I/O*: Sent: {net_sent:.2f} MB, Received: {net_recv:.2f} MB"
        
        log_message(f"System Metrics: CPU: {cpu_usage}%, RAM: {ram_usage}%, Disk R/W: {disk_read}/{disk_write} GB, Net S/R: {net_sent}/{net_recv} MB")
        return message
    except Exception as e:
        log_message(f"Error fetching system metrics: {e}")
        return f"🚨 *Error*: Failed to fetch system metrics.\n*Details*: {str(e)}"

# Function to get GPU-related processes
async def get_gpu_processes() -> str:
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
        
        message = f"🔄 *GPU Processes* - {subprocess.getoutput('hostname')}\n"
        for line in processes.split('\n'):
            pid, proc_name, mem = line.split(', ')
            message += f"*PID*: {pid}, *Name*: {proc_name}, *VRAM*: {mem}\n"
        
        log_message(f"GPU Processes: {processes}")
        return message
    except FileNotFoundError:
        return "🚨 *Error*: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    except subprocess.CalledProcessError as e:
        log_message(f"Error fetching GPU processes: {e}")
        return f"🚨 *Error*: Failed to fetch GPU processes.\n*Details*: {e.stderr}"

# Telegram command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        "👋 *Welcome to the GPU & System Monitoring Bot!*\n"
        "Monitor your Ubuntu server with these options:\n"
        "- 📊 *GPU Metrics*: GPU usage, VRAM, temperature\n"
        "- 💻 *System Metrics*: CPU, RAM, disk, network\n"
        "- 🔄 *GPU Processes*: Processes using the GPU\n"
        "- ▶️ *Start Monitoring*: Continuous updates\n"
        "- ❓ *Help*: Show this message\n\n"
        "Use the buttons below to get started!",
        reply_markup=get_keyboard(is_monitoring=False),
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
        "- 💻 *System Metrics*: Fetch CPU, RAM, disk I/O, network I/O\n"
        "- 🔄 *GPU Processes*: List processes using the GPU\n"
        "- ▶️ *Start Monitoring*: Start continuous updates (every 60s)\n"
        "- ⏹️ *Stop Monitoring*: Stop continuous updates\n"
        "- ❓ *Help*: Show this message\n\n"
        "Click a button to proceed!",
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id

async def gpu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await get_gpu_metrics(),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs)
    )
    last_message_ids[chat_id] = message.message_id

async def system(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await get_system_metrics(),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs)
    )
    last_message_ids[chat_id] = message.message_id

async def processes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    message = await update.message.reply_text(
        await get_gpu_processes(),
        parse_mode='Markdown',
        reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs)
    )
    last_message_ids[chat_id] = message.message_id

async def start_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    if chat_id in monitoring_jobs:
        message = await update.message.reply_text(
            "ℹ️ *Continuous monitoring* is already running. Use the Stop Monitoring button.",
            reply_markup=get_keyboard(is_monitoring=True),
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
        reply_markup=get_keyboard(is_monitoring=True),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id
    monitoring_jobs[chat_id]["message_id"] = message.message_id
    log_message(f"Started monitoring for chat {chat_id}")

async def monitor_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = context.job.context
    if chat_id not in monitoring_jobs:
        return
    
    message_content = f"{await get_gpu_metrics()}\n\n{await get_system_metrics()}"
    
    try:
        # Edit the existing monitoring message
        if monitoring_jobs[chat_id]["message_id"]:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=monitoring_jobs[chat_id]["message_id"],
                text=message_content,
                parse_mode='Markdown',
                reply_markup=get_keyboard(is_monitoring=True)
            )
    except Exception as e:
        log_message(f"Error editing monitoring message: {e}")
        # If editing fails, send a new message
        await delete_previous_message(context, chat_id)
        message = await context.bot.send_message(
            chat_id=chat_id,
            text=message_content,
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=True)
        )
        monitoring_jobs[chat_id]["message_id"] = message.message_id
        last_message_ids[chat_id] = message.message_id

async def stop_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    await delete_previous_message(context, chat_id)
    
    if chat_id not in monitoring_jobs:
        message = await update.message.reply_text(
            "ℹ️ *No continuous monitoring* is running.",
            reply_markup=get_keyboard(is_monitoring=False),
            parse_mode='Markdown'
        )
        last_message_ids[chat_id] = message.message_id
        return
    
    monitoring_jobs[chat_id]["job"].schedule_removal()
    del monitoring_jobs[chat_id]
    
    message = await update.message.reply_text(
        "🛑 *Stopped continuous monitoring*.",
        reply_markup=get_keyboard(is_monitoring=False),
        parse_mode='Markdown'
    )
    last_message_ids[chat_id] = message.message_id
    log_message(f"Stopped monitoring for chat {chat_id}")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    # Delete the previous message
    await delete_previous_message(context, chat_id)
    
    if query.data == "gpu":
        message = await query.message.reply_text(
            await get_gpu_metrics(),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs)
        )
    elif query.data == "system":
        message = await query.message.reply_text(
            await get_system_metrics(),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs)
        )
    elif query.data == "processes":
        message = await query.message.reply_text(
            await get_gpu_processes(),
            parse_mode='Markdown',
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs)
        )
    elif query.data == "help":
        message = await query.message.reply_text(
            "❓ *Help - GPU & System Monitoring Bot*\n"
            "Available commands and buttons:\n"
            "- 📊 *GPU Metrics*: Fetch GPU utilization, VRAM, temperature, power draw\n"
            "- 💻 *System Metrics*: Fetch CPU, RAM, disk I/O, network I/O\n"
            "- 🔄 *GPU Processes*: List processes using the GPU\n"
            "- ▶️ *Start Monitoring*: Start continuous updates (every 60s)\n"
            "- ⏹️ *Stop Monitoring*: Stop continuous updates\n"
            "- ❓ *Help*: Show this message\n\n"
            "Click a button to proceed!",
            reply_markup=get_keyboard(is_monitoring=chat_id in monitoring_jobs),
            parse_mode='Markdown'
        )
    elif query.data == "start_monitoring":
        if chat_id in monitoring_jobs:
            message = await query.message.reply_text(
                "ℹ️ *Continuous monitoring* is already running. Use the Stop Monitoring button.",
                reply_markup=get_keyboard(is_monitoring=True),
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
                reply_markup=get_keyboard(is_monitoring=True),
                parse_mode='Markdown'
            )
            monitoring_jobs[chat_id]["message_id"] = message.message_id
            log_message(f"Started monitoring for chat {chat_id}")
    elif query.data == "stop_monitoring":
        if chat_id not in monitoring_jobs:
            message = await query.message.reply_text(
                "ℹ️ *No continuous monitoring* is running.",
                reply_markup=get_keyboard(is_monitoring=False),
                parse_mode='Markdown'
            )
        else:
            monitoring_jobs[chat_id]["job"].schedule_removal()
            del monitoring_jobs[chat_id]
            message = await query.message.reply_text(
                "🛑 *Stopped continuous monitoring*.",
                reply_markup=get_keyboard(is_monitoring=False),
                parse_mode='Markdown'
            )
            log_message(f"Stopped monitoring for chat {chat_id}")
    
    last_message_ids[chat_id] = message.message_id

def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("gpu", gpu))
    app.add_handler(CommandHandler("system", system))
    app.add_handler(CommandHandler("processes", processes))
    app.add_handler(CommandHandler("start_monitoring", start_monitoring))
    app.add_handler(CommandHandler("stop_monitoring", stop_monitoring))
    app.add_handler(CallbackQueryHandler(button_callback))
    
    app.run_polling()

if __name__ == '__main__':
    main()
