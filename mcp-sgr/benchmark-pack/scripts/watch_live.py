#!/usr/bin/env python3
"""Live monitoring of benchmark progress with pretty output"""

import time
import os
from pathlib import Path
import re

def get_latest_log():
    """Find the latest benchmark log"""
    logs = list(Path("reports").glob("*.log"))
    if not logs:
        return None
    return max(logs, key=lambda x: x.stat().st_mtime)

def parse_log_line(line):
    """Parse and format log line"""
    # Model detection
    if "GPT-4o" in line or "Claude-3.5" in line:
        return f"🏆 TOP MODEL: {line}", "purple"
    elif "Mistral-7B-Free" in line or "Llama-3.2-3B-Free" in line:
        return f"🆓 FREE MODEL: {line}", "yellow"
    elif "Running" in line:
        return f"▶️  {line}", "blue"
    elif "✓" in line or "success" in line.lower():
        return f"✅ {line}", "green"
    elif "✗" in line or "error" in line.lower() or "failed" in line.lower():
        return f"❌ {line}", "red"
    elif "Cost:" in line:
        return f"💰 {line}", "cyan"
    elif "Result:" in line:
        return f"📊 {line}", "green"
    else:
        return line, None

def print_colored(text, color=None):
    """Print with color"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    
    if color and color in colors:
        print(f"{colors[color]}{text}{colors['reset']}")
    else:
        print(text)

def show_stats(log_file):
    """Show current statistics"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    total_tasks = content.count("Running")
    completed = content.count("Result:")
    successful = content.count("✓")
    failed = content.count("✗")
    
    # Calculate cost
    costs = re.findall(r'Cost: \$(\d+\.\d+)', content)
    total_cost = sum(float(c) for c in costs)
    
    print("\n" + "="*60)
    print_colored(f"📈 CURRENT STATISTICS", "cyan")
    print_colored(f"Total Tasks Started: {total_tasks}", "white")
    print_colored(f"Completed: {completed} ({completed/max(total_tasks,1)*100:.1f}%)", "green" if completed > 0 else "white")
    print_colored(f"Successful: {successful}", "green")
    print_colored(f"Failed: {failed}", "red" if failed > 0 else "white")
    print_colored(f"💵 Total Cost So Far: ${total_cost:.4f}", "yellow")
    print("="*60 + "\n")

def main():
    """Main monitoring loop"""
    log_file = get_latest_log()
    if not log_file:
        print("❌ No benchmark log found!")
        return
    
    print_colored(f"📊 Monitoring: {log_file}", "cyan")
    print_colored("Press Ctrl+C to stop\n", "white")
    
    # Track last position
    last_position = 0
    last_stats_time = 0
    
    try:
        while True:
            # Check file size
            current_size = log_file.stat().st_size
            
            if current_size > last_position:
                with open(log_file, 'r') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()
                
                # Process new lines
                for line in new_lines:
                    line = line.strip()
                    if line:
                        formatted, color = parse_log_line(line)
                        print_colored(formatted, color)
            
            # Show stats every 30 seconds
            current_time = time.time()
            if current_time - last_stats_time > 30:
                show_stats(log_file)
                last_stats_time = current_time
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
        show_stats(log_file)

if __name__ == "__main__":
    main()