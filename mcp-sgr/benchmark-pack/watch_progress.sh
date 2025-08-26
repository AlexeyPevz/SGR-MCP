#!/bin/bash
# Watch benchmark progress in real-time

echo "🔍 Watching benchmark progress..."
echo "Press Ctrl+C to stop"
echo ""

# Find the latest log file
LATEST_LOG=$(ls -t reports/benchmark_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No benchmark log found. Waiting..."
    sleep 5
    LATEST_LOG=$(ls -t reports/benchmark_*.log 2>/dev/null | head -1)
fi

if [ -n "$LATEST_LOG" ]; then
    echo "📄 Monitoring: $LATEST_LOG"
    echo ""
    
    # Watch the log file
    tail -f "$LATEST_LOG" | while read line; do
        # Color code the output
        if [[ $line == *"✓ Success!"* ]]; then
            echo -e "\033[32m$line\033[0m"  # Green for success
        elif [[ $line == *"✗ Failed:"* ]]; then
            echo -e "\033[31m$line\033[0m"  # Red for failure
        elif [[ $line == *"Running"* ]]; then
            echo -e "\033[33m$line\033[0m"  # Yellow for running
        elif [[ $line == *"Reports saved"* ]]; then
            echo -e "\033[36m$line\033[0m"  # Cyan for completion
        else
            echo "$line"
        fi
    done
else
    echo "❌ No benchmark log file found!"
fi