#!/bin/bash

# Find the latest benchmark log
LOG_FILE=$(ls -t reports/full_benchmark_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    LOG_FILE=$(ls -t reports/benchmark_*.log 2>/dev/null | head -1)
fi

if [ -z "$LOG_FILE" ]; then
    echo "❌ No benchmark log found!"
    exit 1
fi

echo "📊 Monitoring: $LOG_FILE"
echo "="*60

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Monitor in real-time
tail -f "$LOG_FILE" | while read line; do
    # Color code different types of output
    if [[ $line == *"✓"* ]] || [[ $line == *"success"* ]]; then
        echo -e "${GREEN}$line${NC}"
    elif [[ $line == *"✗"* ]] || [[ $line == *"error"* ]] || [[ $line == *"Failed"* ]]; then
        echo -e "${RED}$line${NC}"
    elif [[ $line == *"GPT-4"* ]] || [[ $line == *"Claude-3.5"* ]]; then
        echo -e "${PURPLE}⭐ TOP MODEL: $line${NC}"
    elif [[ $line == *"Mistral-7B-Free"* ]] || [[ $line == *"Ministral"* ]]; then
        echo -e "${YELLOW}🆓 BUDGET MODEL: $line${NC}"
    elif [[ $line == *"Cost:"* ]]; then
        echo -e "${BLUE}💰 $line${NC}"
    elif [[ $line == *"Result:"* ]]; then
        echo -e "${GREEN}$line${NC}"
    else
        echo "$line"
    fi
    
    # Show running total periodically
    if [[ $line == *"Result:"* ]]; then
        TOTAL=$(grep -c "Result:" "$LOG_FILE")
        SUCCESS=$(grep -c "✓" "$LOG_FILE")
        COST=$(grep "Cost:" "$LOG_FILE" | awk -F'$' '{sum+=$2} END {printf "%.4f", sum}')
        echo -e "${BLUE}📈 Progress: $TOTAL tests, $SUCCESS successful, Total cost: \$$COST${NC}"
    fi
done