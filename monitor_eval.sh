#!/usr/bin/env bash
# Monitor script for llm_eval.py process

echo "==================================================================="
echo "LLM Evaluation Monitor"
echo "==================================================================="
echo ""

# Find the llm_eval process
PID=$(ps aux | grep "python llm_eval.py" | grep -v grep | awk '{print $2}' | head -1)

if [[ -z "$PID" ]]; then
    echo "❌ No llm_eval.py process found"
    exit 0
fi

echo "📊 Process ID: $PID"
echo ""

# Get process info
ELAPSED=$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ')
CPU=$(ps -p $PID -o %cpu= 2>/dev/null | tr -d ' ')
MEM=$(ps -p $PID -o %mem= 2>/dev/null | tr -d ' ')
STATE=$(ps -p $PID -o state= 2>/dev/null | tr -d ' ')

echo "⏱️  Running for: $ELAPSED"
echo "💻 CPU usage: ${CPU}%"
echo "🧠 Memory usage: ${MEM}%"
echo "📍 State: $STATE"
echo ""

# GPU info
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while read line; do
    IFS=',' read -r idx util mem_used mem_total <<< "$line"
    if nvidia-smi | grep -q "$PID"; then
        echo "   GPU $idx: ${util}% utilization, ${mem_used}MB / ${mem_total}MB (⚡ ACTIVE)"
    else
        echo "   GPU $idx: ${util}% utilization, ${mem_used}MB / ${mem_total}MB"
    fi
done
echo ""

# Check for compile workers (indicates model compilation/warmup)
WORKERS=$(ps aux | grep "compile_worker" | grep "parent=$PID" | wc -l)
if [[ $WORKERS -gt 0 ]]; then
    echo "🔧 Status: Model compilation in progress ($WORKERS workers)"
    echo "   This is normal - PyTorch is compiling the model for optimization"
    echo "   GPU utilization will increase once compilation completes"
else
    echo "🚀 Status: Running evaluation"
fi
echo ""

# Check recent file writes (output)
if [[ -d "llm_eval" ]]; then
    LATEST=$(find llm_eval -name "*.json" -mmin -5 2>/dev/null | wc -l)
    if [[ $LATEST -gt 0 ]]; then
        echo "📝 Recent output detected (files modified in last 5 min)"
    fi
fi

# Thread info
echo "🧵 Thread count: $(ps -p $PID -T | wc -l)"

# Show main thread state
MAIN_STATE=$(ps -p $PID -L -o pid,tid,state,wchan:25 | grep "^[[:space:]]*$PID[[:space:]]*$PID" | awk '{print $3, $4}')
echo "🎯 Main thread: $MAIN_STATE"

echo ""
echo "==================================================================="
echo "💡 Tips:"
echo "   - GPU at 0% usually means: model compilation, dataset loading,"
echo "     or CPU-bound preprocessing"
echo "   - Run this script again in a few minutes to check progress"
echo "   - For continuous monitoring: watch -n 30 ./monitor_eval.sh"
echo "==================================================================="
