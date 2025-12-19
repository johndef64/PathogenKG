#!/bin/bash

# PathogenKG generator script
# Runs build_pathogenkg.py for all available targets

set -e  # Exit on error

AVAILABLE_TARGETS=(
    "83332" "224308" "208964" "99287" "71421" "243230"
    "85962" "171101" "243277" "294" "1314" "272631"
    "212717" "36329" "237561" "6183" "5664" "185431" "330879"
)

SCRIPT="build_pathogenkg.py"
LOG_DIR="logs"
TOTAL_TARGETS=${#AVAILABLE_TARGETS[@]}

# Create log directory
mkdir -p "$LOG_DIR"

echo "Starting PathogenKG generation for $TOTAL_TARGETS targets..."
echo "Logs will be saved in $LOG_DIR/"

start_time=$(date +%s)

for i in "${!AVAILABLE_TARGETS[@]}"; do
    target="${AVAILABLE_TARGETS[$i]}"
    log_file="$LOG_DIR/pathogenkg_${target}.log"
    
    echo "[$((i+1))/$TOTAL_TARGETS] Processing target: $target"
    
    if python "$SCRIPT" --target "$target" > "$log_file" 2>&1; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed (see $log_file)"
        exit_code=$?
        echo "Error processing target $target (exit code: $exit_code)"
        exit $exit_code
    fi
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "All targets completed successfully!"
echo "Total time: ${duration}s"
echo "Logs available in: $LOG_DIR/"