import os
import subprocess
import time

AVAILABLE_TARGETS = [
    "83332", "224308", "208964", "99287", "71421", "243230",
    "85962", "171101", "243277", "294", "1314", "272631",
    "212717", "36329", "237561", "6183", "5664", "185431", "330879"
]

SCRIPT = "build_pathogenkg.py"
LOG_DIR = "logs"

def main():
    # Create log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Check if build_pathogenkg.py exists
    if not os.path.exists(SCRIPT):
        print(f"❌ Error: {SCRIPT} not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        print("Make sure build_pathogenkg.py is in the same folder as this script.")
        return
    
    print(f"Starting PathogenKG generation for {len(AVAILABLE_TARGETS)} targets...")
    print(f"Logs will be saved in {LOG_DIR}/")
    print()
    
    start_time = time.time()
    failed_targets = []
    
    for i, target in enumerate(AVAILABLE_TARGETS, start=1):
        log_file = os.path.join(LOG_DIR, f"pathogenkg_{target}.log")
        print(f"[{i}/{len(AVAILABLE_TARGETS)}] Processing target: {target}")
        
        try:
            with open(log_file, "w") as log:
                process = subprocess.run(
                    ["python", SCRIPT, "--target", target], 
                    stdout=log, 
                    stderr=subprocess.STDOUT,
                    timeout=300  # 5 minute timeout per target
                )
                
                if process.returncode == 0:
                    print("  ✅ Completed successfully")
                else:
                    print(f"  ❌ Failed (see {log_file})")
                    failed_targets.append((target, process.returncode))
                    
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout after 5 minutes (see {log_file})")
            failed_targets.append((target, "timeout"))
        except Exception as e:
            print(f"  💥 Exception: {str(e)}")
            failed_targets.append((target, str(e)))
    
    end_time = time.time()
    duration = end_time - start_time
    
    print()
    print("=" * 50)
    
    if failed_targets:
        print(f"❌ {len(failed_targets)} targets failed:")
        for target, error in failed_targets:
            print(f"  - Target {target}: {error}")
        print()
        print(f"✅ {len(AVAILABLE_TARGETS) - len(failed_targets)} targets completed successfully")
    else:
        print("🎉 All targets completed successfully!")
    
    print(f"⏱️  Total time: {int(duration)}s ({duration/60:.1f} minutes)")
    print(f"📁 Logs available in: {LOG_DIR}/")

if __name__ == "__main__":
    main()
