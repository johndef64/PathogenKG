
#!/usr/bin/env python3
"""
Script to find the log file with the best configuration based on validation metrics.
Reads all files in the logs/ directory and identifies the one with the highest best_metric.
"""

import os
import re
import glob
from typing import Dict, Tuple, Optional

def extract_best_metric_from_log(file_path: str) -> Optional[float]:
    """
    Extract the best validation metric from a log file.
    
    Args:
        file_path: Path to the log file
        
    Returns:
        The highest best_metric value found in the file, or None if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for best_metric values in the log
        # Pattern matches: best_metric=0.598, best_metric=0.543, etc.
        pattern = r'best_metric=([0-9]*\.?[0-9]+)'
        matches = re.findall(pattern, content)
        
        if matches:
            # Convert to float and return the maximum value
            best_metrics = [float(match) for match in matches]
            return max(best_metrics)
        else:
            return None
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_test_metrics_from_log(file_path: str) -> Optional[Dict[str, float]]:
    """
    Extract test metrics from the final test results line.
    
    Args:
        file_path: Path to the log file
        
    Returns:
        Dictionary with test metrics or None if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for the test results line
        # Pattern: "Test Auroc: 0.705, Test Auprc: 0.659, Test MRR: 0.562"
        test_pattern = r'Test Auroc: ([0-9]*\.?[0-9]+), Test Auprc: ([0-9]*\.?[0-9]+), Test MRR: ([0-9]*\.?[0-9]+)'
        match = re.search(test_pattern, content)
        
        if match:
            return {
                'test_auroc': float(match.group(1)),
                'test_auprc': float(match.group(2)),
                'test_mrr': float(match.group(3))
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def find_best_configuration(logs_dir: str = 'logs') -> None:
    """
    Find and print the log file with the best configuration.
    
    Args:
        logs_dir: Directory containing log files
    """
    if not os.path.exists(logs_dir):
        print(f"Error: Directory '{logs_dir}' does not exist.")
        return
    
    # Get all files in the logs directory
    log_files = glob.glob(os.path.join(logs_dir, '*'))
    
    # Filter out directories, keep only files
    log_files = [f for f in log_files if os.path.isfile(f)]
    
    if not log_files:
        print(f"No files found in '{logs_dir}' directory.")
        return
    
    print(f"Found {len(log_files)} files in '{logs_dir}' directory.")
    print("Analyzing log files...\n")
    
    best_file = None
    best_metric = -1.0
    results = []
    
    for file_path in log_files:
        filename = os.path.basename(file_path)
        
        # Extract validation metric
        val_metric = extract_best_metric_from_log(file_path)
        
        # Extract test metrics
        test_metrics = extract_test_metrics_from_log(file_path)
        
        if val_metric is not None:
            results.append({
                'filename': filename,
                'val_metric': val_metric,
                'test_metrics': test_metrics
            })
            
            if val_metric > best_metric:
                best_metric = val_metric
                best_file = filename
        else:
            print(f"Warning: Could not extract metrics from {filename}")
    
    if not results:
        print("No valid metrics found in any log files.")
        return
    
    # Sort results by validation metric (descending)
    results.sort(key=lambda x: x['val_metric'], reverse=True)
    
    # Print summary table
    print("=" * 80)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Filename':<30} {'Val Metric':<12} {'Test AUROC':<12} {'Test AUPRC':<12} {'Test MRR':<12}")
    print("-" * 80)
    
    for result in results:
        filename = result['filename']
        val_metric = result['val_metric']
        test_metrics = result['test_metrics']
        
        if test_metrics:
            test_auroc = f"{test_metrics['test_auroc']:.3f}"
            test_auprc = f"{test_metrics['test_auprc']:.3f}"
            test_mrr = f"{test_metrics['test_mrr']:.3f}"
        else:
            test_auroc = test_auprc = test_mrr = "N/A"
        
        # Highlight the best configuration
        marker = ">>> " if filename == best_file else "    "
        print(f"{marker}{filename:<26} {val_metric:<12.3f} {test_auroc:<12} {test_auprc:<12} {test_mrr:<12}")
    
    print("=" * 80)
    print(f"\nBEST CONFIGURATION: {best_file}")
    print(f"Best validation metric: {best_metric:.3f}")
    
    # Print test metrics for the best configuration
    best_result = next(r for r in results if r['filename'] == best_file)
    if best_result['test_metrics']:
        test_metrics = best_result['test_metrics']
        print(f"Test AUROC: {test_metrics['test_auroc']:.3f}")
        print(f"Test AUPRC: {test_metrics['test_auprc']:.3f}")
        print(f"Test MRR: {test_metrics['test_mrr']:.3f}")

if __name__ == "__main__":
    logs_directory = 'logs/'
    
    find_best_configuration(logs_directory)

# def extract_info_from_line(line):
#   """Extract structured info from TSV line."""
#   head, relation, tail, source, type_ = line.strip().split('\t')
  
#   def parse_entity(entity):
#     parts = entity.split('::')
#     entity_type = parts[0]
#     src_id = parts[1].split(':', 1)
#     return entity_type, src_id[0], src_id[1]
  
#   head_type, head_src, head_id = parse_entity(head)
#   tail_type, tail_src, tail_id = parse_entity(tail)
  
#   return {
#     'head_type': head_type, 'head_src': head_src, 'head_id': head_id,
#     'tail_type': tail_type, 'tail_src': tail_src, 'tail_id': tail_id,
#     'relation': relation, 'source': source, 'type': type_
#   }

# count_d = {}
# with open('dataset/pathogenkg/PathogenKG_human_plus_83332.tsv', 'r') as fin:
#   fin.readline()
#   for line in fin:
#     info = extract_info_from_line(line)
#     t = info['type']
#     if t not in count_d:
#       count_d[t] = 1
#     else:
#       count_d[t] += 1
      
# print(count_d)
       
# removed = 0
# with open('dataset/vitagraph.tsv', 'r') as fin:
#   with open('dataset/vitagraph_tmp.tsv', 'w') as fout:
#     header = fin.readline()
#     fout.write(header)
#     for line in fin:
#       info = extract_info_from_line(line)
#       if 'sars' in info['head_id'].lower() or 'sars' in info['tail_id'].lower() or \
#         'anatomy' in info['head_type'].lower() or 'anatomy' in info['tail_type'].lower():
#         removed += 1
#         continue
#       fout.write(line)

# print(f"Removed {removed} lines.")