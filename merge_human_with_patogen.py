import os
import csv
import logging
import argparse
from time import time

from src.bio_utils import get_total_time

available_targets = [
  '83332', '224308', '208964', '99287', '71421', '243230', 
  '85962', '171101', '243277', '294', '1314', '272631',
  '212717', '36329', '237561', '6183', '5664', '185431', '330879'
]

file_path = 'dataset/pathogenkg/'
human_vitagraph_path = os.path.join(file_path, 'VitaGraph_human_extended.tsv')

def extract_info_from_line(line):
  """Extract structured info from TSV line."""
  head, relation, tail, source, type_ = line[:5]
  
  def parse_entity(entity):
    parts = entity.split('::')
    entity_type = parts[0]
    src_id = parts[1].split(':', 1)  # Split only on first ':'
    return entity_type, src_id[0], src_id[1]
  
  head_type, head_src, head_id = parse_entity(head)
  tail_type, tail_src, tail_id = parse_entity(tail)
  
  return {
    'head_type': head_type, 'head_src': head_src, 'head_id': head_id,
    'tail_type': tail_type, 'tail_src': tail_src, 'tail_id': tail_id,
    'relation': relation, 'source': source, 'type': type_
  }

def load_orthology_groups(target_path):
  """Load orthology groups from target file."""
  orthology_groups = set()
  
  with open(target_path, 'r') as fin:
    reader = csv.reader(fin, delimiter='\t')
    next(reader)  # Skip header
    
    for row in reader:
      info = extract_info_from_line(row)
      if info['head_type'] == 'OrthologyGroup':
        orthology_groups.add(info['head_id'])
      if info['tail_type'] == 'OrthologyGroup':
        orthology_groups.add(info['tail_id'])
  
  return orthology_groups

def should_include_line(info, orthology_groups):
  """Check if line should be included based on orthology groups."""
  head_ortho = info['head_type'] == 'OrthologyGroup'
  tail_ortho = info['tail_type'] == 'OrthologyGroup'
  
  if head_ortho and info['head_id'] not in orthology_groups:
    return False
  if tail_ortho and info['tail_id'] not in orthology_groups:
    return False
  return True

def merge_vitagraph_with_pathogenkg(target_path, out_path):
  logging_header = '[merge_vitagraph_with_pathogenkg]'
  
  # Load orthology groups from target
  logging.info(f'{logging_header} Extracting orthology groups from target')
  orthology_groups = load_orthology_groups(target_path)
  logging.info(f'{logging_header} Found {len(orthology_groups)} orthology groups')
  
  human_lines_saved = 0
  target_lines_saved = 0
  
  with open(out_path, 'w') as fout: 
    writer = csv.writer(fout, delimiter='\t')
    writer.writerow(['head', 'interaction', 'tail', 'source', 'type'])
    
    # Process human vitagraph
    logging.info(f'{logging_header} Merging knowledge graphs...')

    with open(human_vitagraph_path, 'r') as fin:
      reader = csv.reader(fin, delimiter='\t')
      next(reader) 
      
      for row in reader:
        info = extract_info_from_line(row)
        if should_include_line(info, orthology_groups):
          writer.writerow(row)
          human_lines_saved += 1
    
    # Add target-specific data
    with open(target_path, 'r') as fin:
      reader = csv.reader(fin, delimiter='\t')
      next(reader)  
      
      for row in reader:
        writer.writerow(row)
        target_lines_saved += 1
  logging.info(f'{logging_header} Done')

  return human_lines_saved, target_lines_saved

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  
  parser = argparse.ArgumentParser(description='Merge human VitaGraph with target-specific data')
  parser.add_argument('--target', default='83332', choices=available_targets,
                      help='Target taxonomy ID')
  args = parser.parse_args()
  
  target = args.target
  in_pathogenkg_path = os.path.join(file_path, f'PathogenKG_{target}.tsv')
  out_path = os.path.join(file_path, f'PathogenKG_human_plus_{target}.tsv')
  
  if not os.path.exists(in_pathogenkg_path):
    logging.error(f'Input file not found: {in_pathogenkg_path}')
    exit(1)
  
  logging.info(f'Merging human data with target {target}')
  logging.info(f'Output: {out_path}')
  
  start = time()
  human_lines_saved, target_lines_saved = merge_vitagraph_with_pathogenkg(in_pathogenkg_path, out_path)
  total_lines = human_lines_saved + target_lines_saved
  elapsed = get_total_time(start, time())
  
  logging.info(f'Completed in {elapsed}s - Human lines: {human_lines_saved} | Target lines: {target_lines_saved} | Total lines: {total_lines}')