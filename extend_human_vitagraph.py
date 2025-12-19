import os
import gzip
import logging
import argparse
from time import time

from src.bio_utils import get_total_time

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)

available_targets = [
  '83332', '224308', '208964', '99287', '71421', '243230', 
  '85962', '171101', '243277', '294', '1314', '272631',
  '212717', '36329', '237561', '6183', '5664', '185431', '330879'
]

out_path = 'dataset/pathogenkg/'

import zipfile
if not os.path.exists('dataset/vitagraph.tsv') and os.path.exists('dataset/vitagraph.zip'):
  with zipfile.ZipFile('dataset/vitagraph.zip', 'r') as zip_ref:
    print("Extracting vitagraph.tsv from vitagraph.zip...")
    zip_ref.extract('vitagraph.tsv', 'dataset/')

def read_gz_file(input_path, encoding='utf-8'):
  with gzip.open(input_path, 'rt', encoding=encoding) as file:
    for line in file:
      yield line.strip()

def get_gene_mapping(target):
  input_path = f'dataset/STRING/{target}.protein.aliases.v12.0.txt.gz'
  gene_mapping = {}

  vita_genes = set()
  # extract vitagarh.zip from vitagraph.zip

  with open('dataset/vitagraph.tsv', 'r') as fin:
    for line in fin.readlines()[1:]:
      line = line.strip().split('\t')
      head = line[0].split('::')
      tail = line[2].split('::')
      head_type = head[0]
      head_src, head_id = head[1].split(':')
      tail_type = tail[0]
      tail_src, tail_id = tail[1].split(':')
      if head_type == 'Gene' and head_src == 'NCBI':
        vita_genes.add(head_id)
      if tail_type == 'Gene' and tail_src == 'NCBI':
        vita_genes.add(tail_id)
  
  for line in read_gz_file(input_path):
    line = line.strip().split('\t')
    if line[0] == '#string_protein_id':
      continue
    if line[2] == 'Ensembl_HGNC_entrez_id':
      ensp_id = line[0].split('.')[1]
      ncbi_id = line[1]
      if ensp_id not in gene_mapping and ncbi_id in vita_genes:
        gene_mapping[ensp_id] = ncbi_id

  return gene_mapping

def get_target_orthology_groups(target, mapping):
  input_path = f'dataset/STRING/{target}.protein.orthology.v12.0.txt.gz'
  target_orthology_groups = []

  orthology_groups = {}
  for line in read_gz_file(input_path):
    line = line.strip().split('\t')
    if line[0] == '#protein':
      continue
    p = line[0].split('.')[1]
    group = line[2]
    if 'NOG' in group:
      continue
    if p in mapping:
      if  p not in orthology_groups or 'KOG' in group:
        orthology_groups[mapping[p]] = group
  
  for key in orthology_groups:
    new_line = f'Gene::NCBI:{key}\tORTHOLOGY\tOrthologyGroup::STRING:{orthology_groups[key]}\tSTRING\tProtein-OrthologyGroup\n'
    target_orthology_groups.append(new_line)

  return target_orthology_groups

def get_target_gene_ontologies(target, mapping):
  input_path = f'dataset/STRING/{target}.protein.enrichment.terms.v12.0.txt.gz'
  target_gene_ontologies = []
  for line in read_gz_file(input_path):
    line = line.strip().split('\t')
    if line[0] == '#string_protein_id':
      continue
    p = line[0].split('.')[1]
    interaction = line[1].split('(')[0].replace(' ', '')
    go = line[2]
    if 'GO:' not in go: continue
    go = go.split(':')[1]
    if p in mapping:
      new_line = f'Gene::NCBI:{mapping[p]}\t{interaction}\tGO::GO:{go}\tSTRING\tProtein-GeneOnthology\n'
    target_gene_ontologies.append(new_line)
  return target_gene_ontologies

def save_pathogenkg(target_orthology_groups, target_gene_ontologies, pathogenkg_path):
  with open('dataset/vitagraph.tsv', 'r') as fin:
    with open(pathogenkg_path, 'w') as fout:
      for line in fin.readlines():
        fout.write(line)

      if target_orthology_groups:
        for line in target_orthology_groups:
          fout.write(line)
      if target_gene_ontologies:
        for line in target_gene_ontologies:
          fout.write(line)

  return None

def generate_pathogenkg_per_target(target, pathogenkg_path):
  fun_log_header = '[generate_pathogenkg_per_target]'

  logging.info(f'{fun_log_header} Getting ENSP to NCBI gene id mapping...')
  start = time()
  mapping = get_gene_mapping(target)
  m = len(mapping) if mapping else 0
  total_time = get_total_time(start, time())
  logging.info(f'{fun_log_header} Done in {total_time} seconds | Mapped {m} proteins')

  logging.info(f'{fun_log_header} Getting orthology groups for {target}...')
  start = time()
  target_orthology_groups = get_target_orthology_groups(target, mapping)
  og = len(target_orthology_groups) if target_orthology_groups else 0
  total_time = get_total_time(start, time())
  logging.info(f'{fun_log_header} Done in {total_time} seconds | Added {og} triplets')

  logging.info(f'{fun_log_header} Getting gene ontologies for {target}...')
  start = time()
  target_gene_ontologies = get_target_gene_ontologies(target, mapping)
  go = len(target_gene_ontologies) if target_gene_ontologies else 0
  total_time = get_total_time(start, time())
  logging.info(f'{fun_log_header} Done in {total_time} seconds | Added {go} triplets')

  logging.info(f'{fun_log_header} Saving extended VitaGraph...')
  start = time()
  save_pathogenkg(target_orthology_groups, target_gene_ontologies, pathogenkg_path)
  total_time = get_total_time(start, time())
  logging.info(f'{fun_log_header} Done in {total_time} seconds')
  logging.info(f'Extended VitaGraph saved to {pathogenkg_path} | Number of triplets added: {og+go}')
  return None

if __name__ == '__main__':
  target = '9606' # Human NCBI Taxonomy ID
  pathogenkg_path = os.path.join(out_path, f'VitaGraph_human_extended_test.tsv')
  logging.info(f'Adding GO terms and COGs/KOGs to VitaGraph')
  logging.info(f'Output file will be saved in {pathogenkg_path}')
  start = time()
  generate_pathogenkg_per_target(target, pathogenkg_path)
  total_time = get_total_time(start, time())
  logging.info(f'Generation completed in {total_time} seconds')