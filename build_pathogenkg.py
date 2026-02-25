#%%
import json
import os
import csv
import gzip
import logging
import argparse
from time import time

from src.bio_utils import get_total_time, is_eukaryote

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)

AVAILABLE_TARGETS_v1 = [
	'83332', '224308', '208964', '99287', '71421', '243230', 
	'85962', '171101', '243277', '294', '1314', '272631',
	'212717', '36329', '237561', '6183', '5664', '185431', '330879'
]
import pandas as pd
taxa_df = pd.read_csv("dataset/DRUGBANK/taxons/drugbank_string_taxa_merged_string_with_pathogen_status.csv")
pathogen_taxa = taxa_df[taxa_df.human_pathogen != 'No']
non_pathogen_taxa = taxa_df[taxa_df.human_pathogen == 'No']
tax_ids = pathogen_taxa.taxonomy_id.astype(str).to_list()
print(f"Preparing marge for {len(tax_ids)} pathogen taxa")

import glob

downloaded_taxa = []
print(os.getcwd())
for file in glob.glob("dataset/pathogenkg/PathogenKG_*.tsv"):
    filename = os.path.basename(file)
    tax_id = filename.split("_")[1].split(".")[0]
    downloaded_taxa.append(tax_id)
    downloaded_taxa = list(set(downloaded_taxa))
print(f"Already buiild taxa IDs len {len(downloaded_taxa)}")

# remove downloadded from tax_ids
tax_ids = [tax_id for tax_id in tax_ids if str(tax_id) not in downloaded_taxa]
print(f"Taxa IDs to build len {len(tax_ids)}")

#%%


AVAILABLE_TARGETS = list(set(tax_ids))

OUT_PATH = 'dataset/pathogenkg/new/'
os.makedirs(OUT_PATH, exist_ok=True)

def read_gz_file(input_path, encoding='utf-8'):
	"""Generator for reading gzipped files line by line."""
	with gzip.open(input_path, 'rt', encoding=encoding) as file:
		for line in file:
			yield line.strip()

def load_string_uniprot_aliases(target):
	"""Load STRING protein -> UniProt alias mapping (source == UniProt_ID)."""
	alias_path = f'dataset/STRING/{target}.protein.aliases.v12.0.txt.gz'
	alias_map = {}
	if not os.path.exists(alias_path):
		return alias_map

	for line in read_gz_file(alias_path):
		parts = line.split('\t')
		if not parts or parts[0] == '#string_protein_id':
			continue
		if len(parts) < 3:
			continue

		string_id, alias, source = parts[0], parts[1], parts[2]
		if source != 'UniProt_ID':
			continue
		if '.' in string_id:
			string_id = string_id.split('.', 1)[1]
		if string_id not in alias_map:
			alias_map[string_id] = alias
	return alias_map

def to_uniprot_alias(protein_id, alias_map):
	"""Return UniProt alias when available, otherwise the original STRING protein id."""
	return alias_map.get(protein_id, protein_id)


def get_target_ppi(target, alias_map):
	"""Extract protein-protein interactions with normalized scores >= 0.6."""
	input_path = f'dataset/STRING/{target}.protein.physical.links.v12.0.txt.gz'
	input_action_file_path = f'dataset/STRING/{target}.protein.actions.v11.0.txt.gz'
	
	seen_pairs = set()
	target_ppi = []
	scores = []
	interactions = []

	actions_d = {
		'binding': 'GENE_BIND',
		'reaction':'REACTION',
		'catalysis':'CATALYSIS',
		'activation':'ACTIVATION',
		'inhibition':'INHIBITION',
		'ptmod':'PTMOD',
		'expression':'EXPRESSION'
	}
	
	# Load action modes if file exists
	action_modes = {}
	if os.path.exists(input_action_file_path):
		for line in read_gz_file(input_action_file_path):
			parts = line.split('\t')
			if parts[0] == 'item_id_a':  # Skip header
				continue
			
			p1 = parts[0].split('.', 1)[1]
			p2 = parts[1].split('.', 1)[1]
			mode = actions_d[parts[2].strip()]
			
			# Store only one direction to avoid duplicates
			pair = tuple(sorted([p1, p2]))
			if pair not in action_modes:
				action_modes[pair] = mode
	
	# First pass: collect all scores and interactions
	for line in read_gz_file(input_path):
		parts = line.split(' ')
		if parts[0] == 'protein1':
			continue
			
		p1 = parts[0].split('.', 1)[1]
		p2 = parts[1].split('.', 1)[1]
		score = int(parts[2])
		pair = tuple(sorted([p1, p2]))
		
		if pair not in seen_pairs:
			seen_pairs.add(pair)
			scores.append(score)
			interactions.append((p1, p2, score))
	
	# Normalize scores to [0, 1] range
	if scores:
		min_score = min(scores)
		max_score = max(scores)
		score_range = max_score - min_score
		
		# Second pass: filter by normalized score >= 0.6
		for p1, p2, score in interactions:
			if score_range > 0:
				normalized_score = (score - min_score) / score_range
			else:
				normalized_score = 1.0  # All scores are identical
			
			if normalized_score >= 0.6:
				pair = tuple(sorted([p1, p2]))
				interaction_type = action_modes.get(pair, 'gene_OTHER_gene')
				p1_alias = to_uniprot_alias(p1, alias_map)
				p2_alias = to_uniprot_alias(p2, alias_map)
				target_ppi.append(f'ExtGene::Uniprot:{p1_alias}\t{interaction_type}\tExtGene::Uniprot:{p2_alias}\tSTRING\tExtGene-ExtGene')
	
	return target_ppi

def get_target_orthology_groups(target, is_eukarya, alias_map):
	"""Extract orthology group mappings."""
	input_path = f'dataset/STRING/{target}.protein.orthology.v12.0.txt.gz'
	orthology_groups = {}
	
	for line in read_gz_file(input_path):
		parts = line.split('\t')
		if parts[0] == '#protein':
			continue
			
		protein = parts[0].split('.', 1)[1]
		group = parts[2]
		
		if 'NOG' in group:
			continue
			
		# Prefer KOG over COG
		if protein not in orthology_groups or 'KOG' in group:
			orthology_groups[protein] = group
	
	# Remove COGs if eukaryotic
	if is_eukarya:
		orthology_groups = {k: v for k, v in orthology_groups.items() if 'COG' not in v}
	
	return [f'ExtGene::Uniprot:{to_uniprot_alias(protein, alias_map)}\tORTHOLOGY\tOrthologyGroup::eggKNOG:{group}\tSTRING\tExtGene-OrthologyGroup'
			for protein, group in orthology_groups.items()]

def get_target_gene_ontologies(target, alias_map):
	"""Extract gene ontology annotations."""
	input_path = f'dataset/STRING/{target}.protein.enrichment.terms.v12.0.txt.gz'
	target_gene_ontologies = []
	
	for line in read_gz_file(input_path):
		parts = line.split('\t')
		if parts[0] == '#string_protein_id':
			continue
			
		protein = parts[0].split('.', 1)[1]
		interaction = parts[1].split('(')[0].replace(' ', '')
		go_term = parts[2]
		
		if 'GO:' not in go_term:
			continue
			
		go_id = go_term.split(':', 1)[1]
		protein_alias = to_uniprot_alias(protein, alias_map)
		target_gene_ontologies.append(
			f'ExtGene::Uniprot:{protein_alias}\t{interaction}\tGO::GO:{go_id}\tSTRING\tExtGene-GeneOnthology'
		)
	
	return target_gene_ontologies

def load_target_proteins(target):
	"""Load target proteins for given taxonomy.
	
	DrugBank_ID,UniProt_ID,taxonomy_id,EntryName
	BE0004508,P46459,9606,ASPQ_PSEPK
	BE0003422,Q96AE4,9606,ASPQ_PSEPK
	BE0003811,P63151,9606,ASPQ_PSEPK
	"""
	targets_path = 'dataset/DRUGBANK/drugbank_uniprot_taxonomy.csv'

	TargetFormat = "UniProt_ID"
	TargetFormat = "EntryName"  # alternative format if needed, but requires additional mapping step from UniProt_ID to EntryName using the same file or an external source like UniProt API.
	# real al columns as strin
	df = pd.read_csv(targets_path, dtype=str)
	#update dataframe with entry name and save i again
		# per favore se manca la colonna EntryName crela usando l'alis json
	alias_json_path = 'dataset/DRUGBANK/uniprot_entry_names.json'
	if not os.path.exists(alias_json_path):
		print(f"Alias JSON file not found: {alias_json_path}")
		return set()
	with open(alias_json_path, 'r') as f:
		alias_map = json.load(f)

	df['EntryName'] = df['UniProt_ID'].map(alias_map)
	df.to_csv(targets_path, index=False)  # save updated dataframe with EntryName column

	df = pd.read_csv(targets_path, dtype=str)
	

	filtered = df[df['taxonomy_id'] == str(target)][TargetFormat].dropna()
	return set(filtered.astype(str))

def load_drugbank_mapping():
	"""Load DrugBank ID to external ID mapping."""
	mapping_path = 'dataset/DRUGBANK/drugbank_drugs_mapping.csv'
	drugbank_mapping = {}
	
	with open(mapping_path, 'r') as fin:
		reader = csv.reader(fin)
		next(reader)  # Skip header
		for row in reader:
			drugbank_id, pubchem_id, chebi_id, chembl_id = row[:4]
			# Fix: proper priority selection
			star_id = (pubchem_id and f'Pubchem:{pubchem_id}' or 
					  chebi_id and f'CHEBI:{chebi_id}' or 
					  chembl_id and f'CHEMBL:{chembl_id}' or 
					  f'DrugBank:{drugbank_id}')
			drugbank_mapping[drugbank_id] = star_id
	
	return drugbank_mapping

def get_target_drugs(target):
	"""Extract drug-target interactions."""
	target_proteins = load_target_proteins(target)
	if not target_proteins:
		return []
	
	drugbank_mapping = load_drugbank_mapping()
	target_drugs = []

	# load alias map to convert STRING protein IDs to UniProt IDs when possible
	file_path  = "dataset/DRUGBANK/uniprot_entry_names.json"
	"""
	{
    "P48167": "GLRB_HUMAN",
    "O75311": "GLRA3_HUMAN",
    "P23416": "GLRA2_HUMAN",
    "P23415": "GLRA1_HUMAN",
    "P35372": "OPRM_HUMAN",
    "P41145": "OPRK_HUMAN",
	"""
	alias_map = {}
	if os.path.exists(file_path):
		with open(file_path, 'r') as f:
			alias_map = json.load(f)
	
	interactions_path = 'dataset/DRUGBANK/pathogenkg_drug_target_triples.csv'
	with open(interactions_path, 'r') as fin:
		reader = csv.reader(fin)
		next(reader)  # Skip header
		
		for row in reader:
			head, interaction, tail, source, type_ = row[:5]
			
			head_id = head.split('::')[1].split(':', 1)[1]
			tail_id = tail.split('::')[1].split(':', 1)[1]

			# use convertion dict to replace tail_id with alias if exists
			if tail_id in alias_map:
				tail_id = alias_map[tail_id]
			
			if tail_id in target_proteins and head_id in drugbank_mapping:
				compound_id = drugbank_mapping[head_id]
				target_drugs.append(
					f'Compound::{compound_id}\t{interaction}\tExtGene::Uniprot:{tail_id}\t{source}\tCompound-ExtGene'
				)
				# print(f"Added drug-target interaction: {compound_id} -> {tail_id} for target {target}")
	
	return target_drugs

def save_pathogenkg(data_lists, pathogenkg_path):
	"""Save all data to PathogenKG file."""
	with open(pathogenkg_path, 'w') as fout:
		fout.write('head\tinteraction\ttail\tsource\ttype\n')
		for data_list in data_lists:
			for line in data_list:
				fout.write(f'{line}\n')

def extract_extgene_uniprot_ids(triples):
	"""Extract ExtGene::Uniprot IDs from head and tail fields of triples."""
	proteins = set()
	for line in triples:
		parts = line.split('\t')
		if len(parts) < 3:
			continue
		head = parts[0]
		tail = parts[2]
		if head.startswith('ExtGene::Uniprot:'):
			proteins.add(head.split(':', 2)[2])
		if tail.startswith('ExtGene::Uniprot:'):
			proteins.add(tail.split(':', 2)[2])
	return proteins

def update_string_drugbank_report(target, pathogenkg_path):
	"""Update report using only the generated dataset.

	- total_drugbank_targets: unique UniProt IDs that appear in DrugBank TARGET relations
	- related_drugbank_targets: among those, how many also appear in non-DrugBank-target relations
	"""
	report_path = os.path.join(OUT_PATH, 'drugbank_string_target_overlap_report.txt')
	os.makedirs(os.path.dirname(report_path), exist_ok=True)

	drugbank_targets = set()
	other_relation_proteins = set()

	with open(pathogenkg_path, 'r', encoding='utf-8') as fin:
		next(fin, None)  # skip header
		for line in fin:
			line = line.strip()
			if not line:
				continue
			parts = line.split('\t')
			if len(parts) < 5:
				continue

			head, interaction, tail, source, type_ = parts[:5]
			is_drug_target_relation = (
				head.startswith('Compound::')
				and interaction == 'TARGET'
				and tail.startswith('ExtGene::Uniprot:')
				and source == 'DRUGBANK'
				and type_ == 'Compound-ExtGene'
			)

			if is_drug_target_relation:
				drugbank_targets.add(tail.split(':', 2)[2])
				continue

			if head.startswith('ExtGene::Uniprot:'):
				other_relation_proteins.add(head.split(':', 2)[2])
			if tail.startswith('ExtGene::Uniprot:'):
				other_relation_proteins.add(tail.split(':', 2)[2])

	related = len(drugbank_targets.intersection(other_relation_proteins))
	total = len(drugbank_targets)
	percentage = (related / total * 100.0) if total else 0.0

	current_rows = {}
	if os.path.exists(report_path):
		with open(report_path, 'r', encoding='utf-8') as fin:
			for line in fin:
				line = line.strip()
				if not line or line.startswith('taxid\t'):
					continue
				row = line.split('\t')
				if row:
					current_rows[row[0]] = line

	current_rows[target] = f'{target}\t{related}\t{total}\t{percentage:.2f}'

	with open(report_path, 'w', encoding='utf-8') as fout:
		fout.write('taxid\trelated_drugbank_targets\ttotal_drugbank_targets\trelation_percent\n')
		for taxid in sorted(current_rows, key=lambda x: int(x)):
			fout.write(f'{current_rows[taxid]}\n')

def log_process(name, target, func, *args):
	"""Helper to log process timing and results."""
	logging.info(f'Getting {name} for {target}...')
	start = time()
	result = func(*args)
	count = len(result) if result else 0
	elapsed = get_total_time(start, time())
	logging.info(f'Done in {elapsed}s | Added {count} triplets')
	return result

def generate_pathogenkg_per_target(target, pathogenkg_path, is_eukarya):
	"""Generate PathogenKG file for target organism."""
	alias_map = load_string_uniprot_aliases(target)
	# Process all data types
	target_ppi = log_process('ppi', target, get_target_ppi, target, alias_map)
	target_orthology = log_process('orthology groups', target, get_target_orthology_groups, target, is_eukarya, alias_map)
	target_go = log_process('gene ontologies', target, get_target_gene_ontologies, target, alias_map)
	target_drugs = log_process('drugs', target, get_target_drugs, target)
	
	# Save results
	logging.info(f'Saving PathogenKG for {target}...')
	start = time()
	save_pathogenkg([target_ppi, target_orthology, target_go, target_drugs], pathogenkg_path)
	elapsed = get_total_time(start, time())
	
	total_lines = sum(len(data) for data in [target_ppi, target_orthology, target_go, target_drugs])
	logging.info(f'Done in {elapsed}s')
	logging.info(f'PathogenKG for {target} saved to {pathogenkg_path} | {total_lines} lines')

	update_string_drugbank_report(target, pathogenkg_path)

"""
Command-line interface example:
python build_pathogenkg.py --target 83332

"""

if __name__ == '__main__':
	# test def load_target_proteins(target):
	target = '83332'
	proteins = load_target_proteins(target)
	print(f"Proteins for target {target}: {proteins}")


#%%
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate PathogenKG files for target organisms')
	parser.add_argument('--target', default='83332', choices=AVAILABLE_TARGETS,
					   help='Target taxonomy ID')
	args = parser.parse_args()
	
	target = args.target

	for target in AVAILABLE_TARGETS:
	# for target in AVAILABLE_TARGETS_v1:
	# for target in [target]:
		# is_eukarya = is_eukaryote(target)
		is_eukarya = False
		pathogenkg_path = os.path.join(OUT_PATH, f'PathogenKG_{target}.tsv')
		
		logging.info(f'Generating PathogenKG for taxonomy {target} | is_eukarya: {is_eukarya}')
		logging.info(f'Output: {pathogenkg_path}')
		
		start = time()
		generate_pathogenkg_per_target(target, pathogenkg_path, is_eukarya)
		total_time = get_total_time(start, time())
		logging.info(f'Generation completed in {total_time}s')
