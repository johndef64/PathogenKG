#%%
import os
import logging
import argparse
import zipfile
from time import time
from collections import defaultdict
import pandas as pd
from src.bio_utils import get_total_time

minimal_available_targets = [
    '83332', '224308', '208964', '99287', '71421', '243230',
    '85962', '171101', '243277', '294', '1314', '272631',
    '212717', '36329', '237561', '6183', '5664', '185431', '330879'
]
# convert to integer taxonomy IDs
minimal_available_targets_int = [int(tax_id) for tax_id in minimal_available_targets]


import pandas as pd
taxa_df = pd.read_csv("dataset/DRUGBANK/taxons/drugbank_string_taxa_merged_string_with_pathogen_status.csv")

taxa_df[taxa_df['STRING_type'] == "core"].domain.value_counts()
taxa_df[taxa_df['STRING_type'] == "periphery"].domain.value_counts()
# taxa_df

#%%
taxa_df.human_pathogen.value_counts()

pathogen_taxa = taxa_df[taxa_df.human_pathogen != 'No']
non_pathogen_taxa = taxa_df[taxa_df.human_pathogen == 'No']
pathogen_taxa_core = pathogen_taxa[pathogen_taxa['STRING_type'] == 'core']
pathogen_taxa_periphery = pathogen_taxa[pathogen_taxa['STRING_type'] == 'periphery']
pathogen_taxa.STRING_type.value_counts() 

pathogen_taxa
#%%
non_pathogen_taxa.STRING_type.value_counts() 
#%%
taxa_df[taxa_df['taxonomy_id'].isin(minimal_available_targets_int)].domain.value_counts()

#%%

tax_ids = pathogen_taxa.taxonomy_id.astype(str).to_list()
available_targets = list(set(tax_ids) )
# available_targets = minimal_available_targets 

# ho fatto una modifica titti i tsv ora sono in pathogenkg_all_pathogens.zip
# devi caricare i tsv da questo zip e non da i tsv liberi

file_path = f'dataset/pathogenkg/'
output_path = "dataset/"

def concatenate_all_targets(targets, out_path):
    """Concatenate all target PathogenKG TSV files (same column structure).

    Output contains a single header row and then all rows from all targets, in order.
    """
    logging_header = '[concatenate_all_targets]'

    expected_cols = ['head', 'interaction', 'tail', 'source', 'type']

    lines_per_target = defaultdict(int)
    dfs = []
    zip_path = os.path.join(file_path, 'pathogenkg_all_pathogens.zip')
    use_zip_source = os.path.exists(zip_path)
    zip_ref = None
    zip_members = {}

    logging.info(f'{logging_header} Concatenating knowledge graphs from {len(targets)} targets...')
    if use_zip_source:
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_members = {
            os.path.basename(member): member
            for member in zip_ref.namelist()
            if member.endswith('.tsv')
        }
        logging.info(f'{logging_header} Loading input files from ZIP: {zip_path}')
    else:
        logging.info(f'{logging_header} ZIP not found, loading input files from directory: {file_path}')

    try:
        for target in targets:
            target_filename = f'PathogenKG_{target}.tsv'
            target_path = os.path.join(file_path, target_filename)

            if use_zip_source:
                zip_member = zip_members.get(target_filename)
                if zip_member is None:
                    logging.warning(f'{logging_header} Skipping missing file in ZIP: {target_filename}')
                    continue
                try:
                    with zip_ref.open(zip_member) as fin:
                        df = pd.read_csv(fin, sep='\t', dtype=str)
                except Exception as exc:
                    logging.warning(f'{logging_header} Failed reading {target_filename} from ZIP: {exc}')
                    continue
            else:
                if not os.path.exists(target_path):
                    logging.warning(f'{logging_header} Skipping missing file: {target_path}')
                    continue

                try:
                    df = pd.read_csv(target_path, sep='\t', dtype=str)
                except Exception as exc:
                    logging.warning(f'{logging_header} Failed reading {target_path}: {exc}')
                    continue

            if df.empty:
                lines_per_target[target] = 0
                logging.warning(f'{logging_header} Empty file (no rows): {target_filename}')
                continue

            if list(df.columns) != expected_cols:
                if all(col in df.columns for col in expected_cols):
                    df = df[expected_cols]
                else:
                    logging.warning(
                        f"{logging_header} Unexpected columns in {target_filename}: {list(df.columns)} (expected {expected_cols})"
                    )
                    continue

            df = df.dropna(how='all')
            lines_per_target[target] = len(df)
            logging.info(f'{logging_header} Target {target}: {len(df)} lines appended')
            dfs.append(df)
    finally:
        if zip_ref is not None:
            zip_ref.close()

    out_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=expected_cols)
    out_path = out_path.replace('.tsv', f'.tsv.zip')
    # save as compressed TSV
    out_df.to_csv(out_path, sep='\t', index=False, compression='zip')

    logging.info(f'{logging_header} Done')
    return int(len(out_df)), lines_per_target

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description='Concatenate all target PathogenKG TSV files (same column structure)'
    )
    parser.add_argument('--output', default=f'PathogenKG_n{str(len(available_targets))}.tsv',
                        help='Output filename')
    args = parser.parse_args()
    
    out_path = os.path.join(output_path, args.output)
    
    logging.info(f'Merging all {len(available_targets)} targets')
    logging.info(f'Output: {out_path}')
    
    start = time()
    total_lines, lines_per_target = concatenate_all_targets(available_targets, out_path)
    elapsed = get_total_time(start, time())
    
    logging.info(f'Completed in {elapsed}s - Total lines: {total_lines}')
    logging.info(f'Lines per target: {dict(lines_per_target)}')

# %%
