#%%
import os
import logging
import argparse
from time import time
from collections import defaultdict
import pandas as pd
from src.bio_utils import get_total_time

minimal_available_targets = [
    '83332', '224308', '208964', '99287', '71421', '243230',
    '85962', '171101', '243277', '294', '1314', '272631',
    '212717', '36329', '237561', '6183', '5664', '185431', '330879'
]

import pandas as pd
taxa_df = pd.read_csv("dataset/DRUGBANK/taxons/drugbank_microorganisms_with_pathogen_status.csv")
tax_ids = taxa_df.taxonomy_id.astype(str).to_list()
available_targets = list(set(tax_ids) )
# available_targets = minimal_available_targets 

file_path = f'dataset/pathogenkg/'

def concatenate_all_targets(targets, out_path):
    """Concatenate all target PathogenKG TSV files (same column structure).

    Output contains a single header row and then all rows from all targets, in order.
    """
    logging_header = '[concatenate_all_targets]'

    expected_cols = ['head', 'interaction', 'tail', 'source', 'type']

    lines_per_target = defaultdict(int)
    dfs = []

    logging.info(f'{logging_header} Concatenating knowledge graphs from {len(targets)} targets...')

    for target in targets:
        target_path = os.path.join(file_path, f'PathogenKG_{target}.tsv')
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
            logging.warning(f'{logging_header} Empty file (no rows): {target_path}')
            continue

        if list(df.columns) != expected_cols:
            if all(col in df.columns for col in expected_cols):
                df = df[expected_cols]
            else:
                logging.warning(
                    f"{logging_header} Unexpected columns in {target_path}: {list(df.columns)} (expected {expected_cols})"
                )
                continue

        df = df.dropna(how='all')
        lines_per_target[target] = len(df)
        logging.info(f'{logging_header} Target {target}: {len(df)} lines appended')
        dfs.append(df)

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
    
    out_path = os.path.join(file_path, args.output)
    
    logging.info(f'Merging all {len(available_targets)} targets')
    logging.info(f'Output: {out_path}')
    
    start = time()
    total_lines, lines_per_target = concatenate_all_targets(available_targets, out_path)
    elapsed = get_total_time(start, time())
    
    logging.info(f'Completed in {elapsed}s - Total lines: {total_lines}')
    logging.info(f'Lines per target: {dict(lines_per_target)}')

# %%
