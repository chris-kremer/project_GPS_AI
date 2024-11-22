# scripts/data_loader.py

import pandas as pd
from .utils import load_config

def load_stakes_data():
    config = load_config()
    stakes_file_path = config['paths']['stakes_file']
    stakes_sheets = pd.read_excel(
        stakes_file_path,
        sheet_name=['QXI-RISK', 'QXIV-TIME', 'QXII-RECIP', 'QXIII-DONATION'],
        header=1
    )
    stakes_df = stakes_sheets['QXI-RISK']
    time_stakes_df = stakes_sheets['QXIV-TIME']
    recip_stakes_df = stakes_sheets['QXII-RECIP']
    donation_stakes_df = stakes_sheets['QXIII-DONATION']
    return stakes_df, time_stakes_df, recip_stakes_df, donation_stakes_df