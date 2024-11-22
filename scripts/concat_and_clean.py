# concat_and_clean.py

import pandas as pd
from pathlib import Path
import logging
import yaml
import argparse

class GPSDataConcatenator:
    def __init__(self, base_folder: Path, output_csv: Path):
        self.base_folder = base_folder
        self.output_csv = output_csv

    def find_most_recent_run_folder(self) -> Path:
        """
        Find the most recent run folder based on its creation/modification time.
        """
        run_folders = [f for f in self.base_folder.iterdir() if f.is_dir() and f.name.startswith("gps_run_")]

        if not run_folders:
            logging.error("No run folders found.")
            return None

        most_recent_run_folder = max(run_folders, key=lambda f: f.stat().st_mtime)
        logging.info(f"Most recent run folder: {most_recent_run_folder}")
        return most_recent_run_folder

    def concat_all_csv_files(self, gps_proc_folder: Path) -> pd.DataFrame:
        """
        Concatenate all CSV files in the 'gps_proc' subfolder.
        """
        csv_files = list(gps_proc_folder.glob("*.csv"))

        if not csv_files:
            logging.error("No CSV files found in the gps_proc folder.")
            return None

        df_list = []
        for csv_file in csv_files:
            logging.info(f"Reading file: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                df_list.append(df)
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty CSV file skipped: {csv_file.name}")
            except Exception as e:
                logging.error(f"Error reading {csv_file.name}: {e}")

        if not df_list:
            logging.error("No valid CSV files to concatenate.")
            return None

        concatenated_df = pd.concat(df_list, ignore_index=True)
        logging.info(f"Concatenated {len(df_list)} CSV files.")
        return concatenated_df

    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning such as removing duplicates and handling missing values.
        """
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_dropped = initial_count - len(df)
        if duplicates_dropped > 0:
            logging.info(f"Dropped {duplicates_dropped} duplicate rows.")

        # Example of handling missing values: drop rows where all elements are NaN
        df = df.dropna(how='all')
        logging.info("Dropped rows where all elements are NaN.")

        # You can add more basic cleaning steps here as needed

        return df

    def run(self):
        """
        Execute the concatenation and basic cleaning process.
        """
        most_recent_run_folder = self.find_most_recent_run_folder()

        if not most_recent_run_folder:
            logging.error("Run folder not found. Exiting.")
            return

        gps_proc_folder = most_recent_run_folder / "gps_proc"

        if not gps_proc_folder.exists():
            logging.error(f"'gps_proc' folder not found in {most_recent_run_folder}.")
            return

        concatenated_df = self.concat_all_csv_files(gps_proc_folder)

        if concatenated_df is not None:
            cleaned_df = self.basic_cleaning(concatenated_df)
            cleaned_df.to_csv(self.output_csv, index=False)
            logging.info(f"Cleaned data saved to: {self.output_csv}")
        else:
            logging.error("Data concatenation failed. Exiting.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Concatenate and clean GPS data.")
    parser.add_argument('--base_folder', type=str, required=True, help='Base folder path for processed data.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the concatenated and cleaned CSV file.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).')
    args = parser.parse_args()

    # Configure logging with timestamp and dynamic level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=numeric_level,  # Set based on argument
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler("concat_and_clean.log")  # Log to a file
        ]
    )

    # Define paths
    base_folder = Path(args.base_folder)
    output_csv = Path(args.output_csv)

    # Create an instance of GPSDataConcatenator and run
    concatenator = GPSDataConcatenator(base_folder, output_csv)
    concatenator.run()