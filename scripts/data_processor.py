# scripts/data_processor.py

import os
import pandas as pd
import logging
import traceback
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Define short titles for question categories
RISK_SHORT_TITLE_PREFIX = "Risk"
DELAY_SHORT_TITLE_PREFIX = "Delay"
RECIP_SHORT_TITLE_PREFIX = "Reciprocation"
DONATION_SHORT_TITLE_PREFIX = "Donation"
# Find the most recent run folder in data/processed/
def find_most_recent_run_folder(base_dir='data/processed'):
    """
    Find the most recent run folder in the base directory.
    """
    base_path = Path(base_dir)
    run_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('gps_run_')]
    if not run_folders:
        logger.error(f"No run folders found in {base_dir}.")
        return None
    # Sort by folder name (which contains the timestamp) and pick the most recent one
    most_recent_run_folder = max(run_folders, key=lambda f: f.name)
    logger.info(f"Most recent run folder found: {most_recent_run_folder}")
    return most_recent_run_folder


def process_risk_delay(df):
    """
    For Risk and Delay questions:
    Replace "Option 2" with the number of "Option 1" answers before it.
    """
    logger.info(f"Processing risk/delay questions for {df['Participant ID'].nunique()} participants.")

    # Define masks for Risk and Delay questions
    risk_mask = df['Short Title'].str.startswith(RISK_SHORT_TITLE_PREFIX)
    delay_mask = df['Short Title'].str.startswith(DELAY_SHORT_TITLE_PREFIX)

    # Combine masks
    combined_mask = risk_mask | delay_mask
    target_df = df[combined_mask].copy()

    # Function to replace "Option 2" with count of "Option 1" before it
    def replace_option2(group):
        count_option1 = 0
        for idx, row in group.iterrows():
            if row['Answer'] == "Option 1":
                count_option1 += 1
            elif row['Answer'] == "Option 2":
                group.at[idx, 'Answer'] = count_option1
                count_option1 = 0  # Reset the count after "Option 2"
        return group

    # Apply the function to each participant's group
    target_df = target_df.groupby('Participant ID', group_keys=False).apply(replace_option2)

    # Manually update the relevant rows in the original DataFrame to avoid any index or dtype issues
    df.loc[target_df.index, 'Answer'] = target_df['Answer']

    return df
def clean_and_extract_numbers(df):
    """
    Delete rows where the Answer is 'Option 1' and extract only numbers from the remaining answers.
    """
    # Step 1: Delete rows where the Answer is "Option 1"
    df = df[df['Answer'] != "Option 1"].copy()

    # Step 2: Extract only numbers from the Answer column and replace the full answer with just the number
    def extract_number(answer):
        # Use regex to find the first number in the answer
        match = re.search(r'\d+', str(answer))  # Ensure answer is treated as string
        if match:
            return match.group(0)  # Return the extracted number as a string
        else:
            return None  # Return None if no number found

    # Apply the extract_number function to the 'Answer' column
    df['Answer'] = df['Answer'].apply(extract_number)

    return df

import re

def process_recip_donation(df):
    """
    For Reciprocity and Donation questions:
    Replace the answer with the numerical percentage donated/reciprocated.
    """
    # Function to extract the total value from the question
    def extract_total_amount(question):
        try:
            # Use regex to extract numbers from the question
            match = re.search(r'\d+', question)
            if match:
                return float(match.group(0))
            else:
                logger.warning(f"No number found in question: {question}")
                return None
        except Exception as e:
            logger.warning(f"Error extracting amount from question: {question}, error: {e}")
            return None

    # Function to calculate the percentage based on the answer and the extracted total amount
    def calculate_percentage(answer, total_amount):
        try:
            answer_value = float(answer)
            if total_amount and total_amount > 0:
                return (answer_value / total_amount) * 100
            else:
                logger.warning(f"Invalid total amount: {total_amount}")
                return None
        except ValueError:
            logger.warning(f"Invalid answer value: {answer}")
            return None

    # Process Reciprocity Questions
    recip_mask = df['Short Title'].str.startswith(RECIP_SHORT_TITLE_PREFIX)
    recip_df = df[recip_mask].copy()

    # Process Donation Questions
    donation_mask = df['Short Title'].str.startswith(DONATION_SHORT_TITLE_PREFIX)
    donation_df = df[donation_mask].copy()

    # Apply to Reciprocity: Extract total from question, calculate percentage from answer
    recip_df['Total Amount'] = recip_df['Question'].apply(extract_total_amount)
    recip_df['Answer'] = recip_df.apply(lambda row: calculate_percentage(row['Answer'], row['Total Amount']), axis=1)

    # Apply to Donation: Extract total from question, calculate percentage from answer
    donation_df['Total Amount'] = donation_df['Question'].apply(extract_total_amount)
    donation_df['Answer'] = donation_df.apply(lambda row: calculate_percentage(row['Answer'], row['Total Amount']), axis=1)

    # Update the original DataFrame
    df.update(recip_df)
    df.update(donation_df)

    return df

def process_file(file_path, output_dir):
    """
    Process a single CSV file and save the result in the output directory.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Processing file: {file_path.name}")

        # Check if essential columns exist
        required_columns = ['Participant ID', 'Short Title', 'Answer']
        if not all(column in df.columns for column in required_columns):
            logger.error(f"Missing required columns in {file_path.name}. Skipping file.")
            return

        # Step 1: Process Risk and Delay Questions
        df = process_risk_delay(df)

        # Step 2: Clean the DataFrame by deleting "Option 1" rows and extracting numbers
        df = clean_and_extract_numbers(df)

        # Step 3: Process Reciprocity and Donation Questions (now the answers should be clean)
        df = process_recip_donation(df)

        # Step 4: Save the processed DataFrame to the output directory
        output_file = output_dir / file_path.name
        df.to_csv(output_file, index=False)
        logger.info(f"Processed file saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error processing file {file_path.name}: {e}")
        logger.error(traceback.format_exc())
def main():
    """
    Main function to find the most recent run folder, process all CSV files in the 'gps' subfolder,
    and save the processed files in a new 'gps_proc' subfolder.
    """
    # Find the most recent run folder
    most_recent_run_folder = find_most_recent_run_folder()
    if not most_recent_run_folder:
        return

    # Set the input and output directories
    input_dir = most_recent_run_folder / 'gps'
    output_dir = most_recent_run_folder / 'gps_proc'

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all CSV files in the input directory
    for csv_file in input_dir.glob("*.csv"):
        process_file(csv_file, output_dir)


if __name__ == "__main__":
    main()
