# scripts/data_saver.py

import asyncio
import pandas as pd
import os
import logging

async def save_results_for_participant(
    results, country, existing_hashes_per_country, country_locks, processed_counts_per_country,
    gps_folder_path
):
    """
    Save the participant's results to a CSV file in the specified GPS folder.

    Parameters:
    - results: List of dictionaries containing participant responses.
    - country: The country of the participant.
    - existing_hashes_per_country: Existing participant hashes to avoid duplicates.
    - country_locks: Asyncio locks per country to prevent race conditions.
    - processed_counts_per_country: Counts of processed participants per country.
    - gps_folder_path: Path to the current run's GPS folder.
    """
    try:
        # Ensure the gps folder path exists
        os.makedirs(gps_folder_path, exist_ok=True)
        output_file = os.path.join(gps_folder_path, f"results_{country}.csv")

        # Get or create a lock for the country
        lock = country_locks.setdefault(country, asyncio.Lock())

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)

        async with lock:
            try:
                # Check if the file exists
                if os.path.exists(output_file):
                    # Read existing data
                    existing_df = pd.read_csv(output_file)

                    # Append new data
                    df_combined = pd.concat([existing_df, df_results], ignore_index=True)
                else:
                    df_combined = df_results

                # Write combined data back to CSV
                df_combined.to_csv(output_file, index=False)

                # Update existing_hashes_per_country
                existing_hashes = existing_hashes_per_country.get(country, set())
                existing_hashes.update(df_results['Participant Hash'])
                existing_hashes_per_country[country] = existing_hashes

                # Update processed counts per country
                processed_counts_per_country[country] = processed_counts_per_country.get(country, 0) + 1

                logging.info(f"Results saved for participant in {country}")

            except Exception as e:
                logging.error(f"Error saving results for {country}: {e}")

    except Exception as e:
        logging.error(f"Unexpected error in save_results_for_participant: {e}")