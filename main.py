import asyncio
import logging
import time
import subprocess
import hashlib
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys
from aiohttp import ClientSession
from scripts.utils import load_config
from scripts.data_loader import load_stakes_data
from scripts.prompt_generator import generate_system_prompts
from scripts.participant_processor import (
    process_participant,
    country_locks,
    processed_counts_per_country
)
from scripts.question_generator import (
    generate_risk_questions_for_country,
    generate_time_questions_for_country,
    generate_recip_questions_for_country,
    generate_donation_questions_for_country
)

# Configure logging
logging.basicConfig(
    format='%(message)s',  # Only display the message part
    level=logging.INFO     # Set the logging level
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate unique participant hash based on system prompt
def generate_unique_hash(system_prompt):
    return hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()

async def main():
    config = load_config()
    countries = config['countries']
    num_samples = config['settings']['num_samples_per_country_gender']  # Renamed in config.yaml
    semaphore_limit = config['settings']['semaphore_limit']
    country_currency_dict = config['country_currency_dict']
    gps_base_folder = Path(config['paths']['gps_folder'])

    # Create a unique run folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = gps_base_folder.parent / f"gps_run_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    gps_folder_path = run_folder / "gps"
    gps_folder_path.mkdir(parents=True, exist_ok=True)

    # Load stakes data
    stakes_df, time_stakes_df, recip_stakes_df, donation_stakes_df = load_stakes_data()

    # Generate system prompts
    system_prompts = generate_system_prompts(countries, num_samples)

    # Prepare existing hashes per country
    existing_hashes_per_country = {}
    total_participants_per_country = {}

    # Initialize processed counts
    for country in countries:
        processed_counts_per_country[country] = 0

    # Load existing participant hashes per country from the new run folder (likely empty)
    for country in countries:
        output_file = gps_folder_path / f"results_{country}.csv"
        if output_file.exists():
            try:
                df_existing = pd.read_csv(output_file)
                if 'Participant Hash' in df_existing.columns:
                    hashes = set(df_existing['Participant Hash'])
                    existing_hashes_per_country[country] = hashes
                    participant_count = len(set(df_existing['Participant ID']))
                else:
                    existing_hashes_per_country[country] = set()
                    participant_count = 0
            except pd.errors.ParserError as e:
                logger.error(f"Error reading {output_file}: {e}")
                existing_hashes_per_country[country] = set()
                participant_count = 0
        else:
            existing_hashes_per_country[country] = set()
            participant_count = 0
        total_participants_per_country[country] = participant_count

    # Print initial overview
    logger.info("Initial participants processed per country:")
    for country in countries:
        logger.info(f"{country}: {total_participants_per_country.get(country, 0)} participants")

    # Define additional questions and short titles
    questions1 = [
        "How willing or unwilling are you to take risks? Using a scale from 0 to 10, please answer with a single number.",
        "How willing are you to give up something that is beneficial for you today in order to benefit more from that in the future? Please indicate your answer on a scale from 0 to 10, answer with a single number, don't add other words.",
        "How willing are you to punish someone who treats you unfairly, even if there may be costs for you? Please indicate your answer on a scale from 0 to 10, please answer with a single number, don't add other words.",
        "How willing are you to punish someone who treats others unfairly, even if there may be costs for you? Please indicate your answer on a scale from 0 to 10, answer with a single number, don't add other words.",
        "How willing are you to give to good causes without expecting anything in return? Please indicate your answer on a scale from 0 to 10, answer with a single number, don't add other words.",
        "How well does the following statement describe you? Please indicate your answer on a scale from 0 to 10. Please answer with a single number: When someone does me a favor, I am willing to return it.",
        "How well does the following statement describe you? Please indicate your answer on a scale from 0 to 10. Please answer with a single number: If I am treated very unjustly, I will take revenge at the first occasion, even if there is a cost to do so.",
        "How well does the following statement describe you? Please indicate your answer on a scale from 0 to 10. Please answer with a single number: I assume that people have only the best intentions.",
        "How well does the following statement describe you? Please indicate your answer on a scale from 0 to 10. Please answer with a single number: I am good at math.",
        "How well does the following statement describe you? Please indicate your answer on a scale from 0 to 10. Please answer with a single number: I tend to postpone tasks even if I know it would be better to do them right away.",
        "You are in an area you are not familiar with, and you realize that you lost your way. You ask a stranger for directions. The stranger offers to take you to your destination. Helping you costs the stranger about 40 U.S. dollars in total. However, the stranger says they do not want any money from you. You have six presents with you with values 10, 20, 30, 50, and 60$ respectively. What is the value of the present you would give? Only answer with a single number, answer with 0 if you don't want to give one.",
        "Imagine the following situation: Today you unexpectedly received 1,600 U.S. dollars. How much of this amount would you donate to a good cause? Please answer with a single number."
    ]

    short_titles1 = [
        "Willingness to take risk",
        "Willingness to delay consumption",
        "Personal retribution",
        "Retribution on others' behalf",
        "Willingness to donate",
        "Will return favor",
        "Will do revenge",
        "People have best intentions",
        "Good at math",
        "Procrastinate",
        "Present giving",
        "Donate"
    ]

    start_time = time.time()
    total_tasks = len(system_prompts)
    current_count = 0
    update_interval = max(1, total_tasks // 100)  # Update every 1% of progress

    semaphore = asyncio.Semaphore(semaphore_limit)

    async with ClientSession() as session:
        tasks = []
        for system_prompt in system_prompts:
            participant_hash = generate_unique_hash(system_prompt)

            # Check if the participant has already been processed
            country = system_prompt.split("from")[1].split()[0].strip()  # Assuming "from Country" format in prompt
            if participant_hash in existing_hashes_per_country.get(country, set()):
                logger.info(f"Participant already processed for {country}, skipping.")
                continue

            task = asyncio.create_task(
                process_participant(
                    session, system_prompt, semaphore,
                    existing_hashes_per_country, stakes_df, time_stakes_df,
                    recip_stakes_df, donation_stakes_df,
                    questions1, short_titles1,
                    gps_folder_path  # pass run folder path
                )
            )
            tasks.append(task)

        for future in asyncio.as_completed(tasks):
            await future
            current_count += 1

            if current_count % update_interval == 0 or current_count == total_tasks:
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / current_count) * total_tasks
                estimated_remaining_time = estimated_total_time - elapsed_time
                logger.info(
                    f"Processed {current_count}/{total_tasks} participants. "
                    f"Estimated time remaining: {estimated_remaining_time / 60:.2f} minutes"
                )

                # Update total participants per country
                for country in processed_counts_per_country:
                    total_participants_per_country[country] = total_participants_per_country.get(country, 0) + processed_counts_per_country[country]
                    processed_counts_per_country[country] = 0  # Reset the count after updating

                # Print periodic overview
                print("\nParticipants processed per country so far:")
                for country in countries:
                    count = total_participants_per_country.get(country, 0)
                    print(f"{country}: {count} participants")
                print("\n")
    script_dir = Path(__file__).resolve().parent  # This gets the directory of the current script
    data_processor_script = script_dir / 'scripts' / 'data_processor.py'  # Append the correct subdirectory and script name

    try:
        # Use the absolute path to the script
        subprocess.run([sys.executable, str(data_processor_script)], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Data processing failed: {e}")
    except FileNotFoundError:
        logger.error("Python executable not found. Check the subprocess.run call.")
if __name__ == "__main__":
    asyncio.run(main())

