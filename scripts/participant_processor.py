# scripts/participant_processor.py

import asyncio
import logging
import uuid
import pandas as pd

from aiohttp import ClientSession
from .utils import compute_hash
from .prompt_generator import extract_age_gender_country
from .api_client import ask_economic_question
from .data_saver import save_results_for_participant
from .question_generator import (
    generate_risk_questions_for_country,
    generate_time_questions_for_country,
    generate_recip_questions_for_country,
    generate_donation_questions_for_country
)

# Initialize locks and counts
country_locks = {}
processed_counts_per_country = {}

async def process_participant(
    session, system_prompt, semaphore, existing_hashes_per_country,
    stakes_df, time_stakes_df, recip_stakes_df, donation_stakes_df,
    questions1, short_titles1, gps_folder_path
):
    """
    Process a single participant by generating questions, making API calls,
    and saving the results.

    Parameters:
    - session: The aiohttp ClientSession for making API calls.
    - system_prompt: The system prompt describing the participant.
    - semaphore: An asyncio.Semaphore to limit concurrent API calls.
    - existing_hashes_per_country: A dictionary of existing participant hashes per country.
    - stakes_df, time_stakes_df, recip_stakes_df, donation_stakes_df: DataFrames with stakes data.
    - questions1, short_titles1: Lists of additional general questions and their short titles.
    - gps_folder_path: Path to the current run's GPS folder.
    """
    participant_id = str(uuid.uuid4())  # Generate a unique UUID for participant
    participant_hash = compute_hash(system_prompt)
    age, gender, country = extract_age_gender_country(system_prompt)

    if country is None:
        logging.warning(f"Could not extract country for participant {participant_id}. Skipping.")
        return

    if participant_hash in existing_hashes_per_country.get(country, set()):
        logging.info(f"Participant {participant_id} from {country} already processed. Skipping.")
        return

    logging.info(f"Processing participant {participant_id} from {country}")

    results = []

    # Generate questions based on country
    risk_questions, risk_short_titles = generate_risk_questions_for_country(country, stakes_df)
    time_questions, time_short_titles = generate_time_questions_for_country(country, time_stakes_df)
    recip_questions, recip_short_titles = generate_recip_questions_for_country(country, recip_stakes_df)
    donation_questions, donation_short_titles = generate_donation_questions_for_country(country, donation_stakes_df)

    # Combine all questions and short titles
    questions = questions1 + risk_questions + time_questions + recip_questions + donation_questions
    short_titles = short_titles1 + risk_short_titles + time_short_titles + recip_short_titles + donation_short_titles
    question_to_short_title = dict(zip(questions, short_titles))

    # Flags to stop asking certain questions based on responses
    stop_asking_risk_questions = False
    stop_asking_time_questions = False

    for question in questions:
        if stop_asking_risk_questions and question in risk_questions:
            continue
        if stop_asking_time_questions and question in time_questions:
            continue

        async with semaphore:
            answer = await ask_economic_question(session, question, system_prompt)

        # Logic to stop asking further risk or time preference questions based on the participant's answer
        if "Option 2" in answer:
            if question in risk_questions:
                stop_asking_risk_questions = True
            elif question in time_questions:
                stop_asking_time_questions = True

        # Append the result to the list
        results.append({
            'Participant ID': participant_id,
            'Participant Hash': participant_hash,
            'Question': question,
            'Answer': answer,
            'Short Title': question_to_short_title[question],
            'Age': age,
            'Gender': gender,
            'Country': country
        })

    logging.info(f"Participant {participant_id} processed.")

    # Update processed counts per country
    processed_counts_per_country[country] = processed_counts_per_country.get(country, 0) + 1

    # Save results
    await save_results_for_participant(
        results, country, existing_hashes_per_country, country_locks, processed_counts_per_country,
        gps_folder_path
    )