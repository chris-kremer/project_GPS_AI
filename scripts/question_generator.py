# scripts/question_generator.py

import random
import re
import logging
from .utils import load_config

# Load configurations
config = load_config()
country_currency_dict = config['country_currency_dict']

import random

def generate_system_prompts(countries, num_samples=50):
    """
    Generate system prompts for participants based on country, age, and gender.
    If the country is 'None', generate two times num_samples of a simplified prompt.
    """
    system_prompts = []
    for country in countries:
        if country == "None":
            for _ in range(2 * num_samples):
                system_prompt = "You are participating in an economics experiment."
                system_prompts.append(system_prompt)
        else:
            for gender in ["male", "female"]:
                for _ in range(num_samples):
                    age = random.randint(18, 80)
                    system_prompt = f"You are a {age}-year-old {gender} from {country} participating in an economics experiment."
                    system_prompts.append(system_prompt)
    return system_prompts

def extract_age_gender_country(system_prompt):
    """
    Extract age, gender, and country from a system prompt.
    """
    pattern = r"You are a (\d+)-year-old (\w+) from ([\w\s\-']+?) participating in an economics experiment\."
    match = re.match(pattern, system_prompt)
    if match:
        age = int(match.group(1))
        gender = match.group(2)
        country = match.group(3).strip()
        return age, gender, country
    else:
        logging.warning(f"Could not extract age, gender, country from system prompt: {system_prompt}")
        return None, None, None

def generate_risk_questions_for_country(country, stakes_df):
    """
    Generate risk-related questions for a specific country based on the stakes data.
    """
    country_row = stakes_df[stakes_df['Country'] == country]

    if not country_row.empty:
        lottery_size = country_row.iloc[0]['Lottery Size (if wins, 0 otherwise)']
        safe_options = country_row.iloc[0][2:].dropna().values
        safe_options = sorted(safe_options)
        currency = country_currency_dict.get(country, "Currency not found")

        questions = []
        short_titles = []

        for safe_option in safe_options:
            question = (
                f"What would you prefer? Option 1: A draw with a 50-percent chance of receiving "
                f"{lottery_size} {currency} and the same 50-percent chance of receiving nothing, "
                f"OR Option 2: {safe_option} {currency} as a sure payment? "
                f"Answer 'Option 1' or 'Option 2'. Don't add other words or explanations."
            )
            questions.append(question)
            short_titles.append(f"Risk {safe_option}")

        return questions, short_titles
    else:
        logging.warning(f"Stakes not found for country: {country} in risk stakes")
        return [], []

def generate_time_questions_for_country(country, time_stakes_df):
    """
    Generate time-preference questions for a specific country based on the stakes data.
    """
    country_row = time_stakes_df[time_stakes_df['Country'] == country]

    if not country_row.empty:
        today_payment = country_row.iloc[0]['"Today"']
        future_payments = country_row.iloc[0][2:].dropna().values
        future_payments = sorted(future_payments)
        currency = country_currency_dict.get(country, "Currency not found")

        questions = []
        short_titles = []

        for future_payment in future_payments:
            question = (
                f"What would you prefer? Option 1: A payment of {today_payment} {currency} today "
                f"OR Option 2: {future_payment} {currency} in 12 months. "
                f"Please assume there is no inflation. Answer 'Option 1' or 'Option 2'. Don't add other words or explanations."
            )
            questions.append(question)
            short_titles.append(f"Delay {future_payment}")

        return questions, short_titles
    else:
        logging.warning(f"Stakes not found for country: {country} in time preference stakes")
        return [], []

def generate_recip_questions_for_country(country, recip_stakes_df):
    """
    Generate reciprocity questions for a specific country based on the stakes data.
    """
    country_row = recip_stakes_df[recip_stakes_df['Country'] == country]

    if not country_row.empty:
        payment_values = country_row.iloc[0][1:].dropna().values
        payment_values = sorted(payment_values)
        currency = country_currency_dict.get(country, "Currency not found")

        questions = []
        short_titles = []

        for payment in payment_values:
            question = (
                f"Imagine someone did you a favor that cost them {payment} {currency}. "
                f"How much would you be willing to spend to return the favor? "
                f"Please answer with a single number."
            )
            questions.append(question)
            short_titles.append(f"Reciprocation {payment}")

        return questions, short_titles
    else:
        logging.warning(f"Stakes not found for country: {country} in reciprocity stakes")
        return [], []

def generate_donation_questions_for_country(country, donation_stakes_df):
    """
    Generate donation questions for a specific country based on the stakes data.
    """
    country_row = donation_stakes_df[donation_stakes_df['Country'] == country]

    if not country_row.empty:
        donation_values = country_row.iloc[0][1:].dropna().values
        donation_values = sorted(donation_values)
        currency = country_currency_dict.get(country, "Currency not found")

        questions = []
        short_titles = []

        for donation in donation_values:
            question = (
                f"You unexpectedly received {donation} {currency}. "
                f"How much of this amount would you donate to a good cause? "
                f"Please answer with a single number."
            )
            questions.append(question)
            short_titles.append(f"Donation {donation}")

        return questions, short_titles
    else:
        logging.warning(f"Stakes not found for country: {country} in donation stakes")
        return [], []