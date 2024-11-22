# scripts/prompt_generator.py

import random
from .utils import load_config

def generate_system_prompts(countries, num_samples):
    system_prompts = []
    for country in countries:
        for gender in ["male", "female"]:
            for _ in range(num_samples):
                age = random.randint(18, 80)
                system_prompt = f"You are a {age}-year-old {gender} from {country} participating in an economics experiment."
                system_prompts.append(system_prompt)
    return system_prompts

def extract_age_gender_country(system_prompt):
    import re
    pattern = r"You are a (\d+)-year-old (\w+) from ([\w\s_]+?) participating in an economics experiment\."
    match = re.match(pattern, system_prompt)
    if match:
        age = int(match.group(1))
        gender = match.group(2)
        country = match.group(3).strip().replace('_', ' ')
        return age, gender, country
    else:
        return None, None, None