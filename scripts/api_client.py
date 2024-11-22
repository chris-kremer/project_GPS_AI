# scripts/api_client.py

import asyncio
import aiohttp
import logging
from .utils import load_config

config = load_config()
API_KEY = config['api']['key']
MODEL = config['api']['model']
API_URL = config['api']['url']
MAX_RETRIES = config['settings']['max_retries']

async def ask_economic_question(session, question, system_prompt):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    }
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(API_URL, headers=headers, json=data) as response:
                response_data = await response.json()
                if response.status == 200 and 'choices' in response_data and response_data['choices']:
                    return response_data['choices'][0]['message']['content']
                elif response.status == 429:
                    error_message = response_data.get('error', {}).get('message', 'Rate limit exceeded')
                    logging.warning(f"Rate limit exceeded: {error_message}. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                else:
                    error_message = response_data.get('error', {}).get('message', 'Unknown error')
                    logging.error(f"API request failed: {error_message}")
                    return f"API request failed: {error_message}"
        except aiohttp.ClientError as e:
            logging.error(f"Client error: {e}")
            return "Client error in making API request."
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return "Unexpected error in making API request."
    return "Failed after maximum retries."