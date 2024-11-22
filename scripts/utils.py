# scripts/utils.py

import yaml
import hashlib
import uuid

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def compute_hash(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def generate_uuid():
    return str(uuid.uuid4())