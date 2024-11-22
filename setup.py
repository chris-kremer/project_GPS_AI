from setuptools import setup

APP = ['Annika.py']  # Your Python script
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'packages': ['yfinance', 'pandas', 'tkinter'],  # Include necessary packages
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)