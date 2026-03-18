#!/bin/bash
# Simple script to activate virtualenv and run the Streamlit app

source venv/bin/activate
PYTHONPATH=. streamlit run ui/app.py
