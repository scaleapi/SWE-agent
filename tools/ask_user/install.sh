#!/bin/bash
# Install dependencies for ask_user tool

pip install -q -r "$(dirname "$0")/requirements.txt"
echo "ask_user tool dependencies installed"
