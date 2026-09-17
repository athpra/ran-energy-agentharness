#!/bin/bash
# Runs once when the CML project is built.
# Installs Python dependencies and the VIAVI ADK package.

set -e

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Install VIAVI ADK from the RSG server (requires RSG_HOST to be set)
if [ -n "$RSG_HOST" ]; then
    echo "Installing VIAVI ADK from http://${RSG_HOST}:8000/adk ..."
    pip install --quiet "http://${RSG_HOST}:8000/adk"
    echo "VIAVI ADK installed."
else
    echo "WARNING: RSG_HOST is not set — skipping VIAVI ADK install."
    echo "Set RSG_HOST as a CML project environment variable and re-run:"
    echo "  pip install http://<rsg-host>:8000/adk"
fi

echo ""
echo "Setup complete. Required project environment variables:"
echo "  RSG_HOST      — IP or hostname of the VIAVI AI RSG server"
echo "  CDSW_API_URL  — CML model serving base URL"
echo "  CDSW_API_KEY  — API key (or leave blank to use JWT auto-refresh)"
echo "  LLM_MODEL     — model name as registered in CML AI Inference"
echo "  AUTH_MODE     — jwt (CML workbench) or api_key (external)"
