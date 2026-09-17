#!/bin/bash
# Runs once when the CML project is built.
# Installs Python dependencies into the project environment.

set -e

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "Dependencies installed."
echo ""
echo "NOTE: The VIAVI AI RSG SDK (viavi.rsg) must be available in your CML"
echo "runtime image or installed separately:"
echo "  pip install /path/to/viavi_rsg-*.whl"
echo ""
echo "Set the following project-level environment variables before running:"
echo "  CDSW_API_URL  — CML model serving base URL"
echo "  CDSW_API_KEY  — API key (or leave blank to use JWT auto-refresh)"
echo "  LLM_MODEL     — model name as registered in CML AI Inference"
echo "  AUTH_MODE     — jwt (CML workbench) or api_key (external)"
