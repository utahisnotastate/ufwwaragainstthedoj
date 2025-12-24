# Configuration constants for the Neuro-Architect Engine
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# --- Deployment/Environment Configuration ---
# Primary env vars requested by the user
GCP_PROJECT = os.getenv("GCP_PROJECT") or os.getenv("GCP_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1") or os.getenv("GCP_REGION", "us-central1")
OFFLINE_MODE = str(os.getenv("OFFLINE_MODE", "0")).strip() in {"1", "true", "True", "yes", "YES"}

# Model selection
# AETHER_STUDIO uses GENESIS_MODEL; UTAH_CODER_PRIME uses MODEL_ID
GENESIS_MODEL = os.getenv("GENESIS_MODEL", "gemini-2.5-pro")
MODEL_ID = os.getenv("MODEL_ID", GENESIS_MODEL)

# --- Vertex AI Training Configuration ---
# Using a standard PyTorch 2.1 GPU container from Google Container Registry
PYTORCH_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"
PYTORCH_SERVING_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest"

# Default compute resources. Can be overridden.
DEFAULT_MACHINE_TYPE = "n1-standard-8"
DEFAULT_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
DEFAULT_ACCELERATOR_COUNT = 1

# --- Hugging Face Configuration ---
# Name of the environment variable that holds your Hugging Face token
HF_TOKEN_ENV_VAR = "HF_TOKEN"

# --- Placeholder for main.py SDK requirement ---
# The Vertex AI SDK requires a local script path even if it's not used.
import pathlib
placeholder_file = pathlib.Path("placeholder_script.py")
if not placeholder_file.exists():
    placeholder_file.touch()
