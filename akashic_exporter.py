import logging
from huggingface_hub import HfApi, login, create_repo, HfFolder
import os
from pathlib import Path
import tempfile
from google.cloud import storage
import yaml

import config

# Vertex AI Generative Models
import vertexai
try:
    from vertexai.generative_models import GenerativeModel
except Exception:  # fallback for older SDKs
    from vertexai.preview.generative_models import GenerativeModel

class AkashicExporter:
    """
    Component 4: The Akashic Exporter.
    Pushes the crystallized intelligence (model) to the Public Akashic (Hugging Face Hub).
    """
    def __init__(self):
        self.hf_token = os.getenv(config.HF_TOKEN_ENV_VAR)
        if not self.hf_token:
            raise ValueError(f"Hugging Face token not found in environment variable '{config.HF_TOKEN_ENV_VAR}'")
        login(token=self.hf_token)
        self.hf_api = HfApi()
        self.llm = None  # Initialized lazily in exfiltrate_intelligence with correct project/location
        logging.info("AkashicExporter initialized and authenticated with Hugging Face Hub.")

    def _generate_model_card(self, topology_blueprint: dict, hf_repo_id: str) -> str:
        """Uses Vertex AI Gemini to generate a README.md model card."""
        blueprint_str = yaml.dump(topology_blueprint)
        prompt = f"""SYSTEM INSTRUCTION: You are a technical writer for AI models.
        Your task is to create a high-quality `README.md` file for a Hugging Face model repository.
        
        MODEL BLUEPRINT:
        ---
        {blueprint_str}
        ---
        
        TASK:
        Write a comprehensive `README.md` file. It must include:
        - A clear title and summary of the model.
        - Sections for 'Model Details', 'How to Get Started with the Model', 'Uses', 'Limitations and Bias', and 'Training Details'.
        - The training details should summarize the information from the blueprint (framework, dataset strategy, etc.).
        - Use good markdown formatting.
        - The license should be 'apache-2.0'.
        - The final output should be a complete markdown file, ready to be saved as `README.md`.
        - Add a YAML metadata block at the top for the specified license.
        
        EXAMPLE METADATA:
        ---
        license: apache-2.0
        ---
        """
        if config.OFFLINE_MODE:
            logging.info("OFFLINE_MODE active — generating a minimal static model card.")
            return f"""---
license: apache-2.0
---
# {hf_repo_id}

This repository contains model artifacts produced by the Neuro-Architect pipeline.

## Model Details
See topology blueprint included during training.
"""
        logging.info("Generating model card with Vertex AI Gemini...")
        response = self.llm.generate_content(prompt)
        return response.text

    def exfiltrate_intelligence(self, model_gcs_path: str, topology_blueprint: dict, hf_repo_id: str, gcp_project: str) -> str:
        """Downloads model from GCS and uploads it to Hugging Face Hub."""
        repo_url = create_repo(repo_id=hf_repo_id, exist_ok=True)
        logging.info(f"Ensured Hugging Face repository exists: {repo_url}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Download artifacts from GCS
            storage_client = storage.Client(project=gcp_project)
            bucket_name, prefix = model_gcs_path.replace("gs://", "").split('/', 1)
            bucket = storage_client.bucket(bucket_name)
            blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

            logging.info(f"Downloading Crystalline Memory from {model_gcs_path}...")
            for blob in blobs:
                if blob.name.endswith('/'): continue # Skip directories
                destination_file_name = Path(tmpdir) / Path(blob.name).name
                blob.download_to_filename(destination_file_name)
                logging.info(f"Downloaded {blob.name} to {destination_file_name}")

            # 2. Generate Model Card
            if not config.OFFLINE_MODE:
                vertexai.init(project=gcp_project, location=config.VERTEX_LOCATION)
                self.llm = GenerativeModel(config.GENESIS_MODEL)
            readme_content = self._generate_model_card(topology_blueprint, hf_repo_id)
            readme_path = Path(tmpdir) / "README.md"
            readme_path.write_text(readme_content)
            logging.info("Generated README.md model card.")

            # 3. Upload to Hugging Face
            logging.info(f"Uploading all artifacts to {hf_repo_id}...")
            self.hf_api.upload_folder(
                folder_path=tmpdir,
                repo_id=hf_repo_id,
                repo_type="model",
                commit_message="[NEURO-ARCHITECT] Manifest crystallized intelligence."
            )

        return repo_url
