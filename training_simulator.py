import logging
import os
import time
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path

import config

# Try to import GCP SDKs lazily; allow module to load even if unavailable
try:
    from google.cloud import aiplatform, storage
    HAS_GCP = True
except Exception:
    aiplatform = None
    storage = None
    HAS_GCP = False


class TrainingSimulator:
    """
    Component 3: The Training Simulator.
    Interfaces with GCP Vertex AI to 'crystallize' model weights.
    """
    def __init__(self, project: str, location: str, staging_bucket: str):
        if config.OFFLINE_MODE:
            raise RuntimeError("OFFLINE_MODE=1 is set. Cloud calls to Vertex AI are disabled.")
        if not HAS_GCP:
            raise ImportError("google-cloud-aiplatform and google-cloud-storage are required for cloud training. Install them or use OFFLINE mode.")
        self.project = project
        self.location = location
        self.staging_bucket_name = staging_bucket.replace("gs://", "")
        aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)
        self.storage_client = storage.Client(project=project)
        logging.info(f"TrainingSimulator initialized for GCP Project '{project}' in '{location}'.")

    def _stage_source_code(self, job_display_name: str, source_code_package: dict) -> str:
        """Uploads the generated source code to GCS for the training job."""
        bucket = self.storage_client.bucket(self.staging_bucket_name)
        gcs_source_path = f"neuro_architect_source/{job_display_name}/{int(time.time())}"

        for filename, code in source_code_package.items():
            blob_path = f"{gcs_source_path}/{filename}"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(code, content_type="text/plain")
            logging.info(f"Staged {filename} to gs://{self.staging_bucket_name}/{blob_path}")

        # For custom containers, we'd also upload a Dockerfile and build it.
        # For this MVP, we use a pre-built container and specify the main python module.
        return gcs_source_path

    def crystallize_memory(self, job_display_name: str, source_code_package: dict) -> str:
        """
        Submits a custom training job to Vertex AI and waits for completion.

        Returns:
            The GCS path to the final model artifacts.
        """
        gcs_source_path = self._stage_source_code(job_display_name, source_code_package)

        # Define the output directory on GCS for the trained model
        final_model_gcs_path = f"gs://{self.staging_bucket_name}/neuro_architect_output/{job_display_name}"

        job = aiplatform.CustomTrainingJob(
            display_name=job_display_name,
            script_path=os.path.join(os.getcwd(), "placeholder_script.py"),  # SDK requires a local path, but we override with GCS source
            container_uri=config.PYTORCH_CONTAINER_URI,  # Using a pre-built container
            requirements=["torchvision", "scikit-learn"],  # Example requirements
            model_serving_container_image_uri=config.PYTORCH_SERVING_CONTAINER_URI,
        )

        logging.info(f"Submitting crystallization job '{job_display_name}' to Vertex AI.")
        model = job.run(
            # This overrides the local script_path with our GCS source code
            base_output_dir=final_model_gcs_path,
            args=["--output-dir", "/gcs/" + final_model_gcs_path.split("gs://")[1]],  # Vertex maps GCS paths to /gcs/
            replica_count=1,
            machine_type=config.DEFAULT_MACHINE_TYPE,
            accelerator_type=config.DEFAULT_ACCELERATOR_TYPE,
            accelerator_count=config.DEFAULT_ACCELERATOR_COUNT,
            # We point the job to the Python module on GCS
            python_package_gcs_uri=f"gs://{self.staging_bucket_name}/{gcs_source_path}",
            python_module_name="train.main",  # Assumes train.py has a main function
        )

        logging.info("Wave Function Collapse... Model training in progress. Waiting for completion.")
        # The job.run() call is synchronous and waits for completion.

        if model.state != aiplatform.models.Model.State.SUCCEEDED:
            logging.error(f"Training job failed with state: {model.state}")
            raise RuntimeError("Vertex AI training job did not succeed.")

        return final_model_gcs_path

    def evolutionary_feedback_loop(self, job_display_name: str, user_feedback: str):
        """Placeholder for the evolutionary feedback loop."""
        logging.info("Evolutionary Feedback Loop initiated.")
        logging.info(f"User Feedback: '{user_feedback}'")
        logging.info("Detecting instability... Formulating prompt for model refinement... Re-initiating crystallization.")
        # In a real implementation, this would:
        # 1. Formulate a new prompt for the Code-Forge based on the feedback.
        # 2. Call code_forge.generate_digital_matter() again.
        # 3. Call self.crystallize_memory() with the new code package.
        pass


class LocalTrainingSimulator:
    """
    Offline local training simulator.
    Writes provided source code to a temp workspace and runs train.py locally.
    Saves artifacts under a local 'artifacts' directory.
    """
    def __init__(self, base_artifacts_dir: str | None = None):
        self.base_artifacts_dir = base_artifacts_dir or os.getenv("NA_ARTIFACTS_DIR", "artifacts")
        Path(self.base_artifacts_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"LocalTrainingSimulator ready. Artifacts base: {self.base_artifacts_dir}")

    def crystallize_memory(self, job_display_name: str, source_code_package: dict, extra_args: list[str] | None = None) -> str:
        workspace = Path(tempfile.mkdtemp(prefix=f"na_{job_display_name}_"))
        logging.info(f"Writing digital matter to local workspace: {workspace}")
        for filename, code in source_code_package.items():
            (workspace / filename).write_text(code)

        output_dir = Path(self.base_artifacts_dir) / job_display_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command to run train.py
        cmd = [sys.executable, str(workspace / "train.py"), "--output-dir", str(output_dir)]
        if extra_args:
            cmd.extend(extra_args)

        logging.info(f"Executing local training: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Local training process failed: {e}")
            raise

        logging.info(f"Local crystallization complete. Artifacts at: {output_dir}")
        return str(output_dir)
