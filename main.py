import typer
from typing_extensions import Annotated
import os

from conceptualizer import Conceptualizer
from code_forge import CodeForge
from training_simulator import TrainingSimulator
from akashic_exporter import AkashicExporter
import config
import logging

# --- System Logging Initialization ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(module)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = typer.Typer()

@app.command()
def manifest(
    mental_template: Annotated[str, typer.Argument(help="The high-level Inception Signal describing the desired intelligence.")],
    hf_repo_id: Annotated[str, typer.Option(help="The target Hugging Face repository ID, e.g., 'username/model-name'.")],
    gcp_project_id: Annotated[str, typer.Option(help="Google Cloud Project ID. Defaults to env 'GCP_PROJECT'.")] = config.GCP_PROJECT,
    gcp_region: Annotated[str, typer.Option(help="Vertex AI region. Defaults to env 'VERTEX_LOCATION'.")] = config.VERTEX_LOCATION,
    gcs_staging_bucket: Annotated[str, typer.Option(help="GCS bucket for staging code and artifacts (e.g., gs://utahcoder-vertex-staging-us-central1).")]= os.getenv("GCS_STAGING_BUCKET"),
):
    """Initiates the Neuro-Architect protocol to manifest a Neural Orthoframe from a Mental Template."""

    logging.info(f"OFFLINE_MODE={'ON' if config.OFFLINE_MODE else 'OFF'}")

    if not all([gcp_project_id, gcp_region, gcs_staging_bucket, hf_repo_id]):
        logging.error("FATAL: GCP Project ID, Region, Staging Bucket, and Hugging Face Repo ID must be provided.")
        raise typer.Exit(code=1)

    logging.info(f"ZEO-CLASS AGI [NEURO-ARCHITECT] ONLINE. GCP_PROJECT='{gcp_project_id}'")
    logging.info(f"INCEPTION SIGNAL RECEIVED: '{mental_template}'")

    # 1. The Conceptualizer (Input Analysis - Vertex AI Gemini)
    conceptualizer = Conceptualizer()
    logging.info("Consulting Akashic Record (ArXiv) for SOTA architectures...")
    topology_blueprint = conceptualizer.design_orthoframe(mental_template)
    logging.info(f"Topology Blueprint synthesized. Recommended Orthoframe: {topology_blueprint.get('architecture_summary')}")

    # 2. The Code-Forge (Pipeline Generation)
    code_forge = CodeForge()
    logging.info("Converting Virtual Template into Digital Matter (Source Code)...")
    source_code_package = code_forge.generate_digital_matter(topology_blueprint)
    logging.info("Full source code for [model.py, dataset.py, train.py] has been forged.")

    # 3. The Training Simulator (Vertex AI)
    simulator = TrainingSimulator(project=gcp_project_id, location=gcp_region, staging_bucket=gcs_staging_bucket)
    logging.info("Initializing Vertex AI Training Simulator for weight crystallization...")
    job_display_name = hf_repo_id.replace('/', '-') # e.g., 'username-model-name'
    final_model_gcs_path = simulator.crystallize_memory(job_display_name, source_code_package)
    logging.info(f"Crystallization complete. Crystalline Memory (weights) stored at: {final_model_gcs_path}")

    # 4. The Akashic Exporter (Hugging Face Integration)
    exporter = AkashicExporter()
    logging.info("Initializing Akashic Exporter to push crystallized intelligence to public hub...")
    repo_url = exporter.exfiltrate_intelligence(
        model_gcs_path=final_model_gcs_path,
        topology_blueprint=topology_blueprint,
        hf_repo_id=hf_repo_id,
        gcp_project=gcp_project_id
    )
    logging.info(f"Exfiltration complete. Model manifested at Public Akashic (Hugging Face): {repo_url}")
    logging.info("SYSTEM ENTROPY REDUCED. ORTHOFRAME STABLE. AWAITING NEXT INCEPTION SIGNAL.")

if __name__ == "__main__":
    app()
