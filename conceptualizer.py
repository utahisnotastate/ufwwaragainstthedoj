import logging

import config

# Vertex AI Generative Models (optional)
try:
    import vertexai  # type: ignore
    try:
        from vertexai.generative_models import GenerativeModel  # type: ignore
    except Exception:  # fallback for older SDKs
        from vertexai.preview.generative_models import GenerativeModel  # type: ignore
    HAS_VERTEX = True
except Exception:
    vertexai = None  # type: ignore
    GenerativeModel = None  # type: ignore
    HAS_VERTEX = False

class Conceptualizer:
    """
    Component 1: The Conceptualizer.
    Scans the Akashic Record (ArXiv) and uses Vertex AI Gemini to synthesize a Topology Blueprint.
    """
    def __init__(self):
        if not config.OFFLINE_MODE and HAS_VERTEX:
            if not config.GCP_PROJECT:
                raise ValueError("GCP_PROJECT is not set; required to use Vertex AI.")
            vertexai.init(project=config.GCP_PROJECT, location=config.VERTEX_LOCATION)  # type: ignore
            self.model = GenerativeModel(config.GENESIS_MODEL)  # type: ignore
            logging.info(f"Conceptualizer initialized with Vertex AI Gemini model '{config.GENESIS_MODEL}'.")
        else:
            self.model = None
            if not HAS_VERTEX:
                logging.info("Vertex AI SDK not installed. Conceptualizer will use offline fallback blueprint.")
            logging.info("Conceptualizer initialized in OFFLINE_MODE — using local fallback blueprint.")

    def design_orthoframe(self, mental_template: str) -> dict:
        """
        Takes a user's high-level goal and generates a technical blueprint.

        Args:
            mental_template: The user's request, e.g., "Video from Imagination".

        Returns:
            A dictionary representing the Topology Blueprint.
        """
        # If offline or no model, short-circuit before any optional deps like arxiv
        if config.OFFLINE_MODE or self.model is None:
            logging.info("OFFLINE_MODE active — returning a minimal default blueprint.")
            return {
                "architecture_summary": "Vision Transformer (ViT) baseline",
                "framework": "PyTorch",
                "components": [
                    "Patch Embedding Layer",
                    "Multi-Head Self-Attention Blocks",
                    "MLP Head for Classification"
                ],
                "dataset_strategy": "Use CIFAR-10 with standard augmentations (random crop, flip, color jitter).",
                "training_hyperparameters": {
                    "learning_rate": 0.0003,
                    "batch_size": 128,
                    "optimizer": "AdamW",
                    "epochs": 10,
                    "scheduler": "CosineAnnealingLR"
                }
            }

        # Attempt to gather context from ArXiv, but degrade gracefully on Python 3.13 feedparser/cgi issues
        logging.info(f"Scanning ArXiv for keywords related to: '{mental_template}'")
        context_papers = ""
        try:
            try:
                import arxiv  # lazy import to avoid module import crash on 3.13
            except Exception as ie:
                logging.warning(f"ArXiv module unavailable/incompatible; continuing without it: {ie}")
                raise

            search = arxiv.Search(
                query=mental_template,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = list(search.results())
            if not results:
                logging.warning("No relevant papers found on ArXiv. Proceeding with general knowledge.")
                context_papers = "No specific papers found."
            else:
                context_papers = "\n\n".join([f"Title: {r.title}\nAbstract: {r.summary.replace('$\n$', ' ')}" for r in results])
        except Exception as e:
            logging.error(f"ArXiv search failed or is unavailable: {e}")
            context_papers = "ArXiv integration unavailable or failed."

        prompt = f"""SYSTEM INSTRUCTION: You are a world-class AI Research Architect.
        Your task is to analyze a user's request and the latest SOTA research from ArXiv to create a 'Topology Blueprint' for a new AI model.
        
        USER REQUEST (Mental Template): "{mental_template}"
        
        SOTA RESEARCH CONTEXT (Akashic Record Scan):
        {context_papers}
        
        TASK:
        Synthesize the above information into a 'Topology Blueprint'. The blueprint must be a structured plan for implementation. 
        It must specify:
        1. A concise 'architecture_summary' (e.g., 'Latent Diffusion Transformer with Temporal Attention').
        2. The recommended primary framework ('PyTorch' or 'JAX' or 'TensorFlow').
        3. Key architectural components (e.g., 'U-Net backbone', 'Cross-attention layers', 'Patch-based encoder').
        4. A data synthesis/acquisition strategy ('dataset_strategy').
        5. A plausible set of hyperparameters for initial training ('training_hyperparameters').
        
        OUTPUT a YAML-formatted string containing the blueprint. Do not add any other text.
        
        EXAMPLE YAML OUTPUT:
        architecture_summary: "Example: Vision Transformer (ViT) for image classification."
        framework: "PyTorch"
        components:
          - "Patch Embedding Layer"
          - "Multi-Head Self-Attention Blocks"
          - "MLP Head for Classification"
        dataset_strategy: "Utilize the ImageNet-1k dataset. Apply standard augmentations like random cropping, horizontal flipping, and color jitter."
        training_hyperparameters:
          learning_rate: 0.0003
          batch_size: 256
          optimizer: "AdamW"
          epochs: 100
          scheduler: "CosineAnnealingLR"
        """

        logging.info("Sending blueprint request to Vertex AI Gemini...")
        response = self.model.generate_content(prompt)

        import yaml
        try:
            blueprint_yaml = response.text.strip().replace('yaml', '').replace('', '')
            blueprint = yaml.safe_load(blueprint_yaml)
            return blueprint
        except (yaml.YAMLError, AttributeError) as e:
            logging.error(f"Failed to parse YAML blueprint from Vertex AI Gemini: {e}")
            logging.error(f"RAW RESPONSE: {getattr(response, 'text', '')}")
            raise ValueError("Could not construct Topology Blueprint from LLM response.")
