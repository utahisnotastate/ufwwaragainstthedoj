import logging
import yaml

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

class CodeForge:
    """
    Component 2: The Code-Forge.
    Uses Vertex AI Gemini to convert the Topology Blueprint into 'Digital Matter' (full source code).
    """
    def __init__(self):
        if not config.OFFLINE_MODE and HAS_VERTEX:
            if not config.GCP_PROJECT:
                raise ValueError("GCP_PROJECT is not set; required to use Vertex AI.")
            vertexai.init(project=config.GCP_PROJECT, location=config.VERTEX_LOCATION)  # type: ignore
            self.model = GenerativeModel(config.GENESIS_MODEL)  # type: ignore
            logging.info(f"Code-Forge initialized with Vertex AI Gemini model '{config.GENESIS_MODEL}'.")
        else:
            self.model = None
            if not HAS_VERTEX:
                logging.info("Vertex AI SDK not installed. Code-Forge will use local template generation.")
            logging.info("Code-Forge initialized in OFFLINE_MODE — cloud LLM disabled.")

    def _generate_local_templates(self, topology_blueprint: dict) -> dict:
        """Generate minimal, runnable PyTorch training suite without internet.
        Uses torchvision FakeData if available, else scikit-learn synthetic data.
        """
        model_py = '''import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features=3*32*32, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)
'''
        dataset_py = '''import torch
from torch.utils.data import Dataset, TensorDataset

# Prefer torchvision FakeData (offline), else fallback to sklearn synthetic
try:
    from torchvision.datasets import FakeData
    from torchvision import transforms
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

class OfflineDataset:
    @staticmethod
    def get_datasets(num_samples=2000, image_size=(3, 32, 32), num_classes=10):
        if HAS_TORCHVISION:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train = FakeData(size=num_samples, image_size=image_size, num_classes=num_classes, transform=transform)
            val = FakeData(size=int(num_samples*0.2), image_size=image_size, num_classes=num_classes, transform=transform)
            return train, val
        else:
            try:
                from sklearn.datasets import make_classification
                import numpy as np
                X, y = make_classification(n_samples=num_samples, n_features=32*32*3, n_informative=50, n_classes=num_classes, random_state=42)
                X = X.astype('float32') / X.max()
                X = X.reshape(-1, 3, 32, 32)
                y = y.astype('int64')
                train = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
                # simple split
                n_val = int(0.2 * len(train))
                val = TensorDataset(train.tensors[0][:n_val], train.tensors[1][:n_val])
                train = TensorDataset(train.tensors[0][n_val:], train.tensors[1][n_val:])
                return train, val
            except Exception as e:
                raise RuntimeError(f"Neither torchvision nor scikit-learn available for dataset: {e}")
'''
        train_py = '''import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import SimpleNet
from dataset import OfflineDataset

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds, val_ds = OfflineDataset.get_datasets(num_samples=args.samples, num_classes=args.num_classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = SimpleNet(num_classes=args.num_classes).to(device)
    opt = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0
        correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)
        train_acc = correct/total if total else 0.0
        scheduler.step()

        # simple val
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                correct += (preds==y).sum().item()
                total += y.size(0)
        val_acc = correct/total if total else 0.0
        print(f"[epoch {epoch}] train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'model.pth')
    torch.save({'model_state': model.state_dict()}, ckpt_path)
    print(f"Saved model to {ckpt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--num_classes', type=int, default=10)
    args = p.parse_args()
    train(args)

if __name__ == '__main__':
    main()
'''
        return {
            'model.py': model_py,
            'dataset.py': dataset_py,
            'train.py': train_py,
        }

    def generate_digital_matter(self, topology_blueprint: dict) -> dict:
        """
        Generates a package of Python scripts based on the blueprint.

        Args:
            topology_blueprint: The structured plan from the Conceptualizer.

        Returns:
            A dictionary of filenames and their source code content.
        """
        if config.OFFLINE_MODE or self.model is None:
            logging.info("OFFLINE_MODE active — generating local template code package.")
            return self._generate_local_templates(topology_blueprint)

        blueprint_str = yaml.dump(topology_blueprint)
        framework = topology_blueprint.get("framework", "PyTorch").lower()

        prompt = f"""SYSTEM INSTRUCTION: You are an elite AI engineer specializing in {framework}.
        Your task is to generate three complete, production-ready Python scripts based on the provided 'Topology Blueprint'.
        
        TOPOLOGY BLUEPRINT:
        ---
        {blueprint_str}
        ---
        
        CRITICAL REQUIREMENTS:
        1.  Generate three separate, complete Python files: `model.py`, `dataset.py`, and `train.py`.
        2.  The code must be fully functional. NO 'pass', 'TODO', or placeholder comments.
        3.  The framework is {framework}. Use idiomatic code and best practices.
        4.  `train.py` MUST be a command-line script using `argparse`. It must be executable on Google Cloud Vertex AI Custom Training. It must accept arguments for all hyperparameters specified in the blueprint and an `--output-dir` for saving the model.
        5.  `model.py` must contain the complete nn.Module (PyTorch) or equivalent model definition.
        6.  `dataset.py` must define a PyTorch Dataset class (or tf.data.Dataset) with a placeholder for data loading, but with all boilerplate and transformations implemented.
        7.  Wrap each file's code in a unique markdown block, e.g.:
            python:model.py
            # ... code for model.py ...
            
            python:dataset.py
            # ... code for dataset.py ...
            
            python:train.py
            # ... code for train.py ...
            
        8.  The code must be self-contained within these three files. Assume standard libraries (`torch`, `torchvision`, `numpy`, etc.) are installed.
        """

        logging.info("Sending code generation request to Vertex AI Gemini...")
        response = self.model.generate_content(prompt)  # type: ignore

        code_package = {}
        raw_text = response.text
        files_to_find = ["model.py", "dataset.py", "train.py"]

        for filename in files_to_find:
            try:
                start_marker = f"python:{filename}"
                end_marker = ""
                start_index = raw_text.find(start_marker)
                if start_index == -1:
                    raise ValueError(f"Marker for {filename} not found.")

                start_index += len(start_marker)
                end_index = raw_text.find(end_marker, start_index)
                if end_index == -1:
                    raise ValueError(f"End marker for {filename} not found.")

                code = raw_text[start_index:end_index].strip()
                code_package[filename] = code
            except Exception as e:
                logging.error(f"Failed to parse '{filename}' from LLM response: {e}")
                logging.error(f"RAW RESPONSE: {raw_text}")
                raise ValueError(f"Could not forge {filename} from LLM response.")

        if len(code_package) != 3:
            raise ValueError("Code-Forge failed to generate all three required files.")

        return code_package
