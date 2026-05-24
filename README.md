```markdown
# Auditing and Mitigating Hallucinatory Biases in VLMs

Official code repository for the paper:
**"How Can Hallucinatory Biases Be Effectively Audited and Mitigated in Vision-Language Models?"**  
*A unified theoretical and empirical framework across GPT‑4o, Grok 3, and Claude Sonnet 4.5*

[![Paper](https://img.shields.io/badge/paper-submitted-blue)](https://arxiv.org) (link to be updated)

---

## 🧠 Overview

This repository implements the core methods presented in the paper:

- **Multimodal Semantic Entropy (MSE)** – A joint uncertainty metric that captures both linguistic and visual uncertainty to detect hallucinations without ground truth.
- **Adaptive Contrastive Logit Subtraction (ACLS)** – An inference‑time algorithm that provably reduces hallucination rates by subtracting contrastive signals scaled by the estimated hallucination probability.
- **ACLS‑Lite** – A lightweight, zero‑entropy approximation suitable for real‑time applications.
- **Bias Disparity Index (BDI)** – A fairness metric for demographic parity in VLM outputs.

All methods are model‑agnostic and require only API‑level access (response sampling and image encoding), making them applicable to closed‑source systems like GPT‑4o, Grok 3, and Claude.

---

## 📦 Repository Structure

```
.
├── mse_audit.py            # MSE computation: textual & visual entropy
├── acls_mitigation.py      # ACLS and ACLS‑Lite inference‑time mitigation
├── evaluation.py           # Benchmark evaluation (CHAIR, POPE, BDI, efficiency)
├── utils.py                # Helper functions, COCO vocabulary, parsing
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/hallucination-audit-mitigation
   cd hallucination-audit-mitigation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `torch>=2.0`
   - `transformers>=4.40`
   - `pillow`
   - `scikit-learn`
   - `numpy`
   - `scipy`
   - `matplotlib`
   - `tqdm`
   - `accelerate`

3. **Download models (optional)**  
   The code automatically downloads the following models from HuggingFace:
   - VLM: `llava-hf/llava-1.5-7b-hf` (for local experiments)
   - NLI: `microsoft/deberta-v3-large-mnli`
   - CLIP: `openai/clip-vit-base-patch32`
   
   If you use closed‑source APIs (GPT‑4o, Grok, Claude), set your API keys as environment variables (see below).

---

## ⚡ Quick Start

### 1. Compute Multimodal Semantic Entropy (MSE)

```python
from mse_audit import MSEAuditor
from PIL import Image

# Initialize auditor (uses local models by default)
auditor = MSEAuditor()

# Load an image and define a query
image = Image.open("path/to/image.jpg")
query = "Describe this scene."

# Compute MSE
H_ms, H_s, H_v = auditor.compute_mse(image, query)
print(f"Multimodal Semantic Entropy: {H_ms:.3f}")
print(f"Textual Entropy (H_s): {H_s:.3f}")
print(f"Visual Entropy (H_v): {H_v:.3f}")

# Interpret: H_ms > tau (e.g., 2.1) suggests likely hallucination
```

### 2. Apply ACLS Mitigation (Logit Level)

```python
from mse_audit import MSEAuditor
from acls_mitigation import adaptive_contrastive_logit_subtraction
import torch

# Set up model and auditor
auditor = MSEAuditor()
model = auditor.vlm
processor = auditor.processor

image = Image.open("example.jpg")
query = "What objects are present?"
# Generate a perturbed image (e.g., Gaussian blur)
perturbed = auditor.perturb_images(image)[0]

# Compute MSE to estimate hallucination probability
H_ms, _, _ = auditor.compute_mse(image, query)

# ACLS logit adjustment
logits_adj = adaptive_contrastive_logit_subtraction(
    model, processor, image, query, perturbed,
    H_ms, alpha=0.5, tau=2.1, return_logits=True
)

# Continue autoregressive generation from adjusted logits...
```

### 3. ACLS‑Lite (Zero‑entropy, fast)

```python
from acls_mitigation import acls_lite

# Pre‑calibrated alpha per model (from paper)
alpha_model = 0.48  # for GPT-4o; 0.42 for Grok 3; 0.35 for Claude 4.5

next_token_id = acls_lite(model, processor, image, query, perturbed, alpha_model)
# Continue autoregressive generation...
```

### 4. Evaluate on Benchmarks

```python
from evaluation import evaluate_on_dataset
from mse_audit import MSEAuditor

auditor = MSEAuditor()
# Provide a dataset in JSON format (see evaluation.py for expected structure)
results = evaluate_on_dataset("data/coco_val_subset.json", auditor, auditor.vlm, auditor.processor)
print(results)
```

---

## 📊 Reproducing Paper Results

### Datasets
- **MS‑COCO** (2017 validation): [coco](https://cocodataset.org/#download)
- **NoCaps**: [nocaps](https://nocaps.org/)
- **MMBench**: [mmbench](https://github.com/open-compass/mmbench)

Prepare JSON annotations with the following keys per sample:
```json
{
  "image_path": "path/to/image.jpg",
  "objects": ["person", "car", ...],         // ground truth objects (for CHAIR)
  "attributes": [...],                        // optional, for CHAIR+
  "category": "spatial"                       // for MMBench analysis
}
```

### Evaluation Script

```bash
python evaluation.py \
    --dataset path/to/annotations.json \
    --model_name gpt-4o \          # or grok-3, claude-4.5, llava
    --output_dir results/
```

The script outputs:
- `chair_scores.csv` – CHAIR$_s$ values per model & condition
- `pope_results.csv` – Precision/Recall/F1 for POPE
- `bdi_report.csv` – Bias Disparity Index per demographic group
- `latency_profile.csv` – Efficiency breakdown

### Human Evaluation Data
The 200‑sample human evaluation annotations are provided in `data/human_eval.csv`. We computed Fleiss’ kappa and confusion matrices against automatic metrics using the script `analysis/human_agreement.py`.

---

## 📈 Key Results (from paper)

| Model              | CHAIR$_s$ Baseline | CHAIR$_s$ +ACLS | BDI Reduction | ACLS‑Lite Latency |
|--------------------|--------------------|-----------------|---------------|-------------------|
| GPT‑4o             | 0.148 ±0.012       | 0.094 ±0.008    | 39%           | 97 ms             |
| Grok 3             | 0.112 ±0.010       | 0.069 ±0.007    | 41%           | –                 |
| Claude Sonnet 4.5  | 0.073 ±0.006       | 0.042 ±0.004    | 44%           | –                 |

All improvements statistically significant at $p<0.001$ (paired t‑test with Bonferroni).

---

## 🧪 Sensitivity & Robustness

We provide scripts to replicate the sensitivity analysis (Figure 3 in paper):
```bash
python analysis/sensitivity_nli.py    # NLI error simulation
python analysis/sensitivity_noise.py  # Visual perturbation noise
```

These generate AUROC curves under varying NLI error rates and perturbation noise levels, demonstrating graceful degradation.

---

## 📝 Citation

If you use this code or data, please cite:

```bibtex
@article{ghajari2026hallucination,
  title={How Can Hallucinatory Biases Be Effectively Audited and Mitigated in Vision-Language Models?},
  author={Ghajari, Amirali},
  journal={Iran Journal of Computer Science},
  year={2026},
  note={Submitted}
}
```

---

## 📧 Contact & Support

- Author: Amirali Ghajari – amirali.ghajari276@gmail.com
- Repository issues: [GitHub Issues](https://github.com/your-org/hallucination-audit-mitigation/issues)

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

**We hope this repository accelerates the development of fair and reliable Vision‑Language Models.**
