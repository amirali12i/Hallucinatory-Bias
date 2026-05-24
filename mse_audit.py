"""
Multimodal Semantic Entropy (MSE) auditing module.
Computes H_ms = H_s + lambda * H_v for a given VLM, image, and query.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torch.nn.functional as F

class MSEAuditor:
    def __init__(self, vlm_model_name="llava-hf/llava-1.5-7b-hf",
                 nli_model_name="microsoft/deberta-v3-large-mnli",
                 device="cuda",
                 lambda_weight=0.4,
                 num_text_samples=10,
                 num_visual_perturbations=20,
                 semantic_clusters=5,
                 visual_clusters=5):
        self.device = device
        self.lambda_weight = lambda_weight
        self.N = num_text_samples
        self.M = num_visual_perturbations
        self.K_text = semantic_clusters
        self.K_vis = visual_clusters

        # Load VLM
        self.processor = AutoProcessor.from_pretrained(vlm_model_name)
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_model_name, torch_dtype=torch.float16).to(device)
        # Load NLI model for semantic clustering
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)

        # For visual entropy: use CLIP ViT-B/32 as vision encoder (or reuse VLM's vision encoder)
        from transformers import CLIPModel, CLIPProcessor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def sample_responses(self, image, query, temperature=0.7):
        """Generate N responses from VLM."""
        inputs = self.processor(text=query, images=image, return_tensors="pt").to(self.device)
        responses = []
        for _ in range(self.N):
            with torch.no_grad():
                output = self.vlm.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=temperature)
                text = self.processor.decode(output[0], skip_special_tokens=True)
                responses.append(text)
        return responses

    def semantic_clustering(self, responses):
        """Cluster responses using bi-directional entailment via NLI."""
        n = len(responses)
        adj_matrix = np.eye(n)  # self-entailment
        for i in range(n):
            for j in range(i+1, n):
                # Check if i entails j and j entails i
                entail_ij = self._check_entailment(responses[i], responses[j])
                entail_ji = self._check_entailment(responses[j], responses[i])
                if entail_ij and entail_ji:
                    adj_matrix[i,j] = adj_matrix[j,i] = 1
        # Simple connected components clustering
        from scipy.sparse.csgraph import connected_components
        n_comp, labels = connected_components(adj_matrix, directed=False)
        # Map to K clusters if needed (if fewer, keep as is)
        return labels, n_comp

    def _check_entailment(self, premise, hypothesis):
        """Return True if premise entails hypothesis (NLI label=0)."""
        inputs = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
        # label 0: entailment, 1: neutral, 2: contradiction
        return torch.argmax(probs).item() == 0

    def textual_entropy(self, responses, cluster_labels):
        """Compute H_s from cluster probabilities."""
        n = len(responses)
        unique, counts = np.unique(cluster_labels, return_counts=True)
        probs = counts / n
        return -np.sum(probs * np.log(probs + 1e-9))

    def perturb_images(self, image):
        """Generate M perturbed versions of the image."""
        imgs = []
        for _ in range(self.M):
            # random choice of perturbation type
            choice = np.random.choice(['blur', 'noise', 'occlusion', 'color'])
            img = image.copy()
            if choice == 'blur':
                sigma = np.random.uniform(0.5, 2.0)
                img = img.filter(ImageFilter.GaussianBlur(sigma))
            elif choice == 'noise':
                sigma = np.random.uniform(0.02, 0.1)
                arr = np.array(img).astype(np.float32)
                noise = np.random.randn(*arr.shape) * sigma * 255
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
            elif choice == 'occlusion':
                # draw random rectangle
                w, h = img.size
                x1 = np.random.randint(0, w//2)
                y1 = np.random.randint(0, h//2)
                x2 = np.random.randint(w//2, w)
                y2 = np.random.randint(h//2, h)
                arr = np.array(img)
                arr[y1:y2, x1:x2] = 0
                img = Image.fromarray(arr)
            elif choice == 'color':
                factor = np.random.uniform(0.5, 1.5)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(factor)
            imgs.append(img)
        return imgs

    def visual_entropy(self, image):
        """Compute H_v from perturbations using k-means on CLIP embeddings."""
        perturbed = self.perturb_images(image)
        embeddings = []
        for img in perturbed:
            inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.clip_model.get_image_features(**inputs)
                embeddings.append(emb.cpu().numpy())
        embeddings = np.vstack(embeddings)
        kmeans = KMeans(n_clusters=self.K_vis, random_state=0).fit(embeddings)
        labels = kmeans.labels_
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return -np.sum(probs * np.log(probs + 1e-9))

    def compute_mse(self, image, query):
        responses = self.sample_responses(image, query)
        cluster_labels, _ = self.semantic_clustering(responses)
        H_s = self.textual_entropy(responses, cluster_labels)
        H_v = self.visual_entropy(image)
        H_ms = H_s + self.lambda_weight * H_v
        return H_ms, H_s, H_v
