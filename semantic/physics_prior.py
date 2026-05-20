import base64
import json
import math
import re
from io import BytesIO
from typing import Dict

import torch

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def default_part_physics_prior(num_parts: int) -> Dict[str, torch.Tensor]:
    return {
        "part_phys_prior_mu": torch.full((num_parts, 1), math.log(1.0e4), dtype=torch.float32),
        "part_phys_prior_log_sigma": torch.zeros((num_parts, 1), dtype=torch.float32),
        "part_phys_prior_conf": torch.zeros((num_parts, 1), dtype=torch.float32),
    }


def _extract_json_blob(response_text: str) -> Dict:
    candidates = re.findall(r"\{.*\}", response_text, flags=re.DOTALL)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return json.loads(response_text)


def _post_ollama(prompt: str, images, model_name: str):
    import requests

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_to_base64(img) for img in images],
        "stream": False,
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "")


def query_vlm_for_part_physics_prior(full_image, cluster_image, cluster_id: int, model_name: str = "llava:13b") -> Dict[str, float]:
    prompt = f'''Look at these two images:
1. The full object image
2. A highlighted region (part {cluster_id}) from the same object

Estimate a rough continuous PHYSICS prior for the highlighted part.
Focus on deformation under manipulation, not material class names.

Return ONLY valid JSON in this exact format:
{{
  "logk_mu": <float>,
  "logk_sigma": <float>,
  "confidence": <float>,
  "reason": "<short phrase>"
}}

Definitions:
- logk_mu: natural-log spring stiffness prior for this part.
- logk_sigma: uncertainty on log stiffness. Use larger values when unsure.
- confidence: between 0 and 1.

Use these rough anchors in natural log space:
- very soft / floppy cloth-like region: 7.0 to 8.5
- soft deformable stuffed region: 8.0 to 9.3
- medium fabric / dense plush: 8.8 to 10.0
- stiff reinforced / shape-preserving region: 9.8 to 11.2

Do not output any extra text.'''
    try:
        response_text = _post_ollama(prompt, [full_image, cluster_image], model_name)
        parsed = _extract_json_blob(response_text)
        mu = float(parsed.get("logk_mu", math.log(1.0e4)))
        sigma = float(parsed.get("logk_sigma", 1.0))
        conf = float(parsed.get("confidence", 0.0))
        sigma = max(0.05, min(sigma, 3.0))
        conf = max(0.0, min(conf, 1.0))
        return {"logk_mu": mu, "logk_sigma": sigma, "confidence": conf}
    except Exception:
        return {
            "logk_mu": float(math.log(1.0e4)),
            "logk_sigma": 1.0,
            "confidence": 0.0,
        }


