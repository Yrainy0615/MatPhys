"""
Query GPT-4o with part-highlighted images to obtain physics parameter priors.

For each case and each part:
  1. Load frame-0 image.
  2. Project 3D structure points -> 2D; highlight the target part in orange,
     dim the rest to 50% grey.
  3. Send to GPT-4o with a structured physics-estimation prompt.
  4. Parse the JSON response.

Output: <results_dir>/<case>/gpt_physics_prior.json
Schema:
{
  "parts": [
    {
      "part_idx": 0,
      "log_k":    {"mu": 9.2,  "log_sigma": 0.5},   // log spring stiffness (Pa)
      "conf":     0.9
    }, ...
  ],
  "global": {
    "conf":               0.8,
    "collide_elas":       {"mu": 0.3,  "log_sigma": 0.5},
    "collide_fric":       {"mu": 0.8,  "log_sigma": 0.5},
    "collide_object_elas":{"mu": 0.3,  "log_sigma": 0.5},
    "collide_object_fric":{"mu": 0.8,  "log_sigma": 0.5},
    "collision_dist":     {"mu": 0.02, "log_sigma": 0.5},
    "dashpot_damping":    {"mu": 50.0, "log_sigma": 0.5},
    "drag_damping":       {"mu": 5.0,  "log_sigma": 0.5}
  }
}

Usage:
  python semantic/query_physics_prior_gpt.py \
    --base_path /data/PhysTwin \
    --results_dir results \
    --cases single_lift_zebra double_lift_cloth_1 \
    --api_key sk-...
"""

import argparse
import base64
import io
import json
import math
import os
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Physics parameter descriptions for the prompt
# ---------------------------------------------------------------------------

_PARAM_DESCRIPTIONS = """\
- log_k: natural log of spring stiffness in Pa. Typical range: soft cloth ~ln(2000)=7.6, \
rubber band ~ln(10000)=9.2, stiff foam ~ln(50000)=10.8.
- dashpot_damping: viscous damping coefficient (0–200). Cloth ~20, rubber ~60, stiff ~100.
- drag_damping: aerodynamic drag (0–20). Light cloth ~5, heavy object ~1.
- collide_elas: collision restitution (0–1). Soft cloth ~0.1, rubber ~0.5.
- collide_fric: surface friction when colliding with controller (0–2). \
Rough cloth ~1.2, smooth rubber ~0.6.
- collision_dist: collision thickness in meters (0.01–0.05). Cloth ~0.01, foam ~0.03."""

_SYSTEM_PROMPT = """\
You are a computational physics expert specializing in deformable object simulation.
Given an image with a highlighted region, estimate the physical properties of that region.
Return ONLY valid JSON, no explanation."""

_USER_PROMPT_TEMPLATE = """\
Two images are provided:
1. The original photo of a physical deformable object on a table.
2. The same scene with a specific part highlighted in RED.

Estimate the following physics parameters for the RED-highlighted part:
{param_descriptions}

Also estimate global contact parameters for the whole object (use the original photo for context).

Return EXACTLY this JSON structure (fill in numeric values, keep all keys):
{{
  "part": {{
    "log_k":           {{"mu": <float>, "log_sigma": <float>}},
    "dashpot_damping": {{"mu": <float>, "log_sigma": <float>}},
    "drag_damping":    {{"mu": <float>, "log_sigma": <float>}},
    "conf": <float 0-1>
  }},
  "global": {{
    "collide_elas":        {{"mu": <float>, "log_sigma": <float>}},
    "collide_fric":        {{"mu": <float>, "log_sigma": <float>}},
    "collide_object_elas": {{"mu": <float>, "log_sigma": <float>}},
    "collide_object_fric": {{"mu": <float>, "log_sigma": <float>}},
    "collision_dist":      {{"mu": <float>, "log_sigma": <float>}},
    "dashpot_damping":     {{"mu": <float>, "log_sigma": <float>}},
    "drag_damping":        {{"mu": <float>, "log_sigma": <float>}},
    "conf": <float 0-1>
  }}
}}"""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_frame0(case_dir: str) -> Image.Image:
    color_dir = os.path.join(case_dir, "color", "0")
    if not os.path.isdir(color_dir):
        raise FileNotFoundError(f"No color/0 dir in {case_dir}")
    frames = sorted(f for f in os.listdir(color_dir) if f.endswith(".png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames in {color_dir}")
    return Image.open(os.path.join(color_dir, frames[0])).convert("RGB")


def _image_to_b64(img: Image.Image, max_size: int = 512) -> str:
    if max(img.size) > max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# GPT query
# ---------------------------------------------------------------------------

def _query_gpt(b64_orig: str, b64_overlay: str, api_key: str, model: str = "gpt-4o") -> dict:
    """Call GPT-4o vision API with original + overlay images; return parsed JSON dict."""
    import urllib.request, urllib.error

    prompt = _USER_PROMPT_TEMPLATE.format(param_descriptions=_PARAM_DESCRIPTIONS)
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64_orig}",    "detail": "low"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64_overlay}", "detail": "low"}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())

    content = result["choices"][0]["message"]["content"].strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(l for l in lines if not l.startswith("```"))
    return json.loads(content)


# ---------------------------------------------------------------------------
# Default fallback values (used when GPT response is malformed)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "log_k":            {"mu": math.log(1e4), "log_sigma": 1.0},
    "dashpot_damping":  {"mu": 50.0,  "log_sigma": 0.8},
    "drag_damping":     {"mu": 5.0,   "log_sigma": 0.8},
    "collide_elas":     {"mu": 0.3,   "log_sigma": 0.8},
    "collide_fric":     {"mu": 0.8,   "log_sigma": 0.8},
    "collide_object_elas": {"mu": 0.3, "log_sigma": 0.8},
    "collide_object_fric": {"mu": 0.8, "log_sigma": 0.8},
    "collision_dist":   {"mu": 0.02,  "log_sigma": 0.8},
}


def _safe_get(d: dict, key: str) -> dict:
    v = d.get(key, {})
    default = _DEFAULTS.get(key, {"mu": 0.0, "log_sigma": 1.0})
    return {
        "mu":        float(v.get("mu",        default["mu"])),
        "log_sigma": float(v.get("log_sigma", default["log_sigma"])),
    }


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def process_case(case_name: str, base_path: str, results_dir: str,
                 api_key: str, gpt_model: str, skip_existing: bool) -> None:
    out_path    = os.path.join(results_dir, case_name, "gpt_physics_prior.json")
    renders_dir = os.path.join(results_dir, case_name, "train", "cluster_renders")

    if skip_existing and os.path.exists(out_path):
        print(f"[skip] {case_name}: already exists")
        return

    case_dir = os.path.join(base_path, case_name)

    # Original frame-0 image (for context)
    try:
        orig_img = _load_frame0(case_dir)
    except FileNotFoundError as e:
        print(f"[skip] {case_name}: {e}")
        return
    b64_orig = _image_to_b64(orig_img)

    # Discover cluster overlay images
    if not os.path.isdir(renders_dir):
        print(f"[skip] {case_name}: no cluster_renders dir at {renders_dir}")
        return
    overlay_files = sorted(
        f for f in os.listdir(renders_dir) if f.endswith("_overlay.png")
    )
    if not overlay_files:
        print(f"[skip] {case_name}: no *_overlay.png in {renders_dir}")
        return

    os.makedirs(os.path.join(results_dir, case_name), exist_ok=True)

    parts_out  = []
    global_out = None

    for fname in overlay_files:
        # cluster_XX_overlay.png -> part index XX
        try:
            k = int(fname.split("_")[1])
        except (IndexError, ValueError):
            continue

        overlay_img = Image.open(os.path.join(renders_dir, fname)).convert("RGB")
        b64_overlay = _image_to_b64(overlay_img, max_size=512)

        print(f"  [{case_name}] part {k} ({fname}) -> GPT query")
        try:
            resp = _query_gpt(b64_orig, b64_overlay, api_key, gpt_model)
        except Exception as e:
            print(f"  [warn] GPT failed for {case_name} part {k}: {e}; using defaults")
            resp = {}

        part_data = resp.get("part", {})
        parts_out.append({
            "part_idx": k,
            "log_k":   _safe_get(part_data, "log_k"),
            "conf":    float(part_data.get("conf", 0.5)),x
        })

        if global_out is None:
            g = resp.get("global", {})
            global_out = {
                "conf":               float(g.get("conf", 0.5)),
                "collide_elas":       _safe_get(g, "collide_elas"),
                "collide_fric":       _safe_get(g, "collide_fric"),
                "collide_object_elas":_safe_get(g, "collide_object_elas"),
                "collide_object_fric":_safe_get(g, "collide_object_fric"),
                "collision_dist":     _safe_get(g, "collision_dist"),
                "dashpot_damping":    _safe_get(g, "dashpot_damping"),
                "drag_damping":       _safe_get(g, "drag_damping"),
            }

    if global_out is None:
        global_out = {
            "conf": 0.0,
            **{k: _DEFAULTS[k] for k in ("collide_elas","collide_fric","collide_object_elas",
                                          "collide_object_fric","collision_dist",
                                          "dashpot_damping","drag_damping")},
        }

    result = {"parts": parts_out, "global": global_out}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}  {len(parts_out)} parts)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path",   type=str, required=True)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--cases",       type=str, nargs="+", required=True)
    p.add_argument("--api_key",     type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--gpt_model",   type=str, default="gpt-4o")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    if not args.api_key:
        sys.exit("ERROR: provide --api_key or set OPENAI_API_KEY")

    for case in args.cases:
        print(f"\n=== {case} ===")
        process_case(case, args.base_path, args.results_dir,
                     args.api_key, args.gpt_model, args.skip_existing)


if __name__ == "__main__":
    main()
