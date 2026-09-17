"""Load conditioning once and expose its actual input geometry, including old caches."""
from pathlib import Path


def load_conditioning(path, device):
    import torch
    from PIL import Image
    data = torch.load(path, map_location="cpu", weights_only=True)
    tags = data["text_token_tags"]
    latents = data.get("condition_latents", [])
    references = []
    saved = data.get("reference_metadata", [])
    files = data.get("keyframe_files", [])
    for index, latent in enumerate(latents):
        item = dict(saved[index]) if index < len(saved) else {}
        if "original_size" not in item and index < len(files):
            try:
                with Image.open(Path(files[index])) as image:
                    item["original_size"] = list(image.size)
            except (OSError, ValueError):
                item["original_size"] = None
        item.update(index=index + 1, normalized_size=[latent.shape[-1] * 16, latent.shape[-2] * 16],
                    latent_shape=list(latent.shape))
        references.append(item)
    metadata = {"reference_short_edge": data.get("reference_size"),
                "references": references, "size_order": "width,height",
                "prompt_tokens": data["prompt_embeds"].shape[0],
                "text_tokens": int((tags == 1).sum()), "vision_tokens": int((tags == 0).sum())}
    conditions = ((tuple(data["keyframe_anchors"]), [c.to(device, torch.float32) for c in latents])
                  if data.get("keyframe_anchors") else None)
    return data["prompt_embeds"].to(device, torch.bfloat16), tags, conditions, metadata
