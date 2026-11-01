"""
Shared prediction code for finalize.py (validation) and final_test.py (test), so the
final choice is scored on test with exactly the same code it was chosen with on validation.

Supports ensembles (average of several models' probabilities) and test-time augmentation
(average over the 8 rotations/flips of each image, the same symmetries used in training).
"""
import numpy as np
import torch
import torch.nn.functional as F

from gotrain_tools import task_kind

DIHEDRAL = [(k, flip) for flip in (False, True) for k in range(4)]  # 4 rotations x 2 flips
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _module(task):
    if task_kind(task) == "classification":
        import baseline_classification as m
    else:
        import baseline_segmentation as m
    return m


def load_data(task: str, split: str, img_size: int):
    """Images and labels for a split. Validation uses the training cache; test is read fresh."""
    m = _module(task)
    t = m.get_task(task)
    if split == "valid":
        x, y, _ = m.cached_split(t, "valid", img_size)
    else:
        x, y, _ = m.load_split(t, split, img_size)
    return x, y


def load_model(task: str, path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if task_kind(task) == "classification":
        import timm
        model = timm.create_model(cfg["backbone"], pretrained=False,
                                  in_chans=ckpt["in_chans"], num_classes=ckpt["n_classes"])
    else:
        import segmentation_models_pytorch as smp
        model = getattr(smp, cfg["arch"])(encoder_name=cfg["encoder"], encoder_weights=None,
                                          in_channels=ckpt["in_chans"], classes=1)
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE).eval(), ckpt


@torch.no_grad()
def predict(task: str, model, ckpt, x: np.ndarray, tta: bool, batch_size: int = 32) -> np.ndarray:
    """Class probabilities (N, C) or foreground probabilities (N, H, W)."""
    seg = task_kind(task) == "segmentation"
    mean = torch.from_numpy(ckpt["mean"]).to(DEVICE)
    std = torch.from_numpy(ckpt["std"]).to(DEVICE)
    transforms = DIHEDRAL if tta else [(0, False)]
    out = []
    for i in range(0, len(x), batch_size):
        xb = (torch.from_numpy(np.asarray(x[i:i + batch_size])).to(DEVICE).float() - mean) / std
        acc = 0
        for k, flip in transforms:
            xt = torch.rot90(xb, k, (-2, -1))
            if flip:
                xt = xt.flip(-1)
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                logits = model(xt).float()
            if seg:
                p = torch.sigmoid(logits)
                if flip:  # undo the transform so pixels line up again
                    p = p.flip(-1)
                p = torch.rot90(p, -k, (-2, -1))[:, 0]
            else:
                p = logits.softmax(1)
            acc = acc + p
        out.append((acc / len(transforms)).cpu().numpy())
    return np.concatenate(out)


def resize_probs(p: np.ndarray, size: int) -> np.ndarray:
    """Resize (N, H, W) segmentation probabilities so models with different img_size can be averaged."""
    if p.shape[-1] == size:
        return p
    t = torch.from_numpy(p)[:, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()


def score(task: str, probs: np.ndarray, y: np.ndarray) -> dict:
    if task_kind(task) == "classification":
        return {"metric": "accuracy", "score": float((probs.argmax(1) == y).mean())}
    pred, gt = probs > 0.5, y.astype(bool)
    iou = []
    for p, g in [(~pred, ~gt), (pred, gt)]:  # background, foreground
        iou.append((p & g).sum() / max((p | g).sum(), 1))
    return {"metric": "mIoU", "score": float(np.mean(iou)), "iou_fg": float(iou[1])}


def ensemble_probs(task: str, members: list[dict], split: str, tta: bool):
    """Average probabilities of several models on a split. Each member keeps its own img_size
    and normalisation; segmentation is scored at the first (best) member's img_size."""
    seg = task_kind(task) == "segmentation"
    ref_size = members[0]["img_size"]
    _, y = load_data(task, split, ref_size)
    total = None
    for mem in members:
        x, _ = load_data(task, split, mem["img_size"])
        model, ckpt = load_model(task, mem["model_path"])
        p = predict(task, model, ckpt, x, tta)
        if seg:
            p = resize_probs(p, ref_size)
        total = p if total is None else total + p
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return total / len(members), y
