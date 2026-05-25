"""RunPod serverless handler for MARDM text-to-motion generation.

Loads the AE, MARDM (SiT-XL by default), and the length estimator once at
container start, then runs inference per job.

Input schema (see README and .runpod/tests.json for examples):
    {
        "text_prompt":     str | list[str],   # required (single or batch)
        "motion_length":   int | list[int],   # optional; 0 / null = auto-estimate
        "seed":            int = 3407,
        "cfg":             float = 4.5,
        "time_steps":      int = 18,
        "temperature":     float = 1.0,
        "repeat_times":    int = 1,
        "dataset":         "t2m" | "kit" = "t2m",
        "model_name":      str = "MARDM_SiT_XL",
        "model_arch":      str = "MARDM-SiT-XL",
        "render_video":    bool = False,      # include base64 mp4 in response
        "hard_pseudo_reorder": bool = False
    }
"""

import base64
import io
import os
import random
import tempfile
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import runpod

from models.AE import AE_models
from models.MARDM import MARDM_models
from models.LengthEstimator import LengthEstimator
from utils.motion_process import (
    recover_from_ric,
    plot_3d_motion,
    kit_kinematic_chain,
    t2m_kinematic_chain,
)


CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "./checkpoints")
DATASET_DIR = os.environ.get("DATASET_DIR", "./datasets")
DEFAULT_DATASET = os.environ.get("DEFAULT_DATASET", "t2m")
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "MARDM_SiT_XL")
DEFAULT_MODEL_ARCH = os.environ.get("DEFAULT_MODEL_ARCH", "MARDM-SiT-XL")
DEFAULT_AE_NAME = os.environ.get("DEFAULT_AE_NAME", "AE")
DEFAULT_AE_ARCH = os.environ.get("DEFAULT_AE_ARCH", "AE_Model")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def make_progress_callback(job):
    """Return a throttled on_step callback that emits RunPod progress updates.

    Emits only on step==1, step==total, or every 3rd step to keep the rate
    near ~1Hz (the API polls /status at 1Hz). Any progress_update failure is
    swallowed so a transient RunPod hiccup never crashes the inference job.
    """
    def _on_step(step, total):
        if not (step == 1 or step == total or step % 3 == 0):
            return
        try:
            runpod.serverless.progress_update(job, {
                "schema_version": 1,
                "type": "diffusion_step",
                "step": int(step),
                "total": int(total),
            })
        except Exception as exc:
            print(f"[MARDM] progress_update failed: {exc}")
    return _on_step


def _dataset_geometry(dataset_name: str):
    if dataset_name == "kit":
        return 64, 21, kit_kinematic_chain
    return 67, 22, t2m_kinematic_chain


def _load_stats(dataset_name: str):
    data_root = (
        pjoin(DATASET_DIR, "KIT-ML") if dataset_name == "kit" else pjoin(DATASET_DIR, "HumanML3D")
    )
    mean = np.load(pjoin(data_root, "Mean.npy"))
    std = np.load(pjoin(data_root, "Std.npy"))
    return mean, std


def _load_models(dataset_name: str, model_name: str, model_arch: str,
                 ae_name: str, ae_arch: str):
    dim_pose, _, _ = _dataset_geometry(dataset_name)

    # Load checkpoints straight to the target device when CUDA is available --
    # skips the CPU staging copy that torch.load(..., map_location="cpu")
    # would otherwise do, shaving several seconds off cold start.
    ckpt_device = str(DEVICE) if DEVICE.type == "cuda" else "cpu"

    ae = AE_models[ae_arch](input_width=dim_pose)
    ae_ckpt_file = "latest.tar" if dataset_name == "t2m" else "net_best_fid.tar"
    ae_ckpt = torch.load(
        pjoin(CHECKPOINTS_DIR, dataset_name, ae_name, "model", ae_ckpt_file),
        map_location=ckpt_device,
    )
    ae.load_state_dict(ae_ckpt["ae"])

    mardm = MARDM_models[model_arch](ae_dim=ae.output_emb_width, cond_mode="text")
    mardm_ckpt = torch.load(
        pjoin(CHECKPOINTS_DIR, dataset_name, model_name, "model", "latest.tar"),
        map_location=ckpt_device,
    )
    missing, unexpected = mardm.load_state_dict(mardm_ckpt["ema_mardm"], strict=False)
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    assert all(k.startswith("clip_model.") for k in missing), f"Missing non-CLIP keys: {missing}"

    length_estimator = LengthEstimator(512, 50)
    len_ckpt = torch.load(
        pjoin(CHECKPOINTS_DIR, dataset_name, "length_estimator", "model", "finest.tar"),
        map_location=ckpt_device,
    )
    length_estimator.load_state_dict(len_ckpt["estimator"])

    ae.to(DEVICE).eval()
    mardm.to(DEVICE).eval()
    length_estimator.to(DEVICE).eval()
    return ae, mardm, length_estimator


# ----------------------------------------------------------------------------
# Warm-load the default model once per container.
# ----------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def _get_models(dataset_name, model_name, model_arch, ae_name, ae_arch):
    key = (dataset_name, model_name, model_arch, ae_name, ae_arch)
    if key not in _MODEL_CACHE:
        ae, mardm, length_estimator = _load_models(
            dataset_name, model_name, model_arch, ae_name, ae_arch
        )
        mean, std = _load_stats(dataset_name)
        _MODEL_CACHE[key] = {
            "ae": ae,
            "mardm": mardm,
            "length_estimator": length_estimator,
            "mean": mean,
            "std": std,
        }
    return _MODEL_CACHE[key]


# Eagerly warm the default model so the first request doesn't pay cold-load cost.
try:
    _get_models(DEFAULT_DATASET, DEFAULT_MODEL_NAME, DEFAULT_MODEL_ARCH,
                DEFAULT_AE_NAME, DEFAULT_AE_ARCH)
    print(f"[MARDM] Pre-loaded {DEFAULT_MODEL_NAME} ({DEFAULT_DATASET}) on {DEVICE}.")
except Exception as exc:
    print(f"[MARDM] WARN: failed to pre-load default model: {exc}")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _b64_npy(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_video_b64(joints: np.ndarray, caption: str, kinematic_chain) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        path = fh.name
    try:
        plot_3d_motion(path, kinematic_chain, joints, title=caption, fps=20)
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    finally:
        if os.path.exists(path):
            os.remove(path)


def handler(job):
    job_input = job.get("input") or {}

    prompts = job_input.get("text_prompt")
    if prompts is None or prompts == "":
        return {"error": "`text_prompt` is required (string or list of strings)."}
    if isinstance(prompts, str):
        prompts = [prompts]
    if not all(isinstance(p, str) and p.strip() for p in prompts):
        return {"error": "Every `text_prompt` entry must be a non-empty string."}

    motion_length = job_input.get("motion_length", 0)
    if isinstance(motion_length, list):
        if len(motion_length) != len(prompts):
            return {"error": "`motion_length` list must match `text_prompt` length."}
        lengths = motion_length
    else:
        lengths = [int(motion_length)] * len(prompts)
    est_length = any((l is None) or (int(l) <= 0) for l in lengths)

    dataset_name = job_input.get("dataset", DEFAULT_DATASET)
    if dataset_name not in ("t2m", "kit"):
        return {"error": "`dataset` must be 't2m' or 'kit'."}

    model_name = job_input.get("model_name", DEFAULT_MODEL_NAME)
    model_arch = job_input.get("model_arch", DEFAULT_MODEL_ARCH)
    ae_name = job_input.get("ae_name", DEFAULT_AE_NAME)
    ae_arch = job_input.get("ae_model", DEFAULT_AE_ARCH)

    seed = int(job_input.get("seed", 3407))
    cfg = float(job_input.get("cfg", 4.5))
    time_steps = int(job_input.get("time_steps", 18))
    temperature = float(job_input.get("temperature", 1.0))
    repeat_times = int(job_input.get("repeat_times", 1))
    hard_pseudo_reorder = bool(job_input.get("hard_pseudo_reorder", False))
    render_video = bool(job_input.get("render_video", False))

    _set_seed(seed)

    try:
        bundle = _get_models(dataset_name, model_name, model_arch, ae_name, ae_arch)
    except Exception as exc:
        return {"error": f"failed to load model: {exc}"}

    ae = bundle["ae"]
    mardm = bundle["mardm"]
    length_estimator = bundle["length_estimator"]
    mean = bundle["mean"]
    std = bundle["std"]
    _, nb_joints, kinematic_chain = _dataset_geometry(dataset_name)

    if est_length:
        with torch.no_grad():
            text_embedding = mardm.encode_text(prompts)
            pred_dis = length_estimator(text_embedding)
            probs = F.softmax(pred_dis, dim=-1)
            token_lens = Categorical(probs).sample()
    else:
        token_lens = torch.LongTensor([int(l) // 4 for l in lengths]).to(DEVICE).long()

    m_length = (token_lens * 4).detach().cpu().tolist()

    progress_cb = make_progress_callback(job)

    results = []
    for r in range(repeat_times):
        with torch.no_grad():
            pred_latents = mardm.generate(
                prompts, token_lens, time_steps, cfg,
                temperature=temperature,
                hard_pseudo_reorder=hard_pseudo_reorder,
                on_step=progress_cb,
            )
            pred_motions = ae.decode(pred_latents).detach().cpu().numpy()
            data = pred_motions * std + mean

        for k, (caption, joint_data) in enumerate(zip(prompts, data)):
            seq_len = int(m_length[k])
            joint_data = joint_data[:seq_len]
            joints = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
            entry = {
                "repeat": r,
                "index": k,
                "prompt": caption,
                "length": seq_len,
                "joints_shape": list(joints.shape),
                "joints_b64": _b64_npy(joints),
            }
            if render_video:
                try:
                    entry["video_b64"] = _render_video_b64(joints, caption, kinematic_chain)
                    entry["video_mime"] = "video/mp4"
                except Exception as exc:
                    entry["video_error"] = str(exc)
            results.append(entry)

    return {
        "dataset": dataset_name,
        "model_name": model_name,
        "device": str(DEVICE),
        "results": results,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
