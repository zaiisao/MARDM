# MARDM serverless image for RunPod Hub.
#
# Bakes the MARDM-SiT-XL HumanML3D checkpoints into the image so the worker can
# start without external downloads. Build args allow swapping in your own
# Drive IDs (e.g. for KIT or for newer model releases) without editing the file.

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    MPLBACKEND=Agg

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg unzip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r /workspace/requirements.txt

# Pre-download OpenAI CLIP ViT-B/32 (~338 MB) into the image so cold starts
# don't pay the runtime download cost. MARDM's text encoder calls clip.load
# at first use and it would otherwise pull these weights every cold start.
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')"

# ---------------------------------------------------------------------------
# Download HumanML3D MARDM-SiT-XL checkpoints. Override via --build-arg if you
# want a different model bundle.
# ---------------------------------------------------------------------------
ARG MARDM_GDRIVE_ID=1TBybFByAd-kD4AuFgMyR3ZBt4VV43Sif
ARG AE_GDRIVE_ID=1nfX_j8VzMmynqKv8x68pXrsL3c0qWLXA
ARG LEN_GDRIVE_ID=1nWoEcN4rEFKi4Xyf_ObKinDmSQNPKXgU

RUN mkdir -p /workspace/checkpoints/t2m && cd /workspace/checkpoints/t2m \
    && gdown --fuzzy "https://drive.google.com/file/d/${MARDM_GDRIVE_ID}/view" -O MARDM_SiT_XL.zip \
    && gdown --fuzzy "https://drive.google.com/file/d/${AE_GDRIVE_ID}/view"    -O AE_humanml3d.zip \
    && gdown --fuzzy "https://drive.google.com/file/d/${LEN_GDRIVE_ID}/view"   -O length_estimator.zip \
    && unzip -o MARDM_SiT_XL.zip   && rm MARDM_SiT_XL.zip \
    && unzip -o AE_humanml3d.zip   && rm AE_humanml3d.zip \
    && unzip -o length_estimator.zip && rm length_estimator.zip \
    && ls -la /workspace/checkpoints/t2m

# ---------------------------------------------------------------------------
# Project code. Copy last so checkpoint layers stay cached on edits.
# ---------------------------------------------------------------------------
COPY . /workspace

# The README notes that the bundled utils/eval_mean_std files are acceptable
# substitutes for the dataset Mean.npy / Std.npy that sample.py expects.
RUN mkdir -p /workspace/datasets/HumanML3D /workspace/datasets/KIT-ML \
    && cp /workspace/utils/eval_mean_std/t2m/eval_mean.npy /workspace/datasets/HumanML3D/Mean.npy \
    && cp /workspace/utils/eval_mean_std/t2m/eval_std.npy  /workspace/datasets/HumanML3D/Std.npy \
    && cp /workspace/utils/eval_mean_std/kit/eval_mean.npy /workspace/datasets/KIT-ML/Mean.npy \
    && cp /workspace/utils/eval_mean_std/kit/eval_std.npy  /workspace/datasets/KIT-ML/Std.npy

ENV CHECKPOINTS_DIR=/workspace/checkpoints \
    DATASET_DIR=/workspace/datasets \
    DEFAULT_DATASET=t2m \
    DEFAULT_MODEL_NAME=MARDM_SiT_XL \
    DEFAULT_MODEL_ARCH=MARDM-SiT-XL

CMD ["python", "-u", "handler.py"]
