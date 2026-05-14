#!/bin/bash
# Setup script to create a working virtual environment for
# train_qwen2_5_vl.py (and related Qwen2.5-VL training jobs).
#
# This script is self-contained: it clones FlagScale, Megatron-LM-FL, and
# apex (at pinned commits) into BASE_DIR if they are not already present,
# then installs all dependencies into a venv.
#
# Usage:
#   bash setup_venv.sh [venv_name] [base_dir]
#
#   venv_name : name of the venv directory created inside FlagScale (default: venv)
#   base_dir  : directory where repos are cloned (default: parent of this script)
#
# On compute nodes load modules first:
#   module load gcc cuda/12.6 && bash setup_venv.sh
module load gcc 
module load cuda/12.6


set -e

# ── Directories ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${2:-$(dirname "${SCRIPT_DIR}")}"

VENV_NAME="${1:-venv}"
VENV_DIR="${SCRIPT_DIR}/${VENV_NAME}"

FLAGSCALE_DIR="${BASE_DIR}/FlagScale"
MEGATRON_DIR="${BASE_DIR}/Megatron-LM-FL"
APEX_DIR="${BASE_DIR}/apex"

# ── Pinned repository sources ─────────────────────────────────────────────────
FLAGSCALE_REPO="https://github.com/flagos-ai/FlagScale.git"
FLAGSCALE_COMMIT="73d9bebacad4d3264c6dbe7e2cef1ec1ec2cecfd"

MEGATRON_REPO="https://github.com/flagos-ai/Megatron-LM-FL.git"
MEGATRON_COMMIT="17a72ee6882eb2215c9ab4fd15fdbf07dcead3d8"

APEX_REPO="https://github.com/NVIDIA/apex.git"
APEX_COMMIT="ba32a259b7aa4a7d797369543ead466fe4a760a5"

# Energon commit pinned to match 2venv
ENERGON_COMMIT="ab40226100830f41de38d1f1204d7848b54b1f3e"

# ── Python interpreter ────────────────────────────────────────────────────────
# Try the well-known CINECA path first; fall back to whatever is on PATH.
_PYTHON311_CINECA=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/python-3.11.7-ziwh63aulhhzxksf42k5u3gnim2rbpmp/bin/python3.11
if [[ -x "${_PYTHON311_CINECA}" ]]; then
    PYTHON311="${_PYTHON311_CINECA}"
elif command -v python3.11 &>/dev/null; then
    PYTHON311="$(command -v python3.11)"
else
    echo "ERROR: python3.11 not found. Install it or load the required module." >&2
    exit 1
fi
echo "Using Python: ${PYTHON311}"

# ── Clone / update repositories ───────────────────────────────────────────────
clone_at_commit() {
    local repo="$1" dir="$2" commit="$3"
    if [[ -d "${dir}/.git" ]]; then
        echo "=== ${dir} already exists – skipping clone ==="
    else
        echo "=== Cloning ${repo} into ${dir} ==="
        git clone "${repo}" "${dir}"
        git -C "${dir}" checkout "${commit}"
    fi
}

clone_at_commit "${FLAGSCALE_REPO}" "${FLAGSCALE_DIR}" "${FLAGSCALE_COMMIT}"
clone_at_commit "${MEGATRON_REPO}"  "${MEGATRON_DIR}"  "${MEGATRON_COMMIT}"
clone_at_commit "${APEX_REPO}"      "${APEX_DIR}"      "${APEX_COMMIT}"

# ── Virtual environment ───────────────────────────────────────────────────────
# Derive the versioned lib path (e.g. python3.11) from the interpreter itself.
PYTHON_VERSION=$("${PYTHON311}" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')

echo "=== Creating venv at ${VENV_DIR} ==="
"${PYTHON311}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "=== Upgrading pip/setuptools/wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing PyTorch 2.10.0 (CUDA 12.8) ==="
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing triton ==="
pip install triton==3.6.0

echo "=== Installing megatron-energon (pinned commit) ==="
pip install "megatron-energon @ git+https://github.com/NVIDIA/Megatron-Energon.git@${ENERGON_COMMIT}"

echo "=== Installing megatron-core from local Megatron-LM-FL ==="
pip install "${MEGATRON_DIR}"

echo "=== Installing flagscale ==="
pip install "${FLAGSCALE_DIR}"

echo "=== Installing apex 0.1 (pure Python, pre-built egg) ==="
# Use the pre-built pure-Python egg to avoid needing CUDA at install time.
# Modern pip does not support .egg files; register it via a .pth file instead.
# If you want to rebuild with CUDA extensions on a compute node:
#   module load cuda/12.6 && cd ${APEX_DIR} && pip install --no-build-isolation -v .
APEX_EGG="${APEX_DIR}/dist/apex-0.1-${PYTHON_VERSION}.egg"
if [[ -f "${APEX_EGG}" ]]; then
    echo "${APEX_EGG}" > "${VENV_DIR}/lib/${PYTHON_VERSION}/site-packages/apex-egg-link.pth"
    echo "Registered apex egg via .pth: ${APEX_EGG}"
else
    echo "Pre-built apex egg not found at ${APEX_EGG}; falling back to source install."
    cd "${APEX_DIR}"
    pip install --no-build-isolation -v --no-cache-dir .
    cd "${SCRIPT_DIR}"
fi

echo "=== Installing TransformerEngine ==="
# cuDNN, NCCL and other CUDA libraries come as Python wheels (nvidia-*-cu12).
# Their headers live under site-packages/nvidia/<lib>/include/ but the C++ build
# doesn't know about them.  Collect all of those include dirs and prepend them to
# CPATH so g++ can find cudnn.h, nccl.h, cublas_v2.h, etc.
NVIDIA_SITE="${VENV_DIR}/lib/${PYTHON_VERSION}/site-packages/nvidia"
NVIDIA_INCLUDES=$(find "${NVIDIA_SITE}" -maxdepth 2 -name include -type d | tr '\n' ':')
export CPATH="${NVIDIA_INCLUDES}${CPATH:+${CPATH}}"
export CUDA_HOME="${CUDA_HOME:-/leonardo/prod/opt/compilers/cuda/12.6/none}"
# --no-build-isolation: reuse the venv's torch 2.10.0+cu128 (avoids mismatched isolated env)
NVTE_FRAMEWORK=pytorch pip install --no-build-isolation transformer-engine[pytorch]

echo "=== Installing remaining packages ==="
pip install \
    aiobotocore==3.3.0 \
    aiohttp==3.13.3 \
    aioitertools==0.13.0 \
    aniso8601==10.0.1 \
    annotated-types==0.7.0 \
    anyio==4.12.1 \
    attrs==26.1.0 \
    av==17.0.0 \
    blinker==1.9.0 \
    botocore==1.42.70 \
    braceexpand==0.1.7 \
    bracex==2.6 \
    cffi==2.0.0 \
    charset-normalizer==3.4.6 \
    click==8.3.1 \
    cuda-bindings==12.9.4 \
    cuda-pathfinder==1.4.3 \
    deprecation==2.1.0 \
    filelock==3.25.2 \
    filetype==1.2.0 \
    Flask==3.1.3 \
    Flask-RESTful==0.3.10 \
    frozenlist==1.8.0 \
    fsspec==2026.2.0 \
    GitPython==3.1.46 \
    huggingface_hub==1.8.0 \
    httpx==0.28.1 \
    importlib_metadata==8.7.1 \
    Jinja2==3.1.6 \
    jmespath==1.1.0 \
    jsonschema==4.26.0 \
    lark==1.3.1 \
    markdown-it-py==4.0.0 \
    mfusepy==3.1.1 \
    mpmath==1.3.0 \
    multi-storage-client==0.44.0 \
    multidict==6.7.1 \
    networkx==3.6.1 \
    numpy==2.4.3 \
    opentelemetry-api==1.40.0 \
    packaging==26.0 \
    pillow==12.1.1 \
    platformdirs==4.9.4 \
    prettytable==3.17.0 \
    protobuf==6.33.6 \
    psutil==7.2.2 \
    pybind11==3.0.4 \
    pydantic==2.12.5 \
    python-dateutil==2.9.0.post0 \
    pytz==2026.1.post1 \
    PyYAML==6.0.3 \
    rapidyaml==0.11.0.post1 \
    regex==2026.2.28 \
    requests==2.32.5 \
    rich==14.3.3 \
    s3fs==2026.2.0 \
    safetensors==0.7.0 \
    scipy==1.17.1 \
    sentencepiece==0.2.1 \
    sentry-sdk==2.55.0 \
    six==1.17.0 \
    sympy==1.14.0 \
    tiktoken==0.12.0 \
    tokenizers==0.22.2 \
    tqdm==4.67.3 \
    typer==0.24.1 \
    typing_extensions==4.15.0 \
    urllib3==2.6.3 \
    wandb==0.25.1 \
    wcmatch==10.1 \
    webdataset==1.0.2 \
    wrapt==2.1.2 \
    xattr==1.3.0


pip install transformers==4.57.6 \
echo ""
echo "=== Verifying key imports ==="
python -c "
import sys
ok = True
for mod in ['torch', 'megatron.core', 'megatron.energon', 'flagscale', 'transformers', 'apex', 'wandb', 'PIL']:
    try:
        __import__(mod)
        print(f'  OK  {mod}')
    except ImportError as e:
        print(f'  FAIL {mod}: {e}')
        ok = False
sys.exit(0 if ok else 1)
"

echo ""
echo "=== Done: ${VENV_DIR} ==="
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo ""
echo "To use this env in qwenvl_run_job.sh, replace the two lines:"
echo "  source 2venv/bin/activate"
echo "  export PYTHONPATH=.../2venv/lib/python3.11/site-packages:..."
echo "with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  export PYTHONPATH=${VENV_DIR}/lib/${PYTHON_VERSION}/site-packages:${VENV_DIR}/lib/${PYTHON_VERSION}/site-packages/flagscale/train:\${PYTHONPATH}"
