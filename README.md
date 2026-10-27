# Dex SuperMimic

> **Fork addition:** an optional model-free PPO baseline and complete-episode return logging.
> Upstream SuperMimic remains the default. See [PPO run instructions and comparison limitations](docs/PPO_BASELINE.md).
> PPO code and comparison notes are additions in this fork, not claims by the upstream author.

![Dex SuperMimic teaser](teaser.jpg)

This repository reproduces [SuperMimic](https://supermimic.github.io/) on a dexterous hand, adapting its world-model-based training approach to in-hand cube reorientation with Wuji Hand 2. It is built on mjlab and MuJoCo Warp.

## Download and Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Clone this repository and install the dependencies:

```bash
git clone https://github.com/CumulusAlpha/Dex-Supermimic.git
cd Dex-Supermimic
uv sync
```

`uv` creates the local Python environment and installs the project dependencies, including the bundled Wuji hand models and meshes. No separate motion dataset or asset download is required.

## Training

Run from the repository root:

```bash
uv run dex-supermimic-train --num-envs 1024 --device cuda:0
```

This launches 1024 parallel environments on the first CUDA GPU. Training logs and checkpoints are saved to `logs/dex_supermimic/<timestamp>/`.

Training logs locally with TensorBoard by default. To use Weights & Biases, append `--logger wandb`; to disable logging, append `--logger none`.

## Visualization

Download [`model_8000.pt`](https://github.com/CumulusAlpha/Dex-Supermimic/releases/download/pretrained-wuji-v1/model_8000.pt) from the [pretrained checkpoint release](https://github.com/CumulusAlpha/Dex-Supermimic/releases/tag/pretrained-wuji-v1) and place it at `logs/dex_supermimic/pretrained/model_8000.pt`. From the repository root:

```bash
mkdir -p logs/dex_supermimic/pretrained
curl -fL --retry 3 \
  https://github.com/CumulusAlpha/Dex-Supermimic/releases/download/pretrained-wuji-v1/model_8000.pt \
  -o logs/dex_supermimic/pretrained/model_8000.pt
```

The pretrained checkpoint is distributed as a Release attachment and is downloaded separately from the source code. The `logs/` directory is ignored by Git.

From the repository root, load the pretrained checkpoint in the Viser web viewer:

```bash
uv run dex-supermimic-play \
  --checkpoint logs/dex_supermimic/pretrained/model_8000.pt
```

Playback uses Viser and one environment by default and runs until you press `Ctrl+C` (`--steps 0` is the default). Open the viewer URL printed in the terminal in your browser. To limit playback, pass a positive control-step count such as `--steps 2000`. Replace the checkpoint path to view your own trained policy.

## Acknowledgments

We thank the [SuperMimic](https://supermimic.github.io/) authors for the original method.

We thank the authors and contributors of the following repositories:

1. [InstinctMJ](https://github.com/project-instinct/InstinctMJ) — the foundation of this repository and its mjlab-based simulation framework.
2. [Wuji MJLab](https://github.com/wuji-technology/wuji-mjlab) — the Wuji hand cube reorientation task and environment implementation.
3. [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) — the SuperMimic reproduction.
