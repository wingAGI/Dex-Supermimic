"""Run three PPO iterations plus two resumed iterations in a real CUDA task.

Explicit, bounded smoke test; never part of CPU CI. Supply an output directory
on a free GPU. This verifies execution and checkpoint wiring, not learning quality.
"""

import argparse
import json
from pathlib import Path
import torch
from dex_supermimic.rl.dex_supermimic.cli import train_config
from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:0")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise SystemExit("Output must be a new directory")
a.output.mkdir(parents=True)
kwargs = dict(
    algorithm="ppo",
    device=a.device,
    num_envs=8,
    max_iterations=3,
    ppo_rollout_steps=32,
    save_interval=1,
    log_interval=1,
    logger="none",
    amp=False,
)
# A checkpoint at iteration 1 also exercises timeout/disturbance lifecycle
# during a full rollout; the follow-up intentionally refills fresh environments.
train_config(DexSuperMimicCfg(**kwargs), a.output / "fresh")
fresh = torch.load(a.output / "fresh/model_final.pt", map_location="cpu", weights_only=False)
assert fresh["iteration"] == 3 and fresh["sim_steps"] == 3 * 32 * 8
assert set(fresh["modules"]) == {"actor", "critic"}
checkpoint = a.output / "fresh/model_1.pt"
train_config(DexSuperMimicCfg(**kwargs, checkpoint=str(checkpoint)), a.output / "resume")
resumed = torch.load(a.output / "resume/model_final.pt", map_location="cpu", weights_only=False)
assert resumed["iteration"] == 3 and resumed["sim_steps"] == fresh["sim_steps"]
assert any(
    not torch.equal(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["modules"]["actor"][k],
        resumed["modules"]["actor"][k],
    )
    for k in resumed["modules"]["actor"]
    if k.startswith("net.")
)
print(
    json.dumps(
        {
            "smoke": "PASS",
            "fresh_iteration": 3,
            "resume_iteration": 3,
            "real_transitions_each": fresh["sim_steps"],
            "world_model_modules": False,
        }
    )
)
