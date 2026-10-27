"""Complete-episode returns on the real simulator clock, never imagined rollouts."""

import json
import time
from pathlib import Path
import torch


class EpisodeReturnRecorder:
    def __init__(self, folder, num_envs, device, interaction_offset=0):
        self.returns = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.finite = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.episode_ids = [0] * num_envs
        self.segment = str(time.time_ns())
        self.interaction_offset = interaction_offset
        self.file = open(Path(folder) / "episode_returns.jsonl", "a", buffering=1)
        self.num_envs = num_envs
        self.file.write(
            json.dumps(
                dict(
                    type="segment_start",
                    segment=self.segment,
                    interaction_offset=interaction_offset,
                    definition="undiscounted sum of real env.step rewards including terminal reward",
                    partial_episodes_discarded_on_resume=True,
                )
            )
            + "\n"
        )

    @torch.no_grad()
    def record(self, reward, terminated, truncated, *, interactions, elapsed_seconds, reasons):
        self.finite &= torch.isfinite(reward)
        self.returns += reward.double()
        self.lengths += 1
        done = terminated | truncated
        ids = done.nonzero().flatten().tolist()
        for i in ids:
            ok = bool(self.finite[i])
            row = dict(
                type="episode",
                segment=self.segment,
                env_id=i,
                episode_id=self.episode_ids[i],
                episode_return=float(self.returns[i]) if ok else None,
                finite=ok,
                episode_steps=int(self.lengths[i]),
                interactions=int(interactions),
                elapsed_seconds=float(elapsed_seconds),
                terminated=bool(terminated[i]),
                truncated=bool(truncated[i]),
                end_reasons={k: bool(v[i]) for k, v in reasons.items()},
            )
            self.file.write(json.dumps(row, allow_nan=False) + "\n")
            self.episode_ids[i] += 1
        self.returns[done] = 0
        self.lengths[done] = 0
        self.finite[done] = True

    def close(self):
        self.file.close()
