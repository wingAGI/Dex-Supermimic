"""Model-free PPO ablation. Never constructs a world model or replay buffer.

Gaussian noise around the unchanged bounded Actor mean, with latent-action
log probabilities and clipping only at the environment action boundary.
"""

import json, math, random, time
from pathlib import Path
import numpy as np
import torch
from .networks import Actor, Critic
from .observations import TaskObservations
from .sim import SimReader
from .metrics import MetricLogger
from .episode_returns import EpisodeReturnRecorder


from .ppo_math import gae, log_prob, clipped_policy_loss


class ModelFreePPORunner:
    def __init__(self, env, cfg, log_dir):
        self.env, self.cfg, self.device = env, cfg, env.device
        self.env.max_common_steps = cfg.max_iterations * cfg.ppo_rollout_steps
        if cfg.model_checkpoint:
            raise ValueError("PPO supports --checkpoint for PPO resume, not WM-only initialization")
        self.obs = env.reset()[0]["policy"]
        self.reader = SimReader(env)
        self.observations = TaskObservations(self.reader)
        self.actor = Actor(self.obs.shape[-1]).to(self.device)
        self.critic = Critic(self.features().shape[-1], cfg.amp).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = MetricLogger(
            cfg.logger, log_dir=self.log_dir, config=cfg.to_dict(), project=cfg.wandb_project, entity=cfg.wandb_entity
        )
        self.iteration = 0
        self.sim_steps = 0
        self.wall_offset = 0.0
        self.started_at = time.perf_counter()
        if cfg.checkpoint:
            self.load(cfg.checkpoint)
        self.episodes = EpisodeReturnRecorder(self.log_dir, env.num_envs, self.device, self.sim_steps)
        (self.log_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        (self.log_dir / "algorithm.json").write_text(
            json.dumps(
                dict(
                    algorithm="ppo",
                    world_model=False,
                    real_rollout_steps=cfg.ppo_rollout_steps,
                    fixed_gaussian_std=cfg.action_noise,
                    action_distribution="Gaussian latent, environment clamp [-1,1]",
                    comparison="algorithm baseline; not an isolated removal of a loss term",
                ),
                indent=2,
            )
        )

    def features(self):
        self.reader.refresh_frame()
        return self.observations.critic(
            self.observations.context(
                self.reader.state(), self.reader.memory(), self.observations.auxiliary(), self.reader.xi
            )
        )

    @torch.no_grad()
    def collect(self):
        c = self.cfg
        # Freeze observation normalization throughout a rollout and its PPO epochs.
        self.actor.norm.update(self.obs)
        self.critic.norm.update(self.features())
        data = {
            k: []
            for k in ["obs", "features", "action", "logp", "reward", "value", "next_value", "terminated", "truncated"]
        }
        for _ in range(c.ppo_rollout_steps):
            obs = self.obs.clone()
            features = self.features().clone()
            mean = self.actor(obs)
            action = mean + c.action_noise * torch.randn_like(mean)
            value = self.critic(features)
            lp = log_prob(mean, action, c.action_noise)
            nxt, reward, term, trunc, _ = self.env.step(action.clamp(-1, 1))
            # auto_reset=False: timeout bootstrap uses the actual terminal successor.
            next_value = self.critic(self.features())
            self.sim_steps += self.env.num_envs
            self.episodes.record(
                reward,
                term,
                trunc,
                interactions=self.sim_steps,
                elapsed_seconds=self.wall_offset + time.perf_counter() - self.started_at,
                reasons={
                    k: self.env.termination_manager.get_term(k) for k in self.env.termination_manager.active_terms
                },
            )
            transition = dict(
                obs=obs,
                features=features,
                action=action,
                logp=lp,
                reward=reward,
                value=value,
                next_value=next_value,
                terminated=term,
                truncated=trunc,
            )
            for key, value in transition.items():
                data[key].append(value.clone())
            done = term | trunc
            self.obs = nxt["policy"]
            ids = done.nonzero().flatten()
            if len(ids):
                reset, _ = self.env.reset(env_ids=ids)
                self.obs = torch.where(done[:, None], reset["policy"], self.obs)
        data = {k: torch.stack(v) for k, v in data.items()}
        if not all(torch.isfinite(v).all() for v in data.values()):
            raise RuntimeError("Nonfinite PPO rollout")
        adv, ret = gae(
            data["reward"], data["value"], data["next_value"], data["terminated"], data["truncated"], c.gamma, c.lam
        )
        data["adv"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        data["return"] = ret
        return {k: v.flatten(0, 1) for k, v in data.items()}

    def update(self, d):
        c = self.cfg
        n = len(d["logp"])
        stats = []
        for _ in range(c.ppo_epochs):
            for ids in torch.tensor_split(torch.randperm(n, device=self.device), c.ppo_minibatches):
                lp = log_prob(self.actor(d["obs"][ids]), d["action"][ids], c.action_noise)
                logratio = lp - d["logp"][ids]
                actor_loss, ratio = clipped_policy_loss(logratio, d["adv"][ids], c.ppo_clip)
                critic_loss = 0.5 * (self.critic(d["features"][ids]) - d["return"][ids]).square().mean()
                for optimizer, loss, module in [
                    (self.actor_optimizer, actor_loss, self.actor),
                    (self.critic_optimizer, c.ppo_value_coefficient * critic_loss, self.critic),
                ]:
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(module.parameters(), c.ppo_grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    if not all(torch.isfinite(p).all() for p in module.parameters()):
                        raise RuntimeError("Nonfinite PPO parameter")
                stats.append(
                    torch.stack(
                        [
                            actor_loss.detach(),
                            critic_loss.detach(),
                            ((ratio - 1) - logratio).mean().detach(),
                            ((ratio - 1).abs() > c.ppo_clip).float().mean().detach(),
                        ]
                    )
                )
        return dict(
            zip(["actor_loss", "critic_loss", "approx_KL", "clip_fraction"], torch.stack(stats).mean(0).tolist())
        )

    def learn(self):
        c = self.cfg
        for it in range(self.iteration, c.max_iterations):
            self.iteration = it
            lr = c.lr * (
                c.lr_final_ratio
                + (1 - c.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * min(it / c.lr_decay_iters, 1)))
            )
            for opt in [self.actor_optimizer, self.critic_optimizer]:
                for group in opt.param_groups:
                    group["lr"] = lr
            d = self.collect()
            stats = self.update(d)
            stats.update(
                reward=float(d["reward"].mean()),
                sim_steps=self.sim_steps,
                wall_seconds=self.wall_offset + time.perf_counter() - self.started_at,
                lr=lr,
            )
            if it % c.log_interval == 0:
                self.writer.log_scalars(stats, it)
                self.writer.flush()
                print(json.dumps(dict(iteration=it, **stats)), flush=True)
            if (it + 1) % c.save_interval == 0:
                self.save(self.log_dir / f"model_{it+1}.pt", it + 1)
        self.save(self.log_dir / "model_final.pt", c.max_iterations)

    def save(self, path, next_iteration=None):
        s = dict(
            format_version=3,
            config=self.cfg.to_dict(),
            iteration=self.iteration if next_iteration is None else next_iteration,
            sim_steps=self.sim_steps,
            wall_seconds=self.wall_offset + time.perf_counter() - self.started_at,
            modules=dict(actor=self.actor.state_dict(), critic=self.critic.state_dict()),
            optimizers=dict(actor=self.actor_optimizer.state_dict(), critic=self.critic_optimizer.state_dict()),
            joint_names=list(self.reader.robot.joint_names),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            common_step=self.env.common_step_counter,
        )
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        torch.save(s, tmp)
        tmp.replace(path)

    def load(self, path):
        s = torch.load(path, map_location=self.device, weights_only=False)
        if s["config"].get("algorithm") != "ppo":
            raise ValueError("PPO resume requires a PPO checkpoint")
        if s["joint_names"] != list(self.reader.robot.joint_names) or s["config"]["control_hz"] != self.cfg.control_hz:
            raise ValueError("Checkpoint task mismatch")
        for key in [
            "num_envs",
            "ppo_rollout_steps",
            "ppo_epochs",
            "ppo_minibatches",
            "ppo_clip",
            "ppo_value_coefficient",
            "ppo_grad_clip",
            "action_noise",
            "gamma",
            "lam",
            "max_iterations",
            "lr",
            "lr_decay_iters",
            "lr_final_ratio",
            "amp",
        ]:
            if s["config"].get(key) != getattr(self.cfg, key):
                raise ValueError(f"PPO resume configuration mismatch: {key}")
        for k in ["actor", "critic"]:
            getattr(self, k).load_state_dict(s["modules"][k])
            getattr(self, k + "_optimizer").load_state_dict(s["optimizers"][k])
        self.iteration = s["iteration"]
        self.sim_steps = s["sim_steps"]
        self.wall_offset = s["wall_seconds"]
        self.env.common_step_counter = s["common_step"]
        torch.set_rng_state(s["torch_rng"].cpu())
        torch.cuda.set_rng_state(s["cuda_rng"].cpu(), self.device)
        np.random.set_state(s["numpy_rng"])
        random.setstate(s["python_rng"])

    def close(self):
        self.episodes.close()
        self.writer.close()
