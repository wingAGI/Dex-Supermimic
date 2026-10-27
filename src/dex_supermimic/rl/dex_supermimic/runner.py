"""Collect -> multi-step dynamics/drop fit -> differentiable imagination -> lambda-return critic.

The environment retains the original reward/termination timing and domain randomization.
Only cage_drop is a learned terminal event. Timeouts/NaNs still end replay sequences.
"""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .rotations import mv, quat_to_mat

from .metrics import MetricLogger
from .episode_returns import EpisodeReturnRecorder
from .buffer import ReplayBuffer
from .common import RunningNorm
from .networks import Actor, Critic, FailureHead, WorldModel
from .observations import BASE_TERMS, CURRENT_TERMS, TaskObservations
from .sim import SimReader
from .state import difference, ema_target, flatten_memory, unflatten_memory


def format_training_log(iteration, max_iterations, stats):
    """Format console metrics without changing TensorBoard scalar precision."""
    elapsed = int(stats["wall_seconds"])
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    lines = [
        "",
        "=" * 72,
        f"Iteration {iteration:,} / {max_iterations:,}"
        f"  |  Time {hours:02d}:{minutes:02d}:{seconds:02d}"
        f"  |  Steps {stats['sim_steps']:,.0f}",
        "-" * 72,
    ]
    groups = (
        ("Rollout", ("reward", "reward_parity", "drop_rate")),
        ("Policy", ("actor_loss", "critic_loss", "survival", "horizon")),
    )
    for title, keys in groups:
        values = [f"{key}={stats[key]:.6g}" for key in keys if key in stats]
        if values:
            lines.append(f"{title}:")
            for offset in range(0, len(values), 2):
                lines.append("  " + "  |  ".join(values[offset : offset + 2]))
    lines.extend(
        [
            "World model:",
            f"  {'Metric':<20} {'Train':>16} {'Holdout':>16}",
        ]
    )
    for key in ("wm_loss", "failure_loss"):
        train = f"{stats[key]:.6g}" if key in stats else "--"
        holdout_key = f"holdout/{key}"
        holdout = f"{stats[holdout_key]:.6g}" if holdout_key in stats else "--"
        lines.append(f"  {key:<20} {train:>16} {holdout:>16}")
    lines.append("=" * 72)
    return "\n".join(lines)


class DexSuperMimicRunner:
    def __init__(self, env, cfg, log_dir=None):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        if torch.distributed.is_initialized():
            raise ValueError("DexSuperMimic currently supports one GPU/process")
        self.obs = env.reset()[0]["policy"]
        self.reader = SimReader(env)
        self.mdp = self.reader.mdp
        self.observations = TaskObservations(self.reader)
        initial_context = self.observations.context(
            self.reader.state(), self.reader.memory(), self.observations.auxiliary(), self.reader.xi
        )
        initial_dynamics = self.observations.dynamics(
            self.reader.state(), self.reader.xi, self.observations.auxiliary()[:, 12:21]
        )
        self.reader.attach_capture()
        self.actor = Actor(self.obs.shape[-1]).to(self.device)
        contact_dim = self.reader.contact().shape[-1]
        dynamics_dim = initial_dynamics.shape[-1]
        self.dynamics_dim = dynamics_dim
        context_dim = self.observations.context_dim
        self.context_dim = context_dim
        self.contact_dim = contact_dim
        self.wm = WorldModel(dynamics_dim, contact_dim, cfg.wm_hidden, cfg.amp).to(self.device)
        self.fail_head = FailureHead(context_dim).to(self.device)
        self.torque_norm = RunningNorm(20, eps=1e-3).to(self.device)
        self.critic = Critic(self.observations.critic(initial_context).shape[-1], cfg.amp).to(self.device)
        self.optimizers = {
            name: torch.optim.Adam(module.parameters(), lr=cfg.lr) for name, module in self.modules().items()
        }
        self.optimizers["actor"] = torch.optim.AdamW(
            self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0
        )
        self.buffer = ReplayBuffer(cfg.buffer_steps, env.num_envs, self.device, cfg.holdout_fraction)
        self.log_dir = Path(log_dir or cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if cfg.wandb_mode:
            os.environ["WANDB_MODE"] = cfg.wandb_mode
        self.writer = MetricLogger(
            cfg.logger,
            log_dir=self.log_dir,
            config=cfg.to_dict(),
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
        )
        self.iteration = 0
        self.sim_steps = 0
        self.wall_offset = 0.0
        self.started_at = time.perf_counter()
        self.env.cfg.auto_reset = False
        self.env.max_common_steps = cfg.warmup_steps + int(cfg.max_iterations / cfg.utd)
        self.step_wm = torch.compile(self.wm) if cfg.compile and str(self.device).startswith("cuda") else self.wm
        self.step_actor = (
            torch.compile(self.actor) if cfg.compile and str(self.device).startswith("cuda") else self.actor
        )
        self.reward_step = self.mdp.rewards
        self.task_features = self.observations.context
        self.wm_features = self.observations.dynamics
        if cfg.compile_policy and str(self.device).startswith("cuda"):
            options = {"dynamic": cfg.compile_policy_dynamic} if cfg.compile_policy_dynamic else {}
            self.reward_step = torch.compile(self.mdp.rewards, **options)
            self.task_features = torch.compile(self.observations.context, **options)
            self.wm_features = torch.compile(self.observations.dynamics, **options)
        if cfg.checkpoint:
            self.load(cfg.checkpoint)
        if cfg.model_checkpoint:
            self.load_model_only(cfg.model_checkpoint)
        (self.log_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        # Preserve the entire task configuration, including callable identities and DR.
        import dataclasses

        def serialize(value):
            if dataclasses.is_dataclass(value):
                return {f.name: serialize(getattr(value, f.name)) for f in dataclasses.fields(value)}
            if isinstance(value, dict):
                return {str(k): serialize(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [serialize(v) for v in value]
            if callable(value):
                return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        (self.log_dir / "environment_config.json").write_text(json.dumps(serialize(env.cfg), indent=2))
        import cloudpickle

        (self.log_dir / "environment_config.pkl").write_bytes(cloudpickle.dumps(env.cfg))
        (self.log_dir / "context.json").write_text(json.dumps(self.reader.context_names, indent=2))
        layout = {
            "base_terms": BASE_TERMS,
            "current_terms": CURRENT_TERMS,
            "wm_history": 1,
            "critic_history": 1,
            "wm_input_dim": self.wm.norm.mean.numel(),
            "critic_input_dim": self.critic.norm.mean.numel(),
            "dynamics_input_dim": dynamics_dim,
            "dr_params": ["object_mass_ratio", "joint_damping_ratio", "object_size_ratio"],
            "wm_extra_terms": {"control_target": 20},
            "simulator_context": self.reader.context_names,
        }
        (self.log_dir / "input_layout.json").write_text(json.dumps(layout, indent=2))

    def modules(self):
        return dict(actor=self.actor, wm=self.wm, fail_head=self.fail_head, critic=self.critic)

    def critic_features(self, context):
        return self.observations.critic(context)

    def replay_dynamics(self, w, x, j):
        """Dynamics features from predicted state; only startup/exogenous context is replayed."""
        simulator = w["sim_context"] if "sim_context" in w else self.reader.xi[w["env"]]
        return self.wm_features(x, simulator, w["aux"][:, j, 12:21])

    def advance_memory(self, x1, memory, action, target, cage, next_goal, soft=True):
        """Advance task/controller state entirely from the imagined transition."""
        updated = self.mdp.command_step(x1, memory, next_goal, soft=soft)
        updated.update(
            c1=target,
            c2=memory["c1"],
            a1=action,
            a2=memory["a1"],
            cage=cage,
            age=memory["age"] + 1,
        )
        return updated

    @staticmethod
    def advance_disturbance(auxiliary, event, world_direction, world_velocity):
        """Replay or sample the task's exogenous interval-disturbance cache."""
        return torch.cat(
            (
                auxiliary[:, :12],
                torch.where(event, world_direction, auxiliary[:, 12:15]),
                torch.where(
                    event,
                    torch.cat((world_velocity, torch.zeros_like(world_velocity)), -1),
                    auxiliary[:, 15:21],
                ),
                auxiliary[:, 21:],
            ),
            -1,
        )

    @staticmethod
    def history_from_obs(obs):
        return torch.cat(
            [v.reshape(*v.shape[:-1], 3, d) for v, d in zip(obs.split((60, 60, 9, 18, 60), -1), (20, 20, 3, 6, 20))],
            -1,
        )

    @torch.no_grad()
    def collect(self, steps):
        if not hasattr(self, "episode_return_recorder"):
            self.episode_return_recorder = EpisodeReturnRecorder(self.log_dir, self.env.num_envs, self.device, self.sim_steps)
        gaps = []
        term_gaps = {}
        rewards = []
        drops = []
        for _ in range(steps):
            self.reader.refresh_frame()
            x = self.reader.state()
            m = self.reader.memory()
            auxiliary = self.observations.auxiliary()
            context = self.observations.context(x, m, auxiliary, self.reader.xi)
            self.actor.norm.update(self.obs[torch.isfinite(self.obs).all(-1)])
            action = (self.actor(self.obs) + self.cfg.action_noise * torch.randn_like(m["a1"])).clamp(-1, 1)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            cap = self.reader.reward_snapshot
            self.episode_return_recorder.record(
                reward, terminated, truncated,
                interactions=self.sim_steps + self.env.num_envs,
                elapsed_seconds=self.wall_offset + time.perf_counter() - self.started_at,
                reasons={k: self.env.termination_manager.get_term(k)
                         for k in self.env.termination_manager.active_terms},
            )
            c = self.env.action_manager.get_term("joint_pos").processed_action.clone()
            u = c - self.reader.robot.data.encoder_bias
            next_x = self.reader.state()
            valid = (
                (~cap["nan"])
                & torch.isfinite(x).all(-1)
                & torch.isfinite(cap["physics"]).all(-1)
            )
            valid = valid & torch.isfinite(cap["contact"]).all(-1) & torch.isfinite(cap["torque"]).all(-1)
            valid &= torch.isfinite(u).all(-1) & torch.isfinite(c).all(-1)
            valid &= torch.isfinite(action).all(-1) & torch.isfinite(flatten_memory(m)).all(-1)
            valid &= torch.isfinite(next_x).all(-1) & torch.isfinite(self.obs).all(-1)
            valid &= torch.isfinite(context).all(-1)
            # Held-out environments never contribute to training statistics.
            train_valid = valid.clone()
            train_valid[self.buffer.train_end :] = False
            self.torque_norm.update(cap["torque"][train_valid])
            if self.cfg.variant == "fixed":
                dynamics = self.observations.dynamics(
                    x[train_valid], self.reader.xi[train_valid], auxiliary[train_valid, 12:21]
                )
                self.wm.norm.update(self.wm.inputs(dynamics, u[train_valid]))
                self.wm.delta_norm.update(difference(x[train_valid], cap["physics"][train_valid]))
                with torch.no_grad():
                    selected_m = {key: value[train_valid] for key, value in m.items()}
                    _, next_cage, _ = self.mdp.rewards(
                        cap["physics"][train_valid],
                        selected_m,
                        action[train_valid],
                        c[train_valid],
                        cap["contact"][train_valid],
                        cap["torque"][train_valid],
                    )
                    next_memory = self.advance_memory(
                        cap["physics"][train_valid],
                        selected_m,
                        action[train_valid],
                        c[train_valid],
                        next_cage,
                        m["goal"][train_valid].reshape(-1, 3, 3),
                        soft=False,
                    )
                    next_context = self.observations.context(
                        cap["physics"][train_valid],
                        next_memory,
                        auxiliary[train_valid],
                        self.reader.xi[train_valid],
                    )
                self.fail_head.norm.update(
                    self.fail_head.features(cap["physics"][train_valid], next_context)
                )
            done = terminated | truncated
            self.buffer.add(
                dict(
                    x=x,
                    x1=cap["physics"],
                    push=next_x - cap["physics"],
                    u=u,
                    c=c,
                    action=action,
                    m=flatten_memory(m),
                    obs=self.obs,
                    aux=auxiliary,
                    contact=cap["contact"],
                    torque=cap["torque"],
                    reward=reward,
                    done=done,
                    terminated=terminated,
                    valid=valid,
                )
            )
            # Match original SuperMimic: reward uses the same post-step state
            # that the world model predicts.
            an, _, analytic_terms = self.mdp.rewards(
                cap["physics"], m, action, c, cap["contact"], cap["torque"]
            )
            denom = valid.float().sum().clamp_min(1)
            for name, value in analytic_terms.items():
                error = (value - cap["terms"][name]).abs()
                term_gaps.setdefault(name, []).append(torch.where(valid, error, 0.0).sum() / denom)
            gaps.append(torch.where(valid, (an - cap["reward"]).abs(), 0.0).sum() / denom)
            rewards.append((reward * valid).sum() / denom)
            drops.append(cap["drop"].float().mean())
            self.obs = obs["policy"]
            self.reader.refresh_frame()
            ids = done.nonzero(as_tuple=False).flatten()
            if ids.numel():
                post, _ = self.env.reset(env_ids=ids)
                self.obs = torch.where(done[:, None], post["policy"], self.obs)
                self.reader.refresh_frame()
            self.sim_steps += self.env.num_envs
        return dict(
            **{"reward_parity/" + k: torch.stack(v).mean() for k, v in term_gaps.items()},
            norm_count=self.wm.norm.count.clone(),
            delta_norm_count=self.wm.delta_norm.count.clone(),
            reward=torch.stack(rewards).mean(),
            reward_parity=torch.stack(gaps).mean(),
            drop_rate=torch.stack(drops).mean(),
        )

    def fit_world_model(self, holdout=False):
        c = self.cfg
        window = c.wm_window
        w = self.buffer.sample(
            c.wm_batch, window, holdout,
            failure_frac=0.0 if holdout or c.variant == "legacy" else c.wm_failure_frac,
            failure_horizon=c.wm_failure_horizon,
        )
        memory = unflatten_memory(w["m"][:, 0])
        auxiliary = w["aux"][:, 0]
        simulator = w["sim_context"] if "sim_context" in w else self.reader.xi[w["env"]]
        if not holdout and c.variant == "legacy":
            # Deliberately reproduce the old WM statistics bug in the control arm.
            # The fixed arm never updates statistics from replay.
            initial_dynamics = self.observations.dynamics(
                w["x"][:, 0], simulator, w["aux"][:, 0, 12:21]
            )
            self.wm.norm.update(self.wm.inputs(initial_dynamics, w["u"][:, 0]))
            self.wm.delta_norm.update(difference(w["x"], w["x1"])[w["mask"].bool()])
        x = w["x"][:, 0]
        losses = []
        failure_features = []
        std = torch.maximum(self.wm.delta_norm.std(), self.wm.error_floor)
        for j in range(window):
            dynamics = self.wm_features(x, simulator, auxiliary[:, 12:21])
            x1, contact = self.step_wm(x, w["u"][:, j], dynamics)
            state_loss = (difference(x1, w["x1"][:, j]) / std).square().mean(-1)
            contact_loss = F.binary_cross_entropy_with_logits(contact, w["contact"][:, j], reduction="none").sum(-1)
            losses.append(
                c.final_state_weight * state_loss
                + c.contact_weight * contact_loss
            )
            tau = self.mdp.pd_torque(x, w["u"][:, j], simulator)
            _, cage, _ = self.reward_step(
                x1,
                memory,
                w["action"][:, j],
                w["c"][:, j],
                contact.sigmoid(),
                tau,
                soft=True,
            )
            # Keep the current goal during supervised open-loop replay; it is
            # not a teacher-forced geometry/context input, only the exogenous
            # command realization when a switch would occur.
            next_goal = memory["goal"].reshape(-1, 3, 3)
            memory = self.advance_memory(
                x1, memory, w["action"][:, j], w["c"][:, j], cage, next_goal
            )
            failure_features.append(
                self.fail_head.features(
                    x1.detach(),
                    self.observations.context(x1.detach(), memory, auxiliary, simulator),
                )
            )
            push = w["push"][:, j]
            event = push[..., -6:-3].abs().sum(-1, keepdim=True) > 0
            world_velocity = push[..., -6:-3]
            world_direction = torch.where(
                event,
                world_velocity / world_velocity.norm(dim=-1, keepdim=True).clamp_min(1e-8),
                world_velocity,
            )
            auxiliary = self.advance_disturbance(
                auxiliary,
                event,
                world_direction,
                world_velocity,
            )
            x = x1 + w["push"][:, j]
        mask = w["mask"]
        denom = mask.sum().clamp_min(1)
        wm_loss = (torch.stack(losses, 1) * mask).sum() / mask.shape[0]
        if not holdout:
            self.optimize("wm", wm_loss)
        true = self.replay_failure_features(w, simulator)
        predicted = torch.stack(failure_features, 1).detach()
        if not holdout and c.variant == "legacy":
            self.fail_head.norm.update(true[w["mask"].bool()])
        failure_logit_true = self.fail_head(true)
        failure_logit_pred = self.fail_head(predicted)
        label = w["terminated"].float()
        failure_loss = (
            (
                F.binary_cross_entropy_with_logits(failure_logit_true, label, reduction="none")
                + F.binary_cross_entropy_with_logits(failure_logit_pred, label, reduction="none")
            )
            * mask
            * w["importance"]
        ).sum() / (2 * denom)
        if not holdout:
            self.optimize("fail_head", failure_loss)
        return dict(
            supervision_window=window,
            wm_loss=wm_loss.detach(),
            failure_loss=failure_loss.detach(),
        )

    @torch.no_grad()
    def replay_failure_features(self, w, simulator):
        # Recorded states are independent across time. Batch B*T together so
        # observation/FK kernels launch once rather than once per window step.
        batch, window = w["x1"].shape[:2]
        x = w["x1"].flatten(0, 1)
        context = self.observations.context(
            x,
            unflatten_memory(w["m"].flatten(0, 1)),
            w["aux"].flatten(0, 1),
            simulator[:, None].expand(-1, window, -1).reshape(batch * window, -1),
        )
        return self.fail_head.features(x, context).reshape(batch, window, -1)

    def load_model_only(self, path):
        """A/B starts actor, critic and optimizers fresh; source checkpoint is read-only."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if (
            state["format_version"] not in (2, 3)
            or state["joint_names"] != list(self.reader.robot.joint_names)
            or state["context_names"] != self.reader.context_names
        ):
            raise ValueError("Model initialization layout mismatch")
        if state["config"]["control_hz"] != self.cfg.control_hz:
            raise ValueError("Model initialization frequency mismatch")
        for name in ("wm", "fail_head"):
            self.modules()[name].load_state_dict(state["modules"][name])
        if "torque_norm" in state:
            self.torque_norm.load_state_dict(state["torque_norm"])
        if self.cfg.reset_model_statistics:
            for norm in (self.wm.norm, self.wm.delta_norm, self.fail_head.norm, self.torque_norm):
                norm.mean.zero_()
                norm.var.fill_(1.0)
                norm.count.zero_()
                norm._host_count = 0.0

    def optimize(self, name, loss):
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {name} loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.modules()[name].parameters(), 1.0)
        if name == "actor":
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
        if not torch.isfinite(norm):
            if name == "actor":
                return
            raise FloatingPointError(f"Non-finite {name} gradient")
        optimizer.step()

    def lr_at(self, iteration):
        c = self.cfg
        progress = min(iteration / c.lr_decay_iters, 1.0)
        return c.lr * (c.lr_final_ratio + (1 - c.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    def sample_starts(self):
        # Like SuperMimic, draw uniformly from all eligible replay states,
        # including held-out envs (the WM fit itself remains held out).
        rows = (self.buffer.ptr - self.buffer.size + torch.arange(self.buffer.size, device=self.device)) % self.buffer.capacity
        eligible = self.buffer.data["valid"][rows].nonzero(as_tuple=False)
        if not len(eligible):
            raise RuntimeError("Replay contains no finite imagination starts")
        selected = eligible[torch.randint(len(eligible), (self.cfg.pi_batch,), device=self.device)]
        row, env = rows[selected[:, 0]], selected[:, 1]
        out = {k: v[row, env] for k, v in self.buffer.data.items()}
        out.update(env=env, sim_context=self.reader.xi[env])
        return out

    def policy_update(self):
        c = self.cfg
        w = self.sample_starts()
        x = w["x"]
        m = unflatten_memory(w["m"])
        history = self.history_from_obs(w["obs"])
        auxiliary = w["aux"]
        simulator = w["sim_context"]
        bias = self.reader.robot.data.encoder_bias[w["env"]]
        horizon = c.horizon
        frozen = (self.wm, self.fail_head, self.critic)
        for mod in frozen:
            mod.requires_grad_(False)
        xs = []
        rewards = []
        continues = []
        survival = x.new_ones(x.shape[0])
        objective = x.new_zeros(x.shape[0])
        # These quantities are constant for this update; keep them out of the
        # recurrent loop to avoid repeating host work and Python-side control flow.
        gamma_powers = [c.gamma**t for t in range(horizon + 1)]
        scale = min(
            1,
            max(
                0,
                (self.env.common_step_counter / self.env.max_common_steps - 0.05) / 0.75,
            ),
        )
        adaptive = float(getattr(self.env, "_adaptive_episode_curriculum", 0.05))
        try:
            for t in range(horizon):
                context = self.task_features(x, m, auxiliary, simulator)
                xs.append(self.critic_features(context))
                a = self.step_actor(self.mdp.flatten_history(history))
                target, u = ema_target(
                    a,
                    m["c1"],
                    self.mdp.default,
                    self.mdp.lower,
                    self.mdp.upper,
                    bias,
                    self.mdp.alpha,
                    m["age"].squeeze(-1),
                    self.mdp.dt,
                )
                x1, contact = self.step_wm(
                    x, u, self.wm_features(x, simulator, auxiliary[:, 12:21])
                )
                tau = self.mdp.pd_torque(x, u, simulator)
                r, cage, _ = self.reward_step(x1, m, a, target, contact.sigmoid(), tau, soft=True)
                # Original termination reads the PRE-reward counter.
                analytic = torch.sigmoid((m["cage"].squeeze(-1) - self.mdp.max_outside + 0.5) / 0.25)
                d = analytic
                next_goal = quat_to_mat(F.normalize(torch.randn(x.shape[0], 4, device=x.device), dim=-1))
                next_memory = self.advance_memory(
                    x1,
                    m,
                    a,
                    target,
                    cage,
                    next_goal,
                )
                if c.failure_head and self.iteration >= c.wm_only_iters + c.failure_warmup_iters:
                    next_context = self.task_features(x1, next_memory, auxiliary, simulator)
                    p_fail = self.fail_head(
                        self.fail_head.features(x1, next_context)
                    ).sigmoid().clamp(max=c.failure_clip)
                    d = 1 - (1 - d) * (1 - p_fail)
                # A timeout is not a physical termination; the critic bootstraps it.
                objective = objective + gamma_powers[t] * survival * r
                survival = survival * (1 - d)
                rewards.append(r.detach())
                continues.append((1 - d).detach())
                updated = next_memory
                # Match interval velocity disturbances independently of the physical model.
                push = torch.zeros_like(x1)
                speed = 0.05 + torch.rand(x.shape[0], 1, device=x.device) * 0.1 * scale * adaptive
                dv = F.normalize(torch.randn(x.shape[0], 3, device=x.device), dim=-1) * speed
                event = torch.rand(x.shape[0], 1, device=x.device) < self.mdp.dt / 1.2
                active = event & (
                    updated["age"] * self.mdp.dt >= 3
                )
                push[..., -6:-3] = dv * active
                # The task caches the most recent sampled world-frame disturbance.
                # A sampled event before warmup clears the cache as in the simulator.
                palm_rotation = auxiliary[:, 3:12].reshape(-1, 3, 3)
                world_dv = mv(palm_rotation, dv * active)
                world_dir = mv(palm_rotation, F.normalize(dv, dim=-1) * active)
                auxiliary = self.advance_disturbance(auxiliary, event, world_dir, world_dv)
                x = x1 + push
                m = updated
                observation = self.mdp.noisy_observation(self.mdp.observation(x, m))
                history = torch.cat((history[:, 1:], observation[:, None]), 1)
            context = self.task_features(x, m, auxiliary, simulator)
            final = self.critic_features(context)
            xs.append(final)
            objective = objective + (c.gamma**horizon) * survival * self.critic(final)
            actor_loss = -objective.mean()
            self.optimize("actor", actor_loss)
        finally:
            for mod in frozen:
                mod.requires_grad_(True)
        inputs = torch.stack(xs).detach()
        rs = torch.stack(rewards)
        cs = torch.stack(continues)
        with torch.no_grad():
            self.critic.norm.update(inputs)
            values = self.critic(inputs)
            ret = values[-1]
            targets = []
            for t in reversed(range(horizon)):
                ret = rs[t] + c.gamma * cs[t] * ((1 - c.lam) * values[t + 1] + c.lam * ret)
                targets.append(ret)
            targets = torch.stack(targets[::-1])
        features = inputs[:-1].flatten(0, 1)
        targets = targets.flatten()
        count = features.shape[0]
        minibatch = count // c.critic_minibatches
        for _ in range(c.critic_epochs):
            permutation = torch.randperm(count, device=self.device)
            for i in range(c.critic_minibatches):
                ids = permutation[i * minibatch : (i + 1) * minibatch]
                critic_loss = (self.critic(features[ids]) - targets[ids]).square().mean()
                self.optimize("critic", critic_loss)
        return dict(
            actor_loss=actor_loss.detach(),
            critic_loss=critic_loss.detach(),
            survival=survival.detach().mean(),
            horizon=horizon,
        )

    def learn(self):
        start = self.started_at
        phase_times = {}
        phase_counts = {}

        class PhaseTimer:
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                self.started = time.perf_counter()
                return self

            def __exit__(self, *_):
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                phase_times[self.name] = phase_times.get(self.name, 0.0) + time.perf_counter() - self.started
                phase_counts[self.name] = phase_counts.get(self.name, 0) + 1

        self.collect(max(self.cfg.warmup_steps, self.cfg.wm_window))
        step_acc = 0.0
        for it in range(self.iteration, self.cfg.max_iterations):
            self.iteration = it
            lr = self.lr_at(it)
            for optimizer in self.optimizers.values():
                for group in optimizer.param_groups:
                    group["lr"] = lr
            step_acc += 1.0 / self.cfg.utd
            collect_steps = int(step_acc)
            step_acc -= collect_steps
            if self.cfg.profile:
                with PhaseTimer("collect") as phase:
                    stats = self.collect(collect_steps)
                with PhaseTimer("wm"):
                    for _ in range(self.cfg.wm_batches):
                        stats.update(self.fit_world_model())
                if it >= self.cfg.wm_only_iters:
                    with PhaseTimer("policy"):
                        stats.update(self.policy_update())
            else:
                stats = self.collect(collect_steps)
                for _ in range(self.cfg.wm_batches):
                    stats.update(self.fit_world_model())
                if it >= self.cfg.wm_only_iters:
                    stats.update(self.policy_update())
            if it % self.cfg.log_interval == 0:
                if self.cfg.holdout_fraction:
                    with torch.no_grad():
                        stats.update({"holdout/" + k: v for k, v in self.fit_world_model(True).items()})
                stats.update(
                    lr=lr,
                    wall_seconds=self.wall_offset + time.perf_counter() - start,
                    sim_steps=self.sim_steps,
                )
                scalar = {k: float(v) for k, v in stats.items()}
                if self.cfg.profile:
                    elapsed = max(sum(phase_times.values()), 1e-12)
                    scalar.update(
                        {
                            **{f"profile/seconds/{k}": v for k, v in phase_times.items()},
                            **{f"profile/calls/{k}": v for k, v in phase_counts.items()},
                            **{
                                f"profile/fraction/{k}": v / elapsed
                                for k, v in phase_times.items()
                            },
                        }
                    )
                self.writer.log_scalars(scalar, it)
                print(format_training_log(it, self.cfg.max_iterations, scalar), flush=True)
            if (it + 1) % self.cfg.save_interval == 0:
                self.save(self.log_dir / f"model_{it + 1}.pt", it + 1)
        self.save(self.log_dir / "model_final.pt", self.cfg.max_iterations)
        self.writer.close()

    def save(self, path, next_iteration=None):
        state = dict(
            format_version=3,
            torque_norm=self.torque_norm.state_dict(),
            config=self.cfg.to_dict(),
            iteration=self.iteration if next_iteration is None else next_iteration,
            sim_steps=self.sim_steps,
            context_dim=self.context_dim,
            contact_dim=self.contact_dim,
            modules={k: v.state_dict() for k, v in self.modules().items()},
            optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
            context_names=self.reader.context_names,
            joint_names=list(self.reader.robot.joint_names),
            wall_seconds=self.wall_offset + time.perf_counter() - self.started_at,
            common_step=self.env.common_step_counter,
            curriculum_scales={
                k: getattr(self.env, k)
                for k in (
                    "_adaptive_episode_curriculum",
                    "_reorient_success_curriculum",
                )
                if hasattr(self.env, k)
            },
            cuda_rng=torch.cuda.get_rng_state_all() if str(self.device).startswith("cuda") else None,
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        )
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def close(self):
        if hasattr(self, "episode_return_recorder"):
            self.episode_return_recorder.close()
        self.writer.close()
        self.reader.detach_capture()

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["config"].get("algorithm", "supermimic") != "supermimic":
            raise ValueError("Cannot resume PPO checkpoint as SuperMimic")
        if "utd" not in state["config"] or "critic_minibatches" not in state["config"]:
            raise ValueError(
                "Checkpoint predates the SuperMimic training recipe. Start a fresh run, "
                "or use --model-checkpoint to initialize only the world model and failure head."
            )
        if state["format_version"] != 3 or state["joint_names"] != list(self.reader.robot.joint_names):
            raise ValueError("Checkpoint schema/joint order mismatch")
        if state["context_names"] != self.reader.context_names:
            raise ValueError("Context layout mismatch")
        if state["config"]["control_hz"] != self.cfg.control_hz:
            raise ValueError("Checkpoint control frequency mismatch")
        if "torque_norm" in state:
            self.torque_norm.load_state_dict(state["torque_norm"])
        for k, mod in self.modules().items():
            mod.load_state_dict(state["modules"][k])
        for k, opt in self.optimizers.items():
            opt.load_state_dict(state["optimizers"][k])
        self.iteration = state["iteration"]
        self.sim_steps = state["sim_steps"]
        self.env.common_step_counter = state.get("common_step", 0)
        self.wall_offset = state.get("wall_seconds", 0.0)
        self.started_at = time.perf_counter()
        for key, value in state.get("curriculum_scales", {}).items():
            setattr(self.env, key, value)
        if state.get("cuda_rng") is not None and str(self.device).startswith("cuda"):
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        # Replay is intentionally refilled in the current randomized environments.
