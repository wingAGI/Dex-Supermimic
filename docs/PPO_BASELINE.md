# Model-free PPO baseline

This fork adds a model-free PPO runner to
[CumulusAlpha/Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic),
based on upstream commit `47017662cf5b3f86ccb7a59fc078584b40e236ae`.
The task, assets, Actor and Critic architectures come from upstream. The PPO
runner, episode logger, tests and this comparison are additions in this fork;
they are not implementations or benchmark claims by the upstream author.

## Install and run

Follow the upstream NVIDIA/CUDA requirements. Install dependencies with the
committed lockfile:

```bash
uv sync --frozen
# Explicitly choose one mode; SuperMimic remains the default.
uv run dex-supermimic-train --algorithm supermimic --device cuda:0 \
  --num-envs 1024 --seed 42 --save-interval 100 --log-dir logs/supermimic
uv run dex-supermimic-train --algorithm ppo --device cuda:1 \
  --num-envs 1024 --seed 42 --save-interval 100 --log-dir logs/ppo
```

Run each command in its own terminal for a two-GPU comparison. Select available
physical CUDA devices; each trainer uses one GPU and does not use DDP. Default
training is 8000 iterations. The shorter save interval in these examples changes
checkpoint frequency, not the update equations. Logs and models stay local.

The shared CLI exposes `--help`. PPO-specific defaults are:

| Option | Default |
|---|---:|
| `--ppo-rollout-steps` | 32 |
| `--ppo-epochs` | 5 |
| `--ppo-minibatches` | 4 |
| `--ppo-clip` | 0.2 |
| `--ppo-value-coefficient` | 0.5 |
| `--ppo-grad-clip` | 1.0 |
| `--action-noise` (fixed Gaussian standard deviation) | 0.1 |
| `--gamma`, `--lam` | 0.99, 0.95 |
| `--lr` | 0.0002 |

There is no learned standard deviation, entropy bonus, value clipping, target-KL
stopping or adaptive learning-rate controller in this initial baseline. KL and
clip fraction are diagnostics; they do not trigger early stopping. Hyperparameters
have not been systematically tuned. World-model options (`wm_*`, `utd`, replay,
holdout, compilation) do not control PPO updates. PPO rejects WM-only checkpoint
initialization. `--amp` applies to the shared Critic's forward pass; PPO does not
compile its update loop.

## Algorithm and action semantics

The unchanged Actor returns a bounded mean. We sample an unbounded Gaussian
latent action around that mean, store its summed log probability over 20 action
dimensions, and clamp only the action passed to the environment to `[-1, 1]`.
PPO scores the original latent sample, not the clamped action. This is a
latent-Gaussian policy followed by a deterministic clamp, not a tanh-transformed
Gaussian density.

The collector records raw policy observations, privileged critic features,
actions, old log probabilities, real rewards and values. Observation statistics
are updated at the start of each rollout, then frozen across sampling and all
PPO epochs. Advantages are normalized across the entire rollout. The actor uses
the clipped PPO surrogate; the critic uses half squared error to GAE returns.
Actor and critic have independent optimizers and gradient clipping. A rollout has
`num_envs * ppo_rollout_steps` samples. Each epoch splits them into the requested
number of minibatches without dropping samples. With the defaults this gives
20 actor steps and 20 critic steps per iteration.

True terminations block bootstrap. Timeouts bootstrap the actual pre-reset
successor value, but both termination types stop the GAE trace across episodes.
The environment uses explicit manual resets (`auto_reset=False`). The runner
sets `env.max_common_steps` before sampling so native disturbance schedules work.
No world model, learned failure head, replay buffer or imagined transition is
constructed by the PPO runner.

## Returns, clocks and plots

`episode_returns.jsonl` contains one row for each completed real episode:

- undiscounted `episode_return`, including the terminal transition reward;
- environment/episode/segment IDs and episode length in control steps;
- global real simulator `interactions` and training `elapsed_seconds` at the end;
- `terminated`, `truncated` and individual termination reasons.

Non-finite episodes are retained with `finite=false` and a null return, not silently
turned into zeros. They must be reported separately. Partial episodes at the end
of a run are not counted as completed. A new resume segment starts from fresh
simulator episodes and does not stitch partial episodes across checkpoints.
Returns are recorded for all environment lanes, including the WM holdout lanes.
Upstream TensorBoard `reward` remains a mean per-step reward, not episode return.

```bash
uv run python tools/plot_episode_returns.py \
  logs/ppo/<timestamp>/episode_returns.jsonl ppo_returns.png
```

This writes two panels (return versus elapsed hours and versus real simulator
transitions) plus a CSV. Each point pools completed episodes in a five-minute
window, using sum of returns divided by episode count; segments stay separate.
The right panel uses those same windows at their actual interaction coordinates.
No episode-length multiplication or interpolation is used. Incomplete final bins
are allowed and contain only observed episodes.

Elapsed time includes task/model initialization for fresh CLI runs, rollout,
optimization and logging. It is not GPU busy time. On resume it is accumulated
training time and excludes the time the process was stopped. Real interactions
count only vectorized simulator steps times the number of environments; WM
imagined transitions are not added. Synchronous episode logging has overhead in
both modes, proportional to completed episode count.

## What the comparison does and does not isolate

This is a model-based versus model-free algorithm baseline, not a one-loss or
one-component causal ablation. The current defaults differ in all these ways:

| Setting | SuperMimic (upstream path) | PPO (added path) |
|---|---|---|
| Real sampling per iteration | Average 2.5 vector steps (`utd=0.4`) | 32 vector steps |
| Policy data | Replay starts and 32-step WM imagination | Current real on-policy rollout |
| Initial phase | 64 real warmup steps; 500 WM-only iterations | PPO from the first rollout |
| Dynamics fitting | 10 batches per iteration; 10% held-out envs | None; all lanes train PPO |
| Actor objective | Differentiable imagined reward plus critic bootstrap | Clipped likelihood-ratio objective |
| Critic targets | Imagined lambda returns | Real rollout GAE returns |
| Actor input normalization | Updated from real observations during collection | Updated once per rollout, frozen for PPO likelihood ratios |
| Critic input normalization | Updated from imagined critic inputs | Updated from current real features at rollout start |
| Actor clipping | Gradient norm 1, then per-element 0.1 | Gradient norm 1 only |
| Learning-rate schedule | Cosine in WM iterations | Same formula, in PPO iterations |
| Disturbance/curriculum time denominator | `64 + floor(max_iterations / utd)` | `max_iterations * ppo_rollout_steps` |
| Compile/precision work | Upstream compiled WM/imagination and AMP options | Eager PPO; shared Critic AMP option |

Consequently, equal iterations, equal elapsed time and equal real interactions
are three different comparisons. Even at equal interactions the disturbance
schedule may be at a different normalized stage, because the total-step
denominators differ. Matching task functions and a seed does not remove this
confound. Actor/Critic initialization is not explicitly paired: model construction
order differs, so equal seeds do not guarantee identical Critic initial weights.

The task, reward function and control chain are shared, but adaptive curricula
also respond to each policy's outcomes. Report episode length, termination versus
timeout counts, finite episode counts and reward/return definitions alongside
means. A longer surviving episode changes an undiscounted return's scale. Training
returns include exploration and are not the same as deterministic `play=True`
evaluation, which disables some randomization and changes the time limit. A single
seed and exploratory curves do not establish superiority, exact paper reproduction
or causal sample-efficiency gains. No experimental plots or pretrained weights
are included in this source-only fork.

## Checkpoints and validation

PPO uses format version 3 and preserves the shared Actor playback layout. Its
`modules` contain only `actor` and `critic`. Optimizers, normalizers, counters and
RNG states are saved. Resume must use the same algorithm and PPO/task settings;
logging paths and save intervals may change. Simulator state is not saved, so
resume is not a bitwise continuation of the original episodes. Upstream checkpoints
without an `algorithm` field are treated as SuperMimic; cross-algorithm resume is
rejected. To play a PPO checkpoint:

```bash
uv run dex-supermimic-play --checkpoint logs/ppo/<timestamp>/model_final.pt
```

CPU tests (no simulator instance or CUDA initialization):

```bash
uv run python -m unittest discover -s tests/dex_supermimic -p 'test_*.py' -v
```

Optional bounded native test on an available GPU:

```bash
uv run python tests/dex_supermimic/smoke_native.py \
  --device cuda:0 --output logs/ppo_smoke
```

On 2026-10-27, eight CPU unit tests passed and the bounded native test passed
on an A100 with Python 3.13 / Torch 2.10 / mjlab 1.6.0 / MuJoCo Warp 3.11.0.
The native test checks real rollouts, optimizer updates, checkpoint contents and
same-algorithm resume. It does not validate convergence or benchmark fairness.
Only load checkpoints from a source you trust: playback/resume uses PyTorch's
full checkpoint deserialization.
