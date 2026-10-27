"""CPU-only plots from complete real episodes; one panel per clock."""

import argparse, json, pathlib, math, csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("output")
a = p.parse_args()
rows = []
invalid_episodes = 0
for line in pathlib.Path(a.input).read_text().splitlines(keepends=True):
    if not line.endswith("\n"):
        continue  # A concurrently-written final line is not a completed record.
    r = json.loads(line)  # Malformed complete lines are errors, not discarded data.
    if r.get("type") != "episode":
        continue
    if not r.get("finite") or r.get("episode_return") is None:
        invalid_episodes += 1
        continue
    if not all(math.isfinite(r[k]) for k in ("episode_return", "elapsed_seconds", "interactions")):
        raise ValueError("Non-finite episode marked finite")
    rows.append(r)
if not rows:
    raise SystemExit("No complete finite episodes; no plot fabricated")
segments = sorted(set(r["segment"] for r in rows))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for segment in segments:
    subset = [r for r in rows if r["segment"] == segment]
    bins = {}
    for r in subset:
        bins.setdefault(int(r["elapsed_seconds"] // 300), []).append(r)
    groups = [bins[k] for k in sorted(bins)]
    means = [sum(r["episode_return"] for r in g) / len(g) for g in groups]
    for ax, key, scale in [(axes[0], "elapsed_seconds", 3600), (axes[1], "interactions", 1e6)]:
        ax.plot([max(r[key] for r in g) / scale for g in groups], means, ".-", label=segment)
        ax.set_ylabel("Mean completed-episode return")
        ax.grid(alpha=0.25)
axes[0].set_xlabel("Training elapsed hours")
axes[1].set_xlabel("Real simulator transitions (millions)")
fig.suptitle("Undiscounted episode return | 5-minute bins | incomplete episodes excluded")
fig.tight_layout()
fig.savefig(a.output, dpi=150)
with open(str(a.output) + ".csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=["segment", "episode_return", "elapsed_seconds", "interactions", "episode_steps"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in w.fieldnames})
print(
    json.dumps(
        {
            "complete_finite_episodes": len(rows),
            "invalid_episodes": invalid_episodes,
            "segments": len(segments),
            "output": a.output,
        }
    )
)
