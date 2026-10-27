"""CPU unit tests: no simulator instance or CUDA initialization."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg
from dex_supermimic.rl.dex_supermimic.episode_returns import EpisodeReturnRecorder
from dex_supermimic.rl.dex_supermimic.ppo_math import gae, log_prob, clipped_policy_loss


class PPOMathTests(unittest.TestCase):
    def test_terminal_and_timeout_have_different_bootstrap(self):
        rewards = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
        values = torch.zeros_like(rewards)
        next_values = torch.full_like(rewards, 10.0)
        terminated = torch.tensor([[True, False], [False, False]])
        truncated = torch.tensor([[False, True], [False, False]])
        adv, returns = gae(rewards, values, next_values, terminated, truncated, 0.9, 0.95)
        torch.testing.assert_close(adv[0], torch.tensor([1.0, 10.0]))
        torch.testing.assert_close(returns, adv)
        # A new episode's large reward must not leak back over either boundary.

    def test_uninterrupted_trace(self):
        r = torch.tensor([[1.0], [2.0]])
        v = torch.zeros_like(r)
        z = torch.zeros_like(r, dtype=torch.bool)
        adv, _ = gae(r, v, v, z, z, 0.9, 0.8)
        torch.testing.assert_close(adv[:, 0], torch.tensor([2.44, 2.0]))

    def test_gaussian_density_matches_torch_and_sums_action_dims(self):
        mean = torch.randn(7, 20)
        latent = mean + torch.randn_like(mean) * 0.1
        expected = torch.distributions.Normal(mean, 0.1).log_prob(latent).sum(-1)
        torch.testing.assert_close(log_prob(mean, latent, 0.1), expected)
        torch.testing.assert_close((log_prob(mean, latent, 0.1) - expected).exp(), torch.ones(7))

    def test_latent_action_is_not_replaced_with_clipped_action(self):
        mean = torch.zeros(1, 20)
        latent = torch.full_like(mean, 2.0)
        self.assertFalse(torch.equal(log_prob(mean, latent, 0.1), log_prob(mean, latent.clamp(-1, 1), 0.1)))

    def test_clipping_for_positive_and_negative_advantages(self):
        # Positive advantage caps the gain at ratio 1.2; negative advantage
        # caps the gain at ratio .8, following PPO's pessimistic surrogate.
        loss, ratio = clipped_policy_loss(torch.tensor([math.log(2.0), math.log(0.5)]), torch.tensor([1.0, -1.0]), 0.2)
        torch.testing.assert_close(loss, torch.tensor(-0.2))
        torch.testing.assert_close(ratio, torch.tensor([2.0, 0.5]))


class ConfigurationTests(unittest.TestCase):
    def test_default_stays_supermimic(self):
        self.assertEqual(DexSuperMimicCfg().algorithm, "supermimic")
        # PPO-only constraints must not reject an upstream WM setting.
        DexSuperMimicCfg(action_noise=0)

    def test_bad_ppo_options_fail_early(self):
        for kw in [
            dict(ppo_epochs=0),
            dict(ppo_minibatches=10**9),
            dict(action_noise=0),
            dict(ppo_grad_clip=float("nan")),
            dict(ppo_clip=1),
            dict(model_checkpoint="model.pt"),
            dict(reset_model_statistics=True),
        ]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                DexSuperMimicCfg(algorithm="ppo", **kw)


class EpisodeReturnTests(unittest.TestCase):
    def test_independent_lanes_terminal_reward_timeout_and_invalid_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = EpisodeReturnRecorder(folder, 2, "cpu")

            def step(r, term, trunc, i):
                recorder.record(
                    torch.tensor(r),
                    torch.tensor(term),
                    torch.tensor(trunc),
                    interactions=i,
                    elapsed_seconds=i / 2,
                    reasons={},
                )

            step([1.0, 10.0], [False, False], [False, False], 2)
            step([2.0, 20.0], [True, False], [False, False], 4)
            step([4.0, 30.0], [False, False], [False, True], 6)
            step([float("nan"), 1.0], [True, False], [False, False], 8)
            recorder.close()
            rows = [json.loads(line) for line in (Path(folder) / "episode_returns.jsonl").read_text().splitlines()][1:]
            self.assertEqual([r["episode_return"] for r in rows], [3.0, 60.0, None])
            self.assertEqual([r["episode_steps"] for r in rows], [2, 3, 2])
            self.assertEqual([r["interactions"] for r in rows], [4, 6, 8])
            self.assertEqual(rows[2]["episode_id"], 1)
            self.assertTrue(rows[1]["truncated"])
            self.assertFalse(rows[2]["finite"])


def tearDownModule():
    assert not torch.cuda.is_initialized(), "CPU tests unexpectedly initialized CUDA"


if __name__ == "__main__":
    unittest.main()
