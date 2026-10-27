"""PPO tensor operations independent of simulator lifecycle."""

import math
import torch


def gae(reward, value, next_value, terminated, truncated, gamma, lam):
    advantage = torch.zeros_like(reward)
    trace = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * (~terminated[t]) * next_value[t] - value[t]
        trace = delta + gamma * lam * (~(terminated[t] | truncated[t])) * trace
        advantage[t] = trace
    return advantage, advantage + value


def log_prob(mean, action, std):
    return (-0.5 * ((action - mean) / std).square() - math.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)


def clipped_policy_loss(logratio, advantage, clip):
    ratio = logratio.exp()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
    return loss, ratio
