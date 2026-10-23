"""Explicit schedules for the 224-pixel experiments (no model dependencies)."""
import math


def parse_drop_epochs(value):
    drops = tuple(int(x.strip()) for x in value.split(',') if x.strip())
    if any(x < 1 for x in drops) or tuple(sorted(set(drops))) != drops:
        raise ValueError('replay_drop_epochs must be unique increasing epochs >= 1')
    return drops


def cosine_hold_lambda(horizon_steps, warm_steps, hold_steps=0, floor=0.015):
    """Hold backbone, warm up, cosine to a nonzero floor, then stay there.

    The horizon is independent of the run cap so extending/resuming does not
    rewrite the already-executed part of the schedule.
    """
    if not (0 <= floor <= 1 and 0 <= hold_steps < horizon_steps):
        raise ValueError('invalid floor or hold/horizon')
    if not 0 < warm_steps < horizon_steps - hold_steps:
        raise ValueError('warmup must fit inside the trainable schedule')

    def scale(step):
        t = step - hold_steps
        if t < 0:
            return 0.0
        if t < warm_steps:
            return (t + 1) / warm_steps
        u = min(1.0, (t - warm_steps) /
                (horizon_steps - hold_steps - warm_steps))
        return floor + (1 - floor) * (1 + math.cos(math.pi * u)) / 2
    return scale
