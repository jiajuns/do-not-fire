"""ROS-agnostic core of the SVoI communication scheduler (portable to robots).

EventTriggeredGate: a per-robot leaky integrate-and-fire (LIF) gate that decides WHEN a
robot broadcasts its state to the team (distributed Pillar-A: de-synchronised event-
triggered communication). The trigger 'signal' (input current) is supplied by the node
(e.g. embedding drift since last broadcast, or pose change). Modes mirror the simulation
(periodic / spiking / evtrig / random) so hardware and sim share the same scheduler logic.

The centralised value-of-information link scheduler (Pillar-B, kWTA over directed links)
lives in a coordinator node; this file is the lightweight per-robot gate used on each robot.
"""
import math

import numpy as np


class PeriodicPhaseSchedule:
    """Wall-clock periodic schedule shared by independently started robots.

    ``start_time`` is a common Unix/ROS-system timestamp.  Robot ``k`` uses
    ``phase_sec=k * period_sec / n_agents``.  Late timer callbacks emit at most
    one event and skip missed slots, so a stalled process cannot burst packets.
    """

    def __init__(self, period_sec=4.0, phase_sec=0.0, start_time=0.0):
        self.period_sec = float(period_sec)
        if not math.isfinite(self.period_sec) or self.period_sec <= 0.0:
            raise ValueError("period_sec must be a positive finite number")
        self.phase_sec = float(phase_sec) % self.period_sec
        self.start_time = float(start_time)
        self.last_cycle = -1

    def reset(self, start_time=None):
        if start_time is not None:
            self.start_time = float(start_time)
        self.last_cycle = -1

    def due_cycle(self, now_sec):
        """Return the newly due cycle index, or ``None`` when nothing is due."""
        elapsed = float(now_sec) - self.start_time - self.phase_sec
        if elapsed < 0.0:
            return None
        cycle = int(math.floor(elapsed / self.period_sec + 1e-9))
        if cycle <= self.last_cycle:
            return None
        self.last_cycle = cycle
        return cycle


class EventTriggeredGate:
    def __init__(self, mode="spiking", threshold=1.0, leak=1.0, refractory=0,
                 interval=1, phase_offset=0, random_prob=0.2, evtrig_delta=0.3, seed=0):
        self.mode = mode
        self.threshold = float(threshold)
        self.leak = float(leak)
        self.refractory = int(refractory)
        self.interval = max(1, int(interval))
        self.phase_offset = int(phase_offset) % self.interval
        self.random_prob = float(random_prob)
        self.evtrig_delta = float(evtrig_delta)
        self.rng = np.random.default_rng(int(seed))
        self.reset()

    def reset(self):
        self.V = 0.0
        self.steps_since = 0
        self.refr_left = 0
        self.fire_count = 0
        self.step_count = 0

    def should_fire(self, step_idx, signal):
        """signal: a non-negative scalar 'novelty' current (e.g. embedding drift or pose
        delta) accumulated since the last broadcast. Returns True if the robot should
        broadcast its state this tick. Non-periodic modes seed at step 0; periodic
        mode honours its configured phase immediately."""
        self.step_count += 1
        s = float(max(0.0, signal))
        # Periodic mode must honour its configured phase from the first slot;
        # forcing every robot at step zero would add a synchronized burst to
        # the desynchronized condition. Other modes keep the seed packet.
        if step_idx == 0 and self.mode != "periodic":
            return self._fire()
        in_refr = self.refr_left > 0
        if self.mode == "periodic":
            fire = ((step_idx - self.phase_offset) % self.interval == 0)
        elif self.mode == "random":
            fire = (not in_refr) and (self.rng.random() < self.random_prob)
        elif self.mode == "evtrig":
            fire = (not in_refr) and (s >= self.evtrig_delta)
        else:  # spiking (LIF)
            self.V = self.leak * self.V + s
            fire = (not in_refr) and (self.V >= self.threshold)
        if fire:
            return self._fire()
        self.steps_since += 1
        if in_refr:
            self.refr_left -= 1
        return False

    def _fire(self):
        # soft reset on firing
        self.V = max(0.0, self.V - self.threshold)
        self.steps_since = 0
        self.refr_left = self.refractory
        self.fire_count += 1
        return True

    @property
    def fire_rate(self):
        return self.fire_count / max(1, self.step_count)


def embedding_drift(curr, prev):
    """Scale-invariant drift ||e_t - e_last|| / (||e_last|| + eps); +inf if no prev."""
    if prev is None:
        return float("inf")
    c = np.asarray(curr, dtype=np.float64).ravel()
    p = np.asarray(prev, dtype=np.float64).ravel()
    return float(np.linalg.norm(c - p) / (np.linalg.norm(p) + 1e-8))
