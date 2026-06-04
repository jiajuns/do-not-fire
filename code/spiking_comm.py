"""
Bio-inspired spiking (Leaky Integrate-and-Fire) event-triggered communication gate.

Replaces fixed-interval (time-triggered) broadcasting with event-triggered firing:
each robot integrates an "information potential" on a virtual membrane and only
broadcasts its learned 64-D state embedding when the potential crosses a threshold,
then resets (soft reset + leak + refractory). This is the test-time control module
that decides WHEN a robot communicates; it does not change WHAT is broadcast or the
planning policy.

Modes (selected by test_parameter.COMM_MODE):
  - "periodic"        : baseline, time-triggered firing every COMMUNICATION_INTERVAL steps
  - "spiking"         : fixed LIF gate, deterministic threshold crossing (no training)
  - "spiking_learned" : tiny learned gate net emits a firing probability (REINFORCE)
  - "random"          : Bernoulli firing at a fixed rate (equal-budget baseline)

Step 0 always fires for every robot to seed receiver mailboxes (agent.get_stacked_msg
reads msgs[idx][-1] and would fail on an empty mailbox).
"""

import os
import numpy as np
import torch
import torch.nn as nn

import test_parameter as tp


_EPS = 1e-8


def _periodic_should_fire(step_index, interval=None):
    """Periodic (time-triggered) baseline gate: fire when the step index is a multiple of the
    interval (default COMMUNICATION_INTERVAL). Re-implemented locally so this module carries no
    external scheduling dependency. Step 0 always fires (0 % n == 0)."""
    if interval is None:
        interval = int(tp.COMMUNICATION_INTERVAL)
    interval = max(1, int(interval))
    return step_index % interval == 0


def normalized_drift(curr, prev):
    """Scale-invariant per-step embedding drift: ||e_t - e_{t-1}|| / (||e_{t-1}|| + eps).

    curr/prev are tensors broadcastable to a flat vector (e.g. shape (1,1,64)).
    Returns a python float. If prev is None, returns +inf (forces a spike).
    """
    if prev is None:
        return float("inf")
    c = curr.reshape(-1).float()
    p = prev.reshape(-1).float()
    return float(torch.linalg.norm(c - p) / (torch.linalg.norm(p) + _EPS))


class GateNet(nn.Module):
    """Tiny per-robot firing-probability head used by the learned gate.

    Input features (default 3): [normalized_drift, membrane_potential, steps_since_last].
    Output: a single logit; firing probability = sigmoid(logit).
    """

    def __init__(self, in_dim=3, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class CommGate:
    """Stateful communication gate for a team of n_agents robots.

    Holds, per robot: membrane potential V, previous-step embedding, steps since the
    last broadcast, and refractory countdown. One instance per episode; call
    reset_episode() at episode start.
    """

    def __init__(self, n_agents, mode="periodic", device="cpu",
                 threshold=1.0, leak=1.0, refractory=0,
                 w_drift=1.0, w_cov=0.0, w_util=0.0,
                 random_prob=0.2, gate_net=None, rng=None, evtrig_delta=40.0, interval=1):
        self.n = int(n_agents)
        self.mode = mode
        self.device = device
        self.interval = max(1, int(interval))  # period for periodic/staggered modes
        self.stagger_pattern = os.environ.get("BLMRE_STAGGER_PATTERN", "even")  # even (optimal) | half
        self.evtrig_delta = float(evtrig_delta)  # distance-triggered baseline threshold (meters)
        self.threshold = float(threshold)
        self.leak = float(leak)
        self.refractory = int(refractory)
        self.w_drift = float(w_drift)
        self.w_cov = float(w_cov)
        self.w_util = float(w_util)
        self.random_prob = float(random_prob)
        self.gate_net = gate_net
        self.rng = rng if rng is not None else np.random.default_rng(int(getattr(tp, "SEED", 0)))

        # training hook: collected (logprob, was_forced) per sampled decision
        self.collect_logprobs = False
        self.logprobs = []

        self.reset_episode()

    def reset_episode(self):
        self.V = np.zeros(self.n, dtype=np.float64)
        self.prev_emb = [None] * self.n
        self.steps_since_last = np.zeros(self.n, dtype=np.int64)
        self.refractory_left = np.zeros(self.n, dtype=np.int64)
        self.pos_last = [None] * self.n  # last-broadcast position (evtrig / cwd believed position)
        self.prev_coverage = 0.0
        # contention-weighted de-synchronisation (cwd) state
        self._cwd_step = -1
        self._cwd_fire = set()
        self._cwd_budget = 0.0
        self._cwd_sigma = float(getattr(tp, "VOI_SIGMA", 50.0))
        # value- and loss-aware Whittle (vlw) state
        self._eff_stale = np.zeros(self.n, dtype=np.float64)  # steps since last DELIVERED broadcast
        self._value = self._compute_value()                   # per-robot informativeness weight
        # self-organising de-synchronisation (desync/stagfix) + per-robot clock-drift state
        _P = float(self.interval)
        self._desync_alpha = float(os.environ.get("BLMRE_DESYNC_ALPHA", "0.95"))
        self._drift = float(os.environ.get("BLMRE_CLOCK_DRIFT", "0.0"))  # std of per-robot clock-rate deviation
        if self._drift > 0:
            self._rate = np.clip(1.0 + self.rng.normal(0.0, self._drift, self.n), 0.5, 1.5)
        else:
            self._rate = np.ones(self.n, dtype=np.float64)
        _init = os.environ.get("BLMRE_DESYNC_INIT", "even")
        if _init == "sync":
            # realistic no-coordinator cold start: robots boot ~synchronised (small boot-time jitter
            # only); DESYNC must self-organise AWAY from synchrony to even spacing using overhearing.
            self._theta = self.rng.uniform(0.0, 0.1 * _P, self.n).astype(np.float64)
        else:
            self._theta = (np.arange(self.n) * _P / max(1, self.n)).astype(np.float64)  # even-stagger init
        self._fire_time = np.full(self.n, -1.0, dtype=np.float64)  # last broadcast step per robot
        # per-episode trace for diagnostics/plots (membrane + spikes)
        self.trace = {"V": [[] for _ in range(self.n)],
                      "drift": [[] for _ in range(self.n)],
                      "spike": [[] for _ in range(self.n)]}
        self.logprobs = []

    def _compute_value(self):
        """Per-robot informativeness weight for value-aware scheduling (vlw). Under heterogeneous
        sensing a larger sensor range observes more area, so that robot's updates carry more team
        value. Read from the same BLMRE_HETERO_SRANGE string the environment uses (e.g. "10,30");
        BLMRE_VALUE_POW controls value vs range (2.0 = sensor-coverage area). Homogeneous teams get
        equal weight 1, so the value-weighted index degrades gracefully to freshness-only (= even
        staggering, Proposition 3)."""
        het = os.environ.get("BLMRE_HETERO_SRANGE", "")
        pow_ = float(os.environ.get("BLMRE_VALUE_POW", "2.0"))
        if het:
            try:
                r = [float(x) for x in het.split(",") if x.strip() != ""]
            except ValueError:
                r = []
            if r:
                v = np.array([r[i % len(r)] ** pow_ for i in range(self.n)], dtype=np.float64)
                m = v.mean()
                if m > 1e-9:
                    return v / m  # normalise to mean 1 (relative informativeness)
        return np.ones(self.n, dtype=np.float64)

    # ---- input current from cheap local signals -------------------------------
    def _input_current(self, robot_id, embedding, coverage, utility):
        drift = normalized_drift(embedding, self.prev_emb[robot_id])
        cov_gain = max(0.0, float(coverage) - self.prev_coverage)
        util_mass = 0.0
        if utility is not None and self.w_util != 0.0:
            u = np.asarray(utility, dtype=np.float64)
            util_mass = float(np.clip(u, 0.0, None).sum())
        # drift may be +inf at the first observation; keep it finite for integration
        drift_eff = drift if np.isfinite(drift) else 0.0
        current = self.w_drift * drift_eff + self.w_cov * cov_gain + self.w_util * util_mass
        return current, drift

    def _corraware_phase(self, robot_id, position):
        """Correlation-aware staggering: assign broadcast phases so that spatially NEAR (hence more
        correlated) robots are maximally separated in phase, rather than by arbitrary index. Phases
        are computed once from the step-0 positions (all robots fire at step 0), by sorting robots
        on a 1-D spatial key and assigning evenly spaced slots through a coprime stride so that
        spatially adjacent robots land ~T/2 apart. Falls back to even index staggering until ready."""
        if getattr(self, "_ca_phase", None) is None:
            if not hasattr(self, "_ca_pos"):
                self._ca_pos = {}
            if position is not None and robot_id not in self._ca_pos:
                self._ca_pos[robot_id] = np.asarray(position, dtype=np.float64).reshape(-1)[:2]
            if len(self._ca_pos) >= self.n:
                ids = sorted(self._ca_pos.keys())
                P = np.array([self._ca_pos[i] for i in ids])
                key = P[:, 0] + 1e-3 * P[:, 1]  # 1-D spatial sort key
                order = list(np.argsort(key))   # indices into ids, in spatial order
                s = self.n // 2
                while s > 1 and np.gcd(int(s), int(self.n)) != 1:
                    s -= 1
                s = max(1, s)
                self._ca_phase = {}
                for rank, pos_in_ids in enumerate(order):
                    slot = (rank * s) % self.n
                    self._ca_phase[ids[pos_in_ids]] = int(round(slot * self.interval / self.n))
        if getattr(self, "_ca_phase", None):
            return self._ca_phase.get(robot_id, int(round(robot_id * self.interval / max(1, self.n))))
        return 0  # before init: only step 0, which is forced anyway

    # ---- main decision --------------------------------------------------------
    def should_broadcast(self, robot_id, step_idx, embedding,
                         coverage=0.0, utility=None, position=None):
        """Return True if robot `robot_id` should broadcast at this step.

        Updates membrane state. Handles all modes. Always True at step 0.
        """
        if self.mode == "periodic":
            fire = _periodic_should_fire(step_idx)
            self._record_trace(robot_id, 0.0, 0.0, fire)
            return bool(fire)

        if self.mode == "staggered":
            # de-synchronised periodic: same per-robot rate and total budget as 'periodic',
            # but robots' broadcast phases are spread so the team never goes blind together.
            # Isolates SYNCHRONISATION from comm volume (the decisive Pillar-A control).
            forced = (step_idx == 0)  # seed all mailboxes once
            if self.stagger_pattern == "corraware":
                phase = self._corraware_phase(robot_id, position)  # separate spatially-near robots in phase
            elif self.stagger_pattern == "half":
                phase = 0 if robot_id < self.n // 2 else self.interval // 2  # 2 clusters: suboptimal de-sync
            else:  # even spacing: bound-optimal for homogeneous teams (Proposition 3)
                phase = int(round(robot_id * self.interval / max(1, self.n)))
            fire = forced or ((step_idx - phase) % self.interval == 0)
            self._record_trace(robot_id, 0.0, 0.0, fire)
            return bool(fire)

        if self.mode in ("desync", "stagfix"):
            # Decentralised de-synchronisation under clock drift / no coordinator. Each robot holds a
            # continuous phase theta in [0,P) advancing by its OWN (drifting) clock rate; it broadcasts
            # when theta wraps past P. Both modes start from even-stagger phases.
            #   stagfix = centralised even offsets set once, then left to drift (no re-adaptation).
            #   desync  = DESYNC (Degesys et al., IPSN 2007): on each fire, nudge theta toward the
            #             midpoint of the two phase-neighbours' fire times, recovering even T/N spacing
            #             from pure overhearing -- NO global clock, NO knowledge of N. The midpoint rule
            #             continuously re-converges as clocks drift / membership changes, where the
            #             fixed schedule's even spacing collapses (toward synchronisation).
            P = float(self.interval)
            if step_idx == 0:
                self._fire_time[robot_id] = 0.0  # seed all mailboxes once
                self._record_trace(robot_id, 0.0, 0.0, True)
                return True
            self._theta[robot_id] += self._rate[robot_id]
            fire = self._theta[robot_id] >= P
            if fire:
                self._theta[robot_id] -= P
                if self.mode == "desync" and self.n >= 2:
                    ago = step_idx - self._fire_time            # steps since each robot last fired (>=0)
                    ago[robot_id] = np.inf
                    prev_j = int(np.argmin(ago))                # fired most recently -> phase just behind
                    ago[robot_id] = -np.inf
                    next_j = int(np.argmax(ago))                # fired longest ago -> fires soonest next
                    gap_prev = float(step_idx - self._fire_time[prev_j])
                    gap_next = float(P - (step_idx - self._fire_time[next_j]))
                    mid = (gap_next - gap_prev) / 2.0           # signed offset of i from the midpoint
                    self._theta[robot_id] = float(np.clip(self._theta[robot_id] - self._desync_alpha * mid, -P, P))
                self._fire_time[robot_id] = float(step_idx)
            self._record_trace(robot_id, 0.0, 0.0, fire)
            return bool(fire)

        if self.mode == "cwd":
            # Contention-weighted de-synchronisation: a Max-Weight / Whittle-index broadcast policy.
            # Each step we refresh the robots that are both stale and currently contending, with
            # contention_i = sum_j exp(-||p_i-p_j||^2 / 2 sigma^2) over believed teammate positions.
            # The budget matches the periodic rate (N/T broadcasts per step) via a token bucket. In
            # the homogeneous limit (equal contention) this reduces to refreshing the stalest, i.e.
            # even staggering (Proposition 3); under heterogeneous contention it adapts.
            forced = (step_idx == 0)
            if self.pos_last[robot_id] is None and position is not None:
                self.pos_last[robot_id] = np.asarray(position, dtype=np.float64).reshape(-1)[:2]
            if forced:
                self.steps_since_last[robot_id] = 0
                self._record_trace(robot_id, 0.0, 0.0, True)
                return True
            if step_idx != self._cwd_step:
                self._cwd_step = step_idx
                self._cwd_budget += self.n / max(1, self.interval)
                k = int(self._cwd_budget)
                self._cwd_budget -= k
                if k > 0 and all(p is not None for p in self.pos_last):
                    P = np.asarray(self.pos_last, dtype=np.float64)
                    idx = np.empty(self.n)
                    for i in range(self.n):
                        d2 = np.sum((P - P[i]) ** 2, axis=1)
                        cont = float(np.sum(np.exp(-d2 / (2.0 * self._cwd_sigma ** 2))) - 1.0)
                        idx[i] = self.steps_since_last[i] * max(cont, 1e-6)
                    self._cwd_fire = set(int(j) for j in np.argsort(-idx)[:k])
                else:
                    self._cwd_fire = set()
            fire = robot_id in self._cwd_fire
            if fire:
                if position is not None:
                    self.pos_last[robot_id] = np.asarray(position, dtype=np.float64).reshape(-1)[:2]
                self.steps_since_last[robot_id] = 0
            else:
                self.steps_since_last[robot_id] += 1
            self._record_trace(robot_id, 0.0, 0.0, fire)
            return bool(fire)

        if self.mode == "submod":
            # Joint submodular de-synchronisation: schedule on the JOINT marginal reduction of team
            # map-overlap, not a separable per-arm index. The redundant-coverage risk lives on PAIRS
            # r_ij = kappa(p_i,p_j)*max(tau_i,tau_j); refreshing robot i resolves all pairs incident
            # to it. Selecting the k robots that cover the most pair-risk is monotone submodular
            # (weighted max-coverage), so greedy is (1-1/e)-optimal and, unlike the per-arm index,
            # avoids refreshing two robots whose contributions overlap. Budget matched via token bucket.
            forced = (step_idx == 0)
            if self.pos_last[robot_id] is None and position is not None:
                self.pos_last[robot_id] = np.asarray(position, dtype=np.float64).reshape(-1)[:2]
            if forced:
                self.steps_since_last[robot_id] = 0
                self._record_trace(robot_id, 0.0, 0.0, True)
                return True
            if step_idx != self._cwd_step:
                self._cwd_step = step_idx
                self._cwd_budget += self.n / max(1, self.interval)
                k = int(self._cwd_budget)
                self._cwd_budget -= k
                if k > 0 and all(p is not None for p in self.pos_last):
                    P = np.asarray(self.pos_last, dtype=np.float64)
                    tau = self.steps_since_last.astype(np.float64)
                    D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
                    Kk = np.exp(-D2 / (2.0 * self._cwd_sigma ** 2))
                    np.fill_diagonal(Kk, 0.0)
                    R = Kk * np.maximum(tau[:, None], tau[None, :])  # symmetric pair risk
                    covered = np.zeros((self.n, self.n), dtype=bool)
                    selected = []
                    for _ in range(min(k, self.n)):
                        gains = np.where(covered, 0.0, R).sum(axis=1)
                        for s in selected:
                            gains[s] = -1.0
                        bi = int(np.argmax(gains))
                        selected.append(bi)
                        covered[bi, :] = True
                        covered[:, bi] = True
                    self._cwd_fire = set(selected)
                else:
                    self._cwd_fire = set()
            fire = robot_id in self._cwd_fire
            if fire:
                if position is not None:
                    self.pos_last[robot_id] = np.asarray(position, dtype=np.float64).reshape(-1)[:2]
                self.steps_since_last[robot_id] = 0
            else:
                self.steps_since_last[robot_id] += 1
            self._record_trace(robot_id, 0.0, 0.0, fire)
            return bool(fire)

        if self.mode == "vlw":
            # Value- and loss-aware Whittle index for HETEROGENEOUS value on an UNRELIABLE channel.
            #   idx_i = value_i * eff_stale_i
            # eff_stale_i = steps since robot i's last DELIVERED broadcast, so a dropped send keeps i
            # stale and high-priority (implicit retransmission); value_i is i's informativeness
            # (sensor-coverage weight). Budget is matched to the periodic rate via a token bucket, so
            # comm volume equals even-staggering and only the ALLOCATION differs. Theory: with
            # heterogeneous cost the Whittle-optimal cycle is asymmetric (serves the more-informative
            # source >1x/cycle) whenever the asymmetry is large, so round-robin is strictly suboptimal
            # (Tripathi-Modiano, TNET 2024); under loss the closed-form AoII/age Whittle index beats
            # round-robin (Ayik-Ceran-Uysal 2023). Under homogeneous value + reliable channel it
            # collapses to refreshing the stalest = even staggering, preserving the de-sync baseline.
            forced = (step_idx == 0)
            if forced:
                self._eff_stale[robot_id] = 0.0
                self._record_trace(robot_id, 0.0, 0.0, True)
                return True
            if step_idx != self._cwd_step:
                self._cwd_step = step_idx
                self._eff_stale += 1.0  # age every robot one step since its last DELIVERED update
                self._cwd_budget += self.n / max(1, self.interval)
                k = int(self._cwd_budget)
                self._cwd_budget -= k
                if k > 0:
                    idx = self._value * self._eff_stale
                    self._cwd_fire = set(int(j) for j in np.argsort(-idx)[:k])
                else:
                    self._cwd_fire = set()
            fire = robot_id in self._cwd_fire
            # eff_stale is reset on DELIVERY (notify_outcome), NOT on fire, so a lost send stays stale
            self._record_trace(robot_id, 0.0, 0.0, fire)
            return bool(fire)

        if self.mode == "owdesync":
            # OW-Desync: overlap-weighted, budget-constrained, conflict-free activation on a RANGE-LIMITED
            # interference graph. Each step pick the max-weight INDEPENDENT SET (no two in-range
            # transmitters fire together) under the token-bucket budget, weight = value_i * eff_stale_i
            # (value = informativeness from heterogeneous sensing; eff_stale = AoI). Generalises
            # even-staggering: symmetric value + complete graph => uniform => even-staggering (Cor 1);
            # heterogeneous value => high-value robots fire more (Prop 2). Conflict edge iff within the
            # interference range 2*R_comm (so the chosen set is collision-free by construction).
            forced = (step_idx == 0)
            if self.pos_last[robot_id] is None and position is not None:
                self.pos_last[robot_id] = np.asarray(position, dtype=np.float64).reshape(-1)[:2]
            if forced:
                self._eff_stale[robot_id] = 0.0
                self._record_trace(robot_id, 0.0, 0.0, True)
                return True
            if step_idx != self._cwd_step:
                self._cwd_step = step_idx
                self._eff_stale += 1.0
                self._cwd_budget += self.n / max(1, self.interval)
                k = int(self._cwd_budget); self._cwd_budget -= k
                Rc = float(os.environ.get("BLMRE_R_COMM", "0") or 0.0)
                if k > 0 and all(p is not None for p in self.pos_last):
                    P = np.asarray(self.pos_last, dtype=np.float64)
                    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
                    conf = (D < 2.0 * Rc) & ~np.eye(self.n, dtype=bool) if Rc > 0 else np.zeros((self.n, self.n), bool)
                    wt = self._value * self._eff_stale                      # overlap-value x AoI
                    avail = np.ones(self.n, dtype=bool); fire = set()
                    for i in np.argsort(-wt):
                        if not avail[i]: continue
                        fire.add(int(i)); avail[i] = False; avail[conf[i]] = False
                        if len(fire) >= k: break
                    self._cwd_fire = fire
                else:
                    self._cwd_fire = set()
            fire = robot_id in self._cwd_fire
            if fire:
                if position is not None:
                    self.pos_last[robot_id] = np.asarray(position, dtype=np.float64).reshape(-1)[:2]
                self._eff_stale[robot_id] = 0.0
            self._record_trace(robot_id, 0.0, 0.0, fire)
            return bool(fire)

        if self.mode == "evtrig":
            # classic distance-triggered communication baseline:
            # broadcast when moved >= delta from last-broadcast position
            forced = (step_idx == 0)
            last = self.pos_last[robot_id]
            d = float("inf") if last is None else float(np.linalg.norm(np.asarray(position) - last))
            in_refr = self.refractory_left[robot_id] > 0
            fire = forced or (d >= self.evtrig_delta and not in_refr)
            self._record_trace(robot_id, 0.0, d if np.isfinite(d) else 0.0, fire)
            if fire:
                self.pos_last[robot_id] = np.asarray(position).copy()
                self.steps_since_last[robot_id] = 0
                self.refractory_left[robot_id] = self.refractory
            else:
                self.steps_since_last[robot_id] += 1
                if in_refr:
                    self.refractory_left[robot_id] -= 1
            return bool(fire)

        forced = (step_idx == 0)
        current, drift = self._input_current(robot_id, embedding, coverage, utility)

        # leaky integrate
        self.V[robot_id] = self.leak * self.V[robot_id] + current
        in_refractory = self.refractory_left[robot_id] > 0

        if forced:
            fire = True
            logp = None
        elif self.mode == "random":
            fire = (not in_refractory) and (self.rng.random() < self.random_prob)
            logp = None
        elif self.mode == "spiking":
            fire = (not in_refractory) and (self.V[robot_id] >= self.threshold)
            logp = None
        elif self.mode == "spiking_learned":
            fire, logp = self._learned_decision(robot_id, drift, in_refractory)
        else:
            raise ValueError(f"unknown COMM_MODE: {self.mode}")

        if self.collect_logprobs and logp is not None:
            self.logprobs.append(logp)

        self._record_trace(robot_id, self.V[robot_id], drift if np.isfinite(drift) else 0.0, fire)

        # advance per-robot bookkeeping
        self.prev_emb[robot_id] = embedding.detach().clone() if hasattr(embedding, "detach") else embedding
        if fire:
            self._reset_on_fire(robot_id)
        else:
            self.steps_since_last[robot_id] += 1
            if in_refractory:
                self.refractory_left[robot_id] -= 1
        return bool(fire)

    def notify_outcome(self, robot_id, delivered, step_idx=None):
        """Channel delivery feedback (ACK). The value/loss-aware scheduler (vlw) resets a robot's
        effective staleness ONLY when its broadcast was actually delivered; a dropped send leaves it
        stale so the Whittle index re-prioritises it (retransmission). No-op for other modes."""
        if self.mode == "vlw" and delivered:
            self._eff_stale[robot_id] = 0.0

    def _learned_decision(self, robot_id, drift, in_refractory):
        # refractory hard-blocks firing; don't sample/learn on a forced-silent step
        if in_refractory:
            return False, None
        feats = torch.tensor(
            [float(drift if np.isfinite(drift) else 0.0),
             float(self.V[robot_id]),
             float(self.steps_since_last[robot_id])],
            dtype=torch.float32, device=self.device,
        )
        logit = self.gate_net(feats)
        prob = torch.sigmoid(logit)
        if self.collect_logprobs:
            dist = torch.distributions.Bernoulli(probs=prob)
            sample = dist.sample()
            fire = bool(sample.item())
            logp = dist.log_prob(sample)
        else:
            fire = (prob.item() >= 0.5)
            logp = None
        return fire, logp

    def _reset_on_fire(self, robot_id):
        # soft reset: subtract threshold (keeps residual potential), refractory hold
        self.V[robot_id] = max(0.0, self.V[robot_id] - self.threshold)
        self.steps_since_last[robot_id] = 0
        self.refractory_left[robot_id] = self.refractory

    def _record_trace(self, robot_id, v, drift, fire):
        self.trace["V"][robot_id].append(float(v))
        self.trace["drift"][robot_id].append(float(drift))
        self.trace["spike"][robot_id].append(int(bool(fire)))

    def end_step(self, coverage):
        """Call once per simulation step after all robots decided, to update the
        shared coverage baseline used for the coverage-gain signal."""
        self.prev_coverage = float(coverage)

    def pop_logprobs(self):
        lp = self.logprobs
        self.logprobs = []
        return lp

    def summary(self):
        """Per-episode broadcast diagnostics for logging/plots.

        Works in every mode (in periodic mode spike_total equals the broadcast
        count and mean_broadcast_interval equals the fixed interval).
        """
        spikes = self.trace["spike"]
        per_robot = [int(np.sum(s)) for s in spikes]
        steps = max((len(s) for s in spikes), default=0)
        intervals = []
        for s in spikes:
            idx = [t for t, v in enumerate(s) if v]
            intervals += [b - a for a, b in zip(idx[:-1], idx[1:])]
        return {
            "comm_mode": self.mode,
            "spike_total": int(sum(per_robot)),
            "spike_per_robot_mean": float(np.mean(per_robot)) if per_robot else 0.0,
            "mean_broadcast_interval": float(np.mean(intervals)) if intervals else float(steps),
            "broadcast_steps": int(steps),
        }

    def trace_arrays(self):
        """Return per-robot membrane/drift/spike traces as numpy arrays (for plots)."""
        return {k: [np.asarray(v) for v in self.trace[k]] for k in ("V", "drift", "spike")}


class LinkScheduler:
    """k-Winners-Take-All spiking scheduler for TARGETED (pairwise) state sharing
    under a hard per-step communication budget.

    Models a limited shared channel: at most ``budget`` directed links (sender->receiver)
    may transmit per step. Each directed link is a "link neuron" whose input current is
    its value of information VoI(i->j) -- how much refreshing j's knowledge of i would
    correct j's belief, scaled by how much that correction matters for coordination.
    Lateral inhibition (kWTA) lets only the top-``budget`` fire; on a silent link the
    receiver keeps the sender's stale position/embedding.

    VoI signal (the key design choice): the *belief error* be(i->j) = || true_pos(i) -
    pos_j(i) ||, i.e. how wrong receiver j currently is about sender i. This grows with
    both staleness AND the sender's motion, and is ~0 for a sender j already tracks well
    (e.g. a stationary teammate) -- so budget is never wasted refreshing correct beliefs.
    The default mode multiplies it by a proximity kernel (a close, contending teammate
    matters more for overlap-avoidance than a distant one).

    This is a centralized, base-station-aided scheduler (it reads true positions for the
    VoI term); a decentralized last-known-position variant is left to future work.

    modes:
      voi          : top-B by belief_error * proximity        (ours, instantaneous)
      voi_lif      : LIF integrate-and-fire of (be*prox + floor) current, soft reset
                     (ours, theorem-backed: floor>0 => starvation-free, bounded staleness)
      err          : top-B by belief_error only               (ablation: how-wrong, ignore who)
      prox         : top-B by proximity only                  (ablation: who, ignore how-wrong)
      stale        : top-B by staleness only                  (ablation: when, ignore motion)
      round_robin  : cycle a fixed link order, B per step      (structured budget baseline)
      random       : B uniformly-random links per step         (naive budget baseline)
    """

    def __init__(self, n_agents, budget, mode="voi", sigma=50.0, leak=1.0, floor=0.0,
                 weights=None, rng=None):
        self.n = int(n_agents)
        self.budget = int(budget)
        self.mode = mode
        self.sigma = float(sigma)
        self.leak = float(leak)
        self.floor = float(floor)  # constant input current per link (LIF); >0 => no starvation
        # learned/optimised linear VoI: score = w . [be/50, prox, (be/50)*prox, stale/5, dist/50]
        # default [0,0,1,0,0] reproduces the hand-set be*prox ranking (mode 'wsum')
        self.weights = (np.asarray(weights, dtype=np.float64) if weights is not None
                        else np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.links = [(i, j) for i in range(self.n) for j in range(self.n) if i != j]
        # decision_voi: filled by TestWorker each step with the counterfactual per-link score
        self.precomputed_score = None
        self.reset()

    def reset(self):
        L = len(self.links)
        self.staleness = np.zeros(L, dtype=np.float64)
        self.V = np.zeros(L, dtype=np.float64)  # link-neuron membrane (voi_lif)
        self.rr_ptr = 0
        self.fired_total = 0
        self.n_steps = 0
        self.fired_per_step = []
        self.max_staleness = 0.0

    def _proximity_vec(self, locs):
        locs = np.asarray(locs, dtype=np.float64)
        prox = np.empty(len(self.links), dtype=np.float64)
        for k, (i, j) in enumerate(self.links):
            d = float(np.linalg.norm(locs[i] - locs[j]))
            prox[k] = np.exp(-0.5 * (d / self.sigma) ** 2)
        return prox

    def _belief_error_vec(self, locs, known):
        """be[(i->j)] = || true_pos(i) - what receiver j currently thinks i's pos is ||."""
        locs = np.asarray(locs, dtype=np.float64)
        known = np.asarray(known, dtype=np.float64)  # known[j][i] = j's belief of i
        be = np.empty(len(self.links), dtype=np.float64)
        for k, (i, j) in enumerate(self.links):
            be[k] = float(np.linalg.norm(locs[i] - known[j][i]))
        return be

    def _dist_vec(self, locs):
        locs = np.asarray(locs, dtype=np.float64)
        d = np.empty(len(self.links), dtype=np.float64)
        for k, (i, j) in enumerate(self.links):
            d[k] = float(np.linalg.norm(locs[i] - locs[j]))
        return d

    def select(self, locs, known, step_idx, uncert=None):
        """Return the indices (into self.links) of the links that fire this step.
        `known[j][i]` is receiver j's currently-held position of sender i.
        `uncert[j][i]` (optional, Kalman mode) = sqrt(trace(P_pos)) of receiver j's belief of i;
        when given, an uncertainty-augmented expected belief error ehat = be + beta*uncert is used,
        so the scheduler favours links that are both wrong AND uncertain."""
        B = min(self.budget, len(self.links))
        beta = getattr(tp, "KF_UNCERT_BETA", 1.0)

        def _ehat(locs, known):
            be = self._belief_error_vec(locs, known)
            if uncert is None:
                return be
            u = np.asarray(uncert, dtype=np.float64)
            add = np.array([u[j][i] for (i, j) in self.links], dtype=np.float64)
            return be + beta * add

        if self.mode == "round_robin":
            chosen = [(self.rr_ptr + k) % len(self.links) for k in range(B)]
            self.rr_ptr = (self.rr_ptr + B) % len(self.links)
        elif self.mode == "random":
            chosen = list(self.rng.choice(len(self.links), size=B, replace=False))
        else:
            if self.mode == "stale":
                score = self.staleness.copy()
            elif self.mode == "prox":
                score = self._proximity_vec(locs)
            elif self.mode == "err":
                score = _ehat(locs, known)
            elif self.mode == "whittle":
                # AoI Whittle index for a reliable link under linear age cost:
                #   W_k = w_k * h_k * (h_k + 2),  h_k = staleness (age), w_k = proximity weight
                # (Kadota2018 reliable-channel index; Hsu2018 greedy-index near-optimality).
                # Quadratic-in-age => super-linear penalty on starved links => bounded worst-case age.
                h = self.staleness
                w = self._proximity_vec(locs)
                score = w * h * (h + 2.0)
            elif self.mode == "decision_voi":
                # decision-aware (counterfactual Howard) VoI. The per-link decision-impact score
                # cannot be computed here (no policy access); TestWorker precomputes it and stashes
                # it on self.precomputed_score just before calling select(). A tiny geometric
                # tiebreak ranks links with DVoI==0 by belief error.
                if getattr(self, "precomputed_score", None) is None:
                    raise RuntimeError("decision_voi: precomputed_score not set by TestWorker")
                score = (np.asarray(self.precomputed_score, dtype=np.float64)
                         + 1e-12 * (self._belief_error_vec(locs, known) * self._proximity_vec(locs)))
            elif self.mode == "voi_lif":
                # leaky integrate-and-fire: each link-neuron accrues a VoI current plus a
                # constant floor, so a chronically-low-VoI link still eventually fires.
                cur = _ehat(locs, known) * self._proximity_vec(locs) + self.floor
                self.V = self.leak * self.V + cur
                score = self.V
            elif self.mode == "wsum":
                # learned/optimised linear value-of-information over normalised features
                be = _ehat(locs, known); prox = self._proximity_vec(locs)
                dist = self._dist_vec(locs)
                phi = np.stack([be / 50.0, prox, (be / 50.0) * prox,
                                self.staleness / 5.0, dist / 50.0], axis=1)
                score = phi @ self.weights
            else:  # voi: how-wrong x how-much-it-matters (+ tiny staleness tiebreak)
                score = (_ehat(locs, known) * self._proximity_vec(locs)
                         + 1e-9 * self.staleness)
            chosen = list(np.argsort(-score, kind="stable")[:B])
        return [int(k) for k in chosen]

    def apply(self, chosen_idx):
        mask = np.zeros(len(self.links), dtype=bool)
        if len(chosen_idx):
            mask[np.asarray(chosen_idx, dtype=int)] = True
        self.staleness[mask] = 0.0
        self.staleness[~mask] += 1.0
        self.V[mask] = 0.0  # soft reset of fired link-neurons (voi_lif)
        self.max_staleness = max(self.max_staleness, float(self.staleness.max()))
        self.fired_total += int(mask.sum())
        self.n_steps += 1
        self.fired_per_step.append(int(mask.sum()))

    def chosen_pairs(self, chosen_idx):
        return [self.links[k] for k in chosen_idx]

    def summary(self):
        return {
            "comm_mode": f"targeted_{self.mode}",
            "link_budget": int(self.budget),
            "spike_total": int(self.fired_total),
            "links_per_step_mean": float(self.fired_total / max(self.n_steps, 1)),
            "max_link_staleness": float(self.max_staleness),
            "broadcast_steps": int(self.n_steps),
        }


def make_comm_gate(n_agents, device="cpu"):
    """Factory that builds a CommGate from test_parameter env-var configuration."""
    mode = getattr(tp, "COMM_MODE", "periodic")
    gate_net = None
    if mode == "spiking_learned":
        gate_net = GateNet()
        path = getattr(tp, "SPIKE_GATE_PATH", "")
        if path:
            import os
            if os.path.exists(path):
                gate_net.load_state_dict(torch.load(path, map_location=device))
        gate_net.to(device)
        gate_net.eval()
    return CommGate(
        n_agents,
        mode=mode,
        device=device,
        threshold=getattr(tp, "SPIKE_THRESHOLD", 1.0),
        leak=getattr(tp, "SPIKE_LEAK", 1.0),
        refractory=getattr(tp, "SPIKE_REFRACTORY", 0),
        w_drift=getattr(tp, "SPIKE_W_DRIFT", 1.0),
        w_cov=getattr(tp, "SPIKE_W_COV", 0.0),
        w_util=getattr(tp, "SPIKE_W_UTIL", 0.0),
        random_prob=getattr(tp, "RANDOM_COMM_PROB", 0.2),
        gate_net=gate_net,
        evtrig_delta=getattr(tp, "EVTRIG_DELTA", 40.0),
        interval=getattr(tp, "COMMUNICATION_INTERVAL", 1),
    )
