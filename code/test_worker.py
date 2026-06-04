import time

import matplotlib.pyplot as plt
import torch
from env import Env
from agent import Agent
from utils import *
from ground_truth_node_manager import Ground_truth_node_manager
import matplotlib.gridspec as gridspec
from test_parameter import *
from episode_metrics import EpisodeMetricsTracker
from spiking_comm import make_comm_gate, LinkScheduler

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class _CVKalman:
    """Constant-velocity Kalman filter for one directed link (s->r).
    State x=[px,py,vx,vy]; covariance P (4x4). predict_to(t) is non-mutating (scoring/planning
    queries); update(t, z) mutates on a delivery. Generalises the two-point CV point predictor:
    mean[:2] == p_hat, plus a covariance P quantifying uncertainty."""
    def __init__(self, p0, t0, q, r0, v_max, s_v0):
        self.x = np.array([p0[0], p0[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([r0, r0, s_v0 * s_v0, s_v0 * s_v0]).astype(np.float64)
        self.t = int(t0)
        self.q = float(q); self.r0 = float(r0)
        self.v_max = float(v_max)
        self.n_upd = 1

    @staticmethod
    def _F(dt):
        F = np.eye(4); F[0, 2] = dt; F[1, 3] = dt; return F

    def _Q(self, dt):
        d = float(dt); blk = np.array([[d**4 / 4.0, d**3 / 2.0], [d**3 / 2.0, d * d]])
        Q = np.zeros((4, 4))
        Q[0, 0] = blk[0, 0]; Q[0, 2] = blk[0, 1]; Q[2, 0] = blk[1, 0]; Q[2, 2] = blk[1, 1]
        Q[1, 1] = blk[0, 0]; Q[1, 3] = blk[0, 1]; Q[3, 1] = blk[1, 0]; Q[3, 3] = blk[1, 1]
        return self.q * Q

    def predict_to(self, step_idx):
        """Non-mutating: return (mean_pos(2,), sqrt(trace(P_pos))) extrapolated to step_idx."""
        dt = int(step_idx) - self.t
        if dt <= 0:
            return self.x[:2].copy(), float(np.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0)))
        F = self._F(dt); xp = F @ self.x; Pp = F @ self.P @ F.T + self._Q(dt)
        return xp[:2].copy(), float(np.sqrt(max(Pp[0, 0] + Pp[1, 1], 0.0)))

    def update(self, step_idx, z):
        dt = int(step_idx) - self.t
        if dt > 0:
            F = self._F(dt); self.x = F @ self.x; self.P = F @ self.P @ F.T + self._Q(dt)
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        R = self.r0 * np.eye(2)
        y = np.asarray(z, np.float64) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T  # Joseph form
        sp = float(np.linalg.norm(self.x[2:]))
        if sp > self.v_max and sp > 0.0:
            self.x[2:] *= (self.v_max / sp)
        self.t = int(step_idx); self.n_upd += 1


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False, greedy=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.greedy = greedy

        self.env = Env(global_step, plot=self.save_image, test=True)
        self.n_agent = TEST_N_AGENTS
        self.comm_gate = make_comm_gate(self.n_agent, self.device)
        self.ground_truth_node_manager = None
        self.robot_list = [Agent(i, policy_net, deepcopy(self.env), deepcopy(self.ground_truth_node_manager), self.device, self.save_image) for i in
                           range(TEST_N_AGENTS)]

        self.perf_metrics = dict()
        self._metrics_tracker = None

    def run_episode(self):
        t_start = time.perf_counter()
        done = False
        steps_taken = 0
        self.comm_gate.reset_episode()
        # last-broadcast positions of each robot (for GATE_POSITIONS scheduling)
        self.comm_loc = self.env.robot_locations.copy()
        self._cur_step = 0  # current step index (for predictive-belief extrapolation in _planning_locs)
        # targeted/pairwise scheduling under a hard per-step link budget (optional)
        self.targeted = COMM_TARGETED
        if self.targeted:
            n = self.n_agent
            # per-receiver belief of every robot's position; init: everyone knows everyone
            self.known_loc = np.stack([self.env.robot_locations.copy() for _ in range(n)])
            # predictive belief: per-(receiver r, sender s) last-two DELIVERED poses + delivery
            # step indices, for constant-velocity extrapolation on silent links. p_prev=None
            # until a link has delivered >=2 times -> fall back to hold. known_loc stays = last pose.
            self.predictive = PREDICTIVE_BELIEF
            self.kalman = KALMAN_BELIEF  # KF mode (implies predictive)
            self.p_last = np.stack([self.env.robot_locations.copy() for _ in range(n)]).astype(np.float64)
            self.t_last = np.zeros((n, n), dtype=np.int64)
            self.p_prev = [[None] * n for _ in range(n)]
            self.t_prev = [[None] * n for _ in range(n)]
            if self.kalman:
                # per-(receiver r, sender s) constant-velocity Kalman filter; seeded at step 0.
                q = A_MAX_PRED ** 2
                self._kf = [[None] * n for _ in range(n)]
                for r in range(n):
                    for s in range(n):
                        if r == s:
                            continue
                        self._kf[r][s] = _CVKalman(
                            self.env.robot_locations[s].astype(np.float64), 0,
                            q=q, r0=KF_MEAS_VAR, v_max=V_MAX_PRED, s_v0=V_MAX_PRED)
            self.link_sched = LinkScheduler(
                n, LINK_BUDGET, mode=TARGET_MODE, sigma=VOI_SIGMA,
                leak=SPIKE_LEAK, floor=VOI_FLOOR, weights=VOI_WEIGHTS, rng=self.comm_gate.rng,
            )
            self.link_sched.reset()
        track_metrics = ENABLE_EVAL_LOGGING
        if track_metrics:
            self._metrics_tracker = EpisodeMetricsTracker(
                self.n_agent,
                self.env.ground_truth,
                save_trajectory=SAVE_TRAJECTORY,
            )
            self._metrics_tracker.record_initial_positions(self.env.robot_locations)
            self._metrics_tracker.record_coverage(0, float(self.env.explored_rate))

        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_planning_state(self._planning_locs(robot.id))

        for i in range(MAX_EPISODE_STEP):
            steps_taken = i + 1
            self._cur_step = i
            selected_locations = []
            dist_list = []
            local_observations = []
            self._embeddings = [None] * self.n_agent

            for robot in self.robot_list:
                local_observations.append(robot.get_no_padding_observation())
                local_observation = local_observations[robot.id]

                local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
                current_coord = torch.tensor(robot.location, dtype=torch.float32, device=self.device).reshape(1, 1, 2)
                enhanced_node_feature, current_state_feature = robot.policy_net.get_current_state_feature(local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index,current_coord)
                self._embeddings[robot.id] = current_state_feature.detach()
                # broadcast model: the spiking gate decides WHEN to send to everyone
                # (periodic mode reproduces the fixed-interval baseline). Skipped in the
                # targeted model, where a budgeted link scheduler runs once per step below.
                if not self.targeted and self.comm_gate.should_broadcast(
                    robot.id, i, current_state_feature.detach(),
                    coverage=float(self.env.explored_rate), utility=robot.utility,
                    position=self.env.robot_locations[robot.id],
                ):
                    # robustness: a fired broadcast may be lost on an unreliable link
                    # (never drop the step-0 broadcast: it seeds teammates' mailboxes)
                    lost = (i > 0 and COMM_DROP > 0.0 and self.comm_gate.rng.random() < COMM_DROP)
                    if not lost:
                        self.send_msg(current_state_feature.detach(), robot.id)
                        if GATE_POSITIONS:
                            # broadcasting also announces this robot's current position
                            self.comm_loc[robot.id] = self.env.robot_locations[robot.id].copy()
                    # delivery feedback (ACK) so a loss-aware scheduler (vlw) can retransmit drops
                    self.comm_gate.notify_outcome(robot.id, not lost, i)

            # targeted/pairwise model: a budgeted scheduler picks which directed links fire
            if self.targeted:
                self._run_targeted_step(i)

            for robot in self.robot_list:
                local_observation = local_observations[robot.id]
                local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation

                stacked_msg = robot.get_stacked_msg()

                visit_map = None
                overlap_diag = None
                if track_metrics and OVERLAP_PENALTY_WEIGHT > 0:
                    visit_map = self._metrics_tracker.visit_count_map
                    overlap_diag = self._metrics_tracker.overlap_diag
                next_location, _, _ = robot.select_next_waypoint(
                    local_observation,
                    stacked_msg,
                    visit_count_map=visit_map,
                    overlap_diag=overlap_diag,
                )
                
                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:,
                                                                          0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].local_node_manager.local_nodes_dict.nearest_neighbors(
                        selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location

            for robot, next_location in zip(self.robot_list, selected_locations):
                self.env.step(next_location, robot.id)
                robot.env.step(next_location, robot.id)
                robot.update_graph(robot.env.belief_info, deepcopy(robot.env.robot_locations[robot.id]))

            if track_metrics:
                self._metrics_tracker.record_positions_after_step(self.env.robot_locations)
                self._metrics_tracker.record_robot_visits(
                    self.env.belief_info, self.env.robot_locations
                )
                self._metrics_tracker.record_coverage(steps_taken, float(self.env.explored_rate))

            for robot in self.robot_list:
                robot.update_planning_state(self._planning_locs(robot.id))

            # update shared coverage baseline used by the coverage-gain trigger signal
            self.comm_gate.end_step(float(self.env.explored_rate))

            if self.env.explored_rate >= 0.95:
                done = True

            if self.save_image:
                self.plot_local_env(i)

            if done:
                break

        # save metrics (episode-level logging; does not affect control logic)
        map_name = os.path.basename(self.env.map_path) if self.env.map_path else ""
        map_dir = "maps_test" if self.env.test else "maps_medium"
        map_list = sorted(os.listdir(map_dir)) if os.path.isdir(map_dir) else []
        map_id = map_list.index(map_name) if map_name in map_list else self.global_step % max(len(map_list), 1)

        self.perf_metrics['success_rate'] = done
        self.perf_metrics['final_coverage_ratio'] = float(self.env.explored_rate)
        self.perf_metrics['episode_steps'] = steps_taken
        self.perf_metrics['map_name'] = map_name
        self.perf_metrics['map_id'] = map_id
        self.perf_metrics['computation_time'] = time.perf_counter() - t_start
        if getattr(self, "targeted", False):
            self.perf_metrics.update(self.link_sched.summary())
        else:
            self.perf_metrics.update(self.comm_gate.summary())

        if track_metrics:
            extra = self._metrics_tracker.finalize(self.env)
            self.perf_metrics.update(extra)
            if SAVE_TRAJECTORY and (
                SAVE_TRAJECTORY_EPISODE_IDS is None
                or self.global_step in SAVE_TRAJECTORY_EPISODE_IDS
            ):
                from episode_metrics import save_trajectory_bundle

                traj_path = save_trajectory_bundle(
                    EXPERIMENT_NAME,
                    SEED,
                    self.global_step,
                    self._metrics_tracker,
                    map_name,
                    map_id,
                    self.perf_metrics,
                )
                self.perf_metrics["trajectory_path"] = traj_path
        else:
            travel_dists = [robot.travel_dist for robot in self.robot_list]
            self.perf_metrics['travel_dist'] = max(travel_dists)
            self.perf_metrics['max_single_robot_distance'] = float(max(travel_dists))
            self.perf_metrics['average_single_robot_distance'] = float(np.mean(travel_dists))
            self.perf_metrics['total_travel_distance'] = float(max(travel_dists))

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def _planning_locs(self, robot_id):
        # targeted/pairwise: this receiver sees each teammate at the position it last
        # received over an active link (per-pair staleness); its own position is live.
        if getattr(self, "targeted", False):
            if getattr(self, "predictive", False):
                loc = self._predicted_known(self._cur_step)[robot_id].copy()
            else:
                loc = self.known_loc[robot_id].copy()
            loc[robot_id] = self.env.robot_locations[robot_id].copy()
            return loc
        # default: all robots' live positions (original behaviour).
        # GATE_POSITIONS: teammates appear at their last-broadcast (stale) position;
        # this robot still knows its own live position.
        if not GATE_POSITIONS:
            return self.env.robot_locations
        loc = self.comm_loc.copy()
        loc[robot_id] = self.env.robot_locations[robot_id].copy()
        return loc

    def _predicted_known(self, step_idx):
        """Per-receiver beliefs of every sender's position at `step_idx`.
        With predictive belief ON, silent links are constant-velocity extrapolated from the
        last two DELIVERED poses (p_hat = p_last + v_est*(t - t_last)); otherwise the held
        (stale) copy. With Kalman belief ON, the KF posterior mean is used instead.
        Shape (n, n, 2): out[r][s] = receiver r's belief of sender s."""
        if getattr(self, "kalman", False):
            n = self.n_agent
            out = self.known_loc.astype(np.float64).copy()
            for r in range(n):
                for s in range(n):
                    if r == s or self._kf[r][s] is None:
                        continue
                    out[r][s], _ = self._kf[r][s].predict_to(step_idx)
            return out
        if not getattr(self, "predictive", False):
            return self.known_loc
        n = self.n_agent
        out = self.known_loc.astype(np.float64).copy()
        for r in range(n):
            for s in range(n):
                if r == s or self.p_prev[r][s] is None:
                    continue  # <2 deliveries -> hold (out already = last delivered pose)
                dt = self.t_last[r][s] - self.t_prev[r][s]
                if dt <= 0:
                    continue
                v_est = (self.p_last[r][s] - self.p_prev[r][s]) / float(dt)
                speed = float(np.linalg.norm(v_est))
                if speed > V_MAX_PRED and speed > 0.0:
                    v_est = v_est * (V_MAX_PRED / speed)  # safety clip on the two-point fit
                out[r][s] = self.p_last[r][s] + v_est * float(step_idx - self.t_last[r][s])
        return out

    def _predicted_uncertainty(self, step_idx):
        """u[r][s] = sqrt(trace(P_pos)) of receiver r's KF belief of sender s at step_idx.
        Returns None unless Kalman mode is on (so non-KF scoring is byte-identical to before)."""
        if not getattr(self, "kalman", False):
            return None
        n = self.n_agent
        u = np.zeros((n, n), dtype=np.float64)
        for r in range(n):
            for s in range(n):
                if r == s or self._kf[r][s] is None:
                    continue
                _, u[r][s] = self._kf[r][s].predict_to(step_idx)
        return u

    def _decision_voi_scores(self, known_for_score, step_idx):
        """Decision-aware (counterfactual Howard, howard1966infovalue) value of information per
        directed link s->r: how much receiver r's FROZEN-policy waypoint decision changes if its
        belief of sender s is refreshed from the stale/predicted pose to the true pose. Tractable:
        geometric prefilter -> top-M links -> ONE counterfactual policy forward per candidate; the
        per-receiver baseline (its step-i belief, all teammates predicted) is computed once and
        reused, so the score isolates the impact of refreshing s. Non-prefiltered links get 0."""
        links = self.link_sched.links
        L = len(links)
        scores = np.zeros(L, dtype=np.float64)
        M = min(int(DVOI_PREFILTER), L)
        if M <= 0:
            return scores
        true_locs = self.env.robot_locations
        geo = (self.link_sched._belief_error_vec(true_locs, known_for_score)
               * self.link_sched._proximity_vec(true_locs))
        cand = list(np.argsort(-geo, kind="stable")[:M])
        attrs = ("local_node_coords", "utility", "guidepost", "occupancy",
                 "local_adjacent_matrix", "current_local_index", "local_neighbor_indices")
        base = {}  # r -> (base_lp, base_locs, saved_state)
        for k in cand:
            s, r = links[k]
            if float(np.linalg.norm(true_locs[s] - known_for_score[r][s])) < 1e-6:
                continue  # belief already correct -> no decision impact
            robot = self.robot_list[r]
            if r not in base:
                saved = tuple(getattr(robot, a) for a in attrs)
                base_locs = known_for_score[r].astype(np.float64).copy()
                base_locs[r] = true_locs[r].astype(np.float64)
                robot.update_planning_state(base_locs)
                with torch.no_grad():
                    base_lp = self._policy_logp(robot, robot.get_no_padding_observation(),
                                                robot.get_stacked_msg())
                base[r] = (base_lp, base_locs, saved)
            base_lp, base_locs, _saved = base[r]
            cf_locs = base_locs.copy()
            cf_locs[s] = true_locs[s].astype(np.float64)
            robot.update_planning_state(cf_locs)
            with torch.no_grad():
                cf_lp = self._policy_logp(robot, robot.get_no_padding_observation(),
                                          robot.get_stacked_msg())
            scores[k] = self._dvoi_metric(base_lp, cf_lp) + float(DVOI_FLOOR)
        # restore every touched receiver's planning state (DVoI must not perturb action selection)
        for r, (_lp, _bl, saved) in base.items():
            for a, v in zip(attrs, saved):
                setattr(self.robot_list[r], a, v)
        return scores

    def _policy_logp(self, robot, obs, msg):
        lni, lnpm, lem, cli, cle, lepm = obs
        cc = torch.tensor(robot.location, dtype=torch.float32, device=self.device).reshape(1, 1, 2)
        return robot.policy_net(lni, lnpm, lem, cli, cle, lepm, cc, msg)

    def _dvoi_metric(self, base_lp, cf_lp):
        b = base_lp.reshape(-1).float(); c = cf_lp.reshape(-1).float()
        m = DVOI_METRIC
        if m == "argmax":
            return float(int(torch.argmax(b).item() != torch.argmax(c).item()))
        if m == "kl":
            pc = torch.softmax(c, -1)
            return float((pc * (torch.log(pc + 1e-9) - torch.log_softmax(b, -1))).sum().item())
        a = int(torch.argmax(b).item())  # default 'logit': |shift of stale-optimal action log-prob|
        return float(abs(c[a].item() - b[a].item()))

    def _run_targeted_step(self, step_idx):
        """Targeted/pairwise communication: under a hard per-step link budget, a
        LinkScheduler picks which directed (sender->receiver) links transmit. An active
        link refreshes the receiver's stored position AND embedding of the sender; silent
        links keep the stale copy. Step 0 seeds every mailbox once (full sync)."""
        n = self.n_agent
        if step_idx == 0:
            for s in range(n):
                emb = self._embeddings[s]
                for r in range(n):
                    if r == s:
                        continue
                    self.robot_list[r].msgs[s].append(emb.clone())
                    if self._metrics_tracker is not None:
                        self._metrics_tracker.record_targeted_msg(emb)
            return
        # score on the PREDICTED belief (post-prediction residual VoI); every select() branch
        # reads VoI from `known`, so feeding predicted beliefs needs no scoring-code change.
        known_for_score = self._predicted_known(step_idx)
        uncert_for_score = self._predicted_uncertainty(step_idx)
        if getattr(self.link_sched, "mode", "") == "decision_voi":
            self.link_sched.precomputed_score = self._decision_voi_scores(known_for_score, step_idx)
        chosen = self.link_sched.select(self.env.robot_locations, known_for_score, step_idx,
                                        uncert=uncert_for_score)
        delivered = []
        for k in chosen:
            s, r = self.link_sched.links[k]
            # lossy channel: a scheduled link may be dropped in transit; the receiver keeps
            # its stale copy and (since staleness is not reset) the link is naturally retried.
            if COMM_DROP > 0.0 and self.comm_gate.rng.random() < COMM_DROP:
                continue
            if getattr(self, "kalman", False):
                self._kf[r][s].update(step_idx, self.env.robot_locations[s].astype(np.float64))
            elif getattr(self, "predictive", False):
                # roll the two-point CV tracker on this delivery (silent links keep prior anchor)
                self.p_prev[r][s] = self.p_last[r][s].copy()
                self.t_prev[r][s] = int(self.t_last[r][s])
                self.p_last[r][s] = self.env.robot_locations[s].astype(np.float64).copy()
                self.t_last[r][s] = int(step_idx)
            self.known_loc[r][s] = self.env.robot_locations[s].copy()
            emb = self._embeddings[s]
            mb = self.robot_list[r].msgs[s]
            if len(mb) > 5:
                mb.pop(0)
            mb.append(emb.clone())
            if self._metrics_tracker is not None:
                self._metrics_tracker.record_targeted_msg(emb)
            delivered.append(k)
        self.link_sched.apply(delivered)

    def send_msg(self, msg, robot_id):
        if self._metrics_tracker is not None:
            self._metrics_tracker.record_send_msg(msg, len(self.robot_list))
        # range-limited interference channel (OW-Desync): a broadcast from robot_id reaches only
        # receivers within R_COMM; out-of-range teammates keep their stale copy. R_COMM<=0 => full mesh.
        sloc = self.env.robot_locations[robot_id] if R_COMM > 0 else None
        for robot in self.robot_list:
            if R_COMM > 0 and robot.id != robot_id:
                if float(np.linalg.norm(self.env.robot_locations[robot.id] - sloc)) > R_COMM:
                    continue  # out of range: receiver keeps the stale copy
            if len(robot.msgs[robot_id]) > 5:
                # delete the oldest msg
                robot.msgs[robot_id].pop(0)
            robot.msgs[robot_id].append(msg.clone())

    def stack_msgs(self, msgs, robot_id):
        stacked_msg = []
        for i in range(self.n_agent):
            if i == robot_id:
                continue
            stacked_msg.append(msgs[i])
        stacked_msg = torch.stack(stacked_msg, dim=1).squeeze(2)
        return stacked_msg.clone()

    def plot_local_env(self, step):
        plt.switch_backend('agg')
        plt.style.use('fast')
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.imshow(self.env.robot_belief, cmap='gray')
        ax1.axis('off')
        color_list = ['r', 'b', 'g', 'y']
        frontiers = get_frontier_in_map(self.env.belief_info)
        frontiers = get_cell_position_from_coords(frontiers, self.env.belief_info).reshape(-1, 2)
        ax1.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=1)
        for robot in self.robot_list:
            c = color_list[robot.id]
            robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
            ax1.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)
            ax1.plot((np.array(robot.trajectory_x) - robot.global_map_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - robot.global_map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)
        
        ax1.set_title('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.env.explored_rate,
                                                                              max([robot.travel_dist for robot in
                                                                                   self.robot_list])))
            
        for i in range(self.n_agent):
            ax = fig.add_subplot(gs[i // 2, 2 + i % 2])
            ax.imshow(self.robot_list[i].local_map_info.map, cmap='gray')
            frontiers = get_frontier_in_map(self.robot_list[i].env.belief_info)
            frontiers = get_cell_position_from_coords(frontiers, self.robot_list[i].env.belief_info).reshape(-1, 2)
            ax.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=1)
            for robot in self.robot_list:
                c = color_list[robot.id]
                if robot.id == i:
                    nodes = get_cell_position_from_coords(robot.local_node_coords, robot.global_map_info)
                    ax.imshow(robot.global_map_info.map, cmap='gray')
                    ax.axis('off')
                    untility = robot.utility
                    # for plotting, if any element in utility is -1, set it to 0
                    untility[untility == -1] = 0
                    ax.scatter(nodes[:, 0], nodes[:, 1], c=untility, s=5, zorder=2)
                robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
                ax.plot(robot_cell[0], robot_cell[1], c+'o', markersize=4, zorder=5)
            
            ax.axis('off')
            ax.set_title('Robot {}'.format(i))
            
            
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)
        plt.close()