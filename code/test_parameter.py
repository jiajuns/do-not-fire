import os

TEST_N_AGENTS = int(os.environ.get("BLMRE_N_AGENTS", "8"))

NODE_INPUT_DIM = 5
OTHER_INFO_INPUT_DIM = 3
EMBEDDING_DIM = 64
K_SIZE = 25  # the number of neighbors

USE_GPU = True  # do you want to use GPUS?
NUM_GPU = 1
NUM_META_AGENT = int(os.environ.get("BLMRE_NUM_META_AGENT", "24"))  # the number of processes
FOLDER_NAME = 'clean'
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{FOLDER_NAME}/gifs'

NUM_TEST = int(os.environ.get("BLMRE_NUM_TEST", "100"))
NUM_RUN = 1
SAVE_GIFS = False  # do you want to save GIFs

# --- Experiment logging (Stage 3+) ---
ENABLE_EVAL_LOGGING = True
SEED = int(os.environ.get("BLMRE_SEED", "0"))
COMMUNICATION_INTERVAL = int(os.environ.get("BLMRE_COMM_INTERVAL", "1"))
OVERLAP_PENALTY_WEIGHT = float(os.environ.get("BLMRE_OVERLAP_PENALTY_WEIGHT", "0.0"))

# --- Spiking event-triggered communication (SET-Comm) ---
# COMM_MODE selects the broadcast scheduler used at test_worker.py:
#   periodic | spiking | spiking_learned | random
COMM_MODE = os.environ.get("BLMRE_COMM_MODE", "periodic")
SPIKE_THRESHOLD = float(os.environ.get("BLMRE_SPIKE_THRESHOLD", "1.0"))   # theta
SPIKE_LEAK = float(os.environ.get("BLMRE_SPIKE_LEAK", "1.0"))            # lambda in (0,1]
SPIKE_REFRACTORY = int(os.environ.get("BLMRE_SPIKE_REFRACTORY", "0"))    # min steps between fires
SPIKE_W_DRIFT = float(os.environ.get("BLMRE_SPIKE_W_DRIFT", "1.0"))      # weight on embedding drift
SPIKE_W_COV = float(os.environ.get("BLMRE_SPIKE_W_COV", "0.0"))         # weight on coverage gain
SPIKE_W_UTIL = float(os.environ.get("BLMRE_SPIKE_W_UTIL", "0.0"))       # weight on local utility mass
RANDOM_COMM_PROB = float(os.environ.get("BLMRE_RANDOM_COMM_PROB", "0.2"))  # equal-budget baseline rate
SPIKE_GATE_PATH = os.environ.get("BLMRE_SPIKE_GATE_PATH", "model/spiking_gate.pth")
# When set, schedule the robots' POSITION/state sharing too: on silent steps teammates
# see each other's last-broadcast (stale) position. This makes communication carry the
# coordination-critical info, so cutting it has a real performance cost.
GATE_POSITIONS = os.environ.get("BLMRE_GATE_POSITIONS", "0") == "1"
# classic distance-triggered baseline (COMM_MODE=evtrig): broadcast when moved >= delta meters
EVTRIG_DELTA = float(os.environ.get("BLMRE_EVTRIG_DELTA", "40.0"))
# robustness: probability a broadcast is lost (teammates keep stale info)
COMM_DROP = float(os.environ.get("BLMRE_COMM_DROP", "0.0"))
# range-limited interference channel (OW-Desync): 0 = full-mesh broadcast (default); >0 = a broadcast
# reaches only receivers within R_COMM meters (out-of-range teammates keep the stale copy)
R_COMM = float(os.environ.get("BLMRE_R_COMM", "0.0"))

# --- Targeted / pairwise communication under a hard per-step link budget ---
# When enabled, position+embedding sharing is point-to-point: at most LINK_BUDGET
# directed (sender->receiver) links transmit per step; a LinkScheduler chooses which.
# On a silent link the receiver keeps the sender's last-received (stale) position/embedding.
COMM_TARGETED = os.environ.get("BLMRE_COMM_TARGETED", "0") == "1"
LINK_BUDGET = int(os.environ.get("BLMRE_LINK_BUDGET", "4"))   # B: directed links per step
TARGET_MODE = os.environ.get("BLMRE_TARGET_MODE", "voi")      # voi|stale|prox|round_robin|random
VOI_SIGMA = float(os.environ.get("BLMRE_VOI_SIGMA", "50.0"))  # proximity kernel width (meters)
VOI_FLOOR = float(os.environ.get("BLMRE_VOI_FLOOR", "0.0"))   # LIF constant floor current (voi_lif)
# learned/optimised linear VoI weights for mode 'wsum' (comma list of 5); None -> be*prox default
_vw = os.environ.get("BLMRE_VOI_WEIGHTS", "").strip()
VOI_WEIGHTS = [float(x) for x in _vw.split(",")] if _vw else None

# --- Predictive belief maintenance (constant-velocity extrapolation on silent links) ---
# When ON, a silent receiver extrapolates each sender's pose from the last TWO delivered
# poses+timestamps (CV model) instead of holding the stale copy; VoI/firing is then scored on
# the post-prediction RESIDUAL, so budget flows to UNPREDICTABLE (turning/stopping) senders.
# Default OFF so 'hold' (current behaviour) vs 'predictive' is a clean matched-budget ablation.
PREDICTIVE_BELIEF = os.environ.get("BLMRE_PREDICTIVE_BELIEF", "0") == "1"
# Optional safety clip on the per-step estimated velocity (meters/step); large => inactive.
V_MAX_PRED = float(os.environ.get("BLMRE_V_MAX_PRED", "1e9"))

# --- Decision-aware (counterfactual / Howard) value of information (Addition 5) ---
# TARGET_MODE='decision_voi': link-VoI = change in the RECEIVER's frozen-policy waypoint decision
# if its belief of the SENDER were refreshed from the stale/predicted pose to the true pose
# (Howard 1966). Computed in TestWorker (it holds the policy); the scheduler consumes the score.
# Tractable: geometric prefilter to top-M links, then ONE counterfactual forward per candidate.
DVOI_PREFILTER = int(os.environ.get("BLMRE_DVOI_PREFILTER", "8"))   # M: links re-scored/step (0 => geometric)
DVOI_METRIC = os.environ.get("BLMRE_DVOI_METRIC", "logit")          # argmax | logit | kl
DVOI_FLOOR = float(os.environ.get("BLMRE_DVOI_FLOOR", "0.0"))       # anti-starvation constant current

# --- Kalman-filter uncertainty belief (Addition 6) ---
# When ON, each directed link (s->r) is tracked by a constant-velocity KALMAN FILTER giving a
# predicted MEAN (used exactly like the CV-point predictor) AND a covariance P. sqrt(trace(P_pos))
# augments the expected belief error in VoI, so the scheduler refreshes links that are both WRONG
# and UNCERTAIN. KF implies predictive belief.
KALMAN_BELIEF = os.environ.get("BLMRE_KALMAN_BELIEF", "0") == "1"
if KALMAN_BELIEF:
    PREDICTIVE_BELIEF = True  # the KF mean IS the prediction; compose with the predictive path
A_MAX_PRED = float(os.environ.get("BLMRE_A_MAX_PRED", "1.0"))      # a_max (Lemma 1'): accel std; q = A_MAX_PRED**2
KF_MEAS_VAR = float(os.environ.get("BLMRE_KF_MEAS_VAR", "1e-4"))   # R = KF_MEAS_VAR * I_2 (deliveries near-exact)
KF_UNCERT_BETA = float(os.environ.get("BLMRE_KF_UNCERT_BETA", "1.0"))  # beta on sqrt(trace(P_pos)) in ehat

# --- Non-learned frontier explorer: an alternative BACKBONE for the cross-backbone generality study ---
# When ON, select_next_waypoint uses a classical highest-utility-frontier rule instead of the frozen
# policy: pick the reachable neighbour with the most observable frontiers (node utility), guided toward
# distant frontiers by the guidepost signal and repelled from believed-teammate positions (occupancy
# channel). The env, node graph and communication-scheduling layer are reused unchanged, so the same
# synchronised-vs-de-synchronised comparison runs on a different planner.
FRONTIER_EXPLORER = os.environ.get("BLMRE_FRONTIER", "0") == "1"
FRONTIER_GUIDEPOST_W = float(os.environ.get("BLMRE_FRONTIER_GUIDE_W", "0.5"))     # weight on guidepost signal
FRONTIER_DECONFLICT_W = float(os.environ.get("BLMRE_FRONTIER_DECONFLICT_W", "1.0"))  # teammate repulsion
FRONTIER_STAY_PENALTY = float(os.environ.get("BLMRE_FRONTIER_STAY", "0.2"))       # discourage staying put
# Two distinct classical planners (give two non-learned backbones for the generality study):
#   'utility' = highest observable-frontier count (info-greedy);  'nearest' = Yamauchi nearest-frontier.
FRONTIER_MODE = os.environ.get("BLMRE_FRONTIER_MODE", "utility")


def _default_method_name():
    if COMM_MODE == "spiking":
        return f"SETComm_th{SPIKE_THRESHOLD:g}"
    if COMM_MODE == "spiking_learned":
        return "SETComm_L"
    if COMM_MODE == "random":
        return f"RandomComm_p{RANDOM_COMM_PROB:g}"
    # periodic/interval baseline naming (overlap-penalty path retired; weight is 0 in all runs)
    ci, ow = int(COMMUNICATION_INTERVAL), float(OVERLAP_PENALTY_WEIGHT)
    if ow > 0.0:
        return f"OverlapPenalty_{ow}" if ci == 1 else "CommPlusOverlap"
    return "Original" if ci == 1 else f"CommInterval_{ci}"


EXPERIMENT_NAME = os.environ.get("BLMRE_EXPERIMENT_NAME", "original_baseline")
METHOD_NAME = os.environ.get("BLMRE_METHOD_NAME", _default_method_name())

SAVE_TRAJECTORY = os.environ.get("BLMRE_SAVE_TRAJECTORY", "0") == "1"
_ids = os.environ.get("BLMRE_SAVE_TRAJECTORY_EPISODE_IDS", "").strip()
SAVE_TRAJECTORY_EPISODE_IDS = (
    {int(x) for x in _ids.split(",") if x.strip() != ""} if _ids else None
)

# Minimal smoke test: 1 seed, 1 episode (set False for full evaluation)
SMOKE_TEST = os.environ.get("BLMRE_SMOKE_TEST", "0") == "1"
if SMOKE_TEST:
    NUM_TEST = 1
    NUM_META_AGENT = 1
    SAVE_TRAJECTORY = os.environ.get("BLMRE_SAVE_TRAJECTORY", "0") == "1"
