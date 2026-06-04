import os

import ray
import numpy as np
import torch
from test_worker import TestWorker
import test_parameter as tp
from test_parameter import *
from parameter import MAX_EPISODE_STEP, SENSOR_RANGE
from model import PolicyNet
from experiment_logger import (
    EvaluationLogger,
    build_episode_row,
    record_checkpoint_used,
    save_run_config,
)
from episode_metrics import save_coverage_curve

def _collect_run_config():
    import parameter as param
    import test_parameter as tp

    cfg = {}
    for mod in (tp, param):
        prefix = mod.__name__ + "."
        for name in dir(mod):
            if not name.isupper():
                continue
            val = getattr(mod, name, None)
            if isinstance(val, (bool, int, float, str)) or val is None:
                cfg[prefix + name] = val
    return cfg


def run_test():
    if ENABLE_EVAL_LOGGING:
        np.random.seed(tp.SEED)
        torch.manual_seed(tp.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tp.SEED)

    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    if device == 'cuda':
        checkpoint = torch.load(f'{model_path}/checkpoint.pth')
    else:
        checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location = torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])

    checkpoint_file = os.path.abspath(os.path.join(model_path, "checkpoint.pth"))
    logger = None
    row_defaults = None
    if ENABLE_EVAL_LOGGING:
        from experiment_logger import get_experiment_group

        group = get_experiment_group()
        logger = EvaluationLogger(EXPERIMENT_NAME, group=group)
        record_checkpoint_used(checkpoint_file, EXPERIMENT_NAME)
        save_run_config(
            EXPERIMENT_NAME,
            _collect_run_config(),
            group=group,
            extra={
                "method_name": METHOD_NAME,
                "seed": tp.SEED,
                "communication_interval": tp.COMMUNICATION_INTERVAL,
                "overlap_penalty_weight": tp.OVERLAP_PENALTY_WEIGHT,
                "checkpoint_path": checkpoint_file,
                "max_episode_steps": MAX_EPISODE_STEP,
                "save_trajectory": SAVE_TRAJECTORY,
                "num_test": NUM_TEST,
            },
        )
        row_defaults = {
            "method_name": METHOD_NAME,
            "seed": tp.SEED,
            "num_agents": TEST_N_AGENTS,
            "sensor_range": SENSOR_RANGE,
            "communication_interval": tp.COMMUNICATION_INTERVAL,
            "overlap_penalty_weight": tp.OVERLAP_PENALTY_WEIGHT,
            "max_episode_steps": MAX_EPISODE_STEP,
            "total_planning_time": "",
            "average_planning_time_per_step": "",
            "checkpoint_path": checkpoint_file,
        }
        print(f"|#Eval logging enabled -> {logger.path}")

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    dist_history = []
    success = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(dist_history) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                dist_history.append(metrics['travel_dist'])
                success.append(metrics['success_rate'])
                if logger is not None:
                    episode_id = info.get("episode_number", len(dist_history) - 1)
                    row = build_episode_row(metrics, episode_id, row_defaults)
                    logger.log_episode(row)
                    cov_steps = metrics.get("coverage_steps")
                    if cov_steps is not None:
                        curve_path = save_coverage_curve(
                            EXPERIMENT_NAME,
                            tp.SEED,
                            episode_id,
                            cov_steps,
                            metrics.get("coverage_values", []),
                        )
                        print(f"|#Coverage curve -> {curve_path}")
                    if metrics.get("trajectory_path"):
                        print(f"|#Trajectory -> {metrics['trajectory_path']}")
            if curr_test < NUM_TEST:
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                curr_test += 1

        print('|#Total test:', NUM_TEST)
        print('|#Average length:', np.array(dist_history).mean())
        print('|#Length std:', np.array(dist_history).std())
        print('|#Success rate', float(np.array(success).sum()) / NUM_TEST)

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        worker = TestWorker(self.meta_agent_id, self.local_network, episode_number, device=self.device, save_image=SAVE_GIFS, greedy=True)
        worker.run_episode()

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description="Run evaluation with optional comm-interval override.")
    parser.add_argument(
        "--communication-interval",
        type=int,
        default=None,
        help="Override COMMUNICATION_INTERVAL (steps between message broadcasts).",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument(
        "--overlap-penalty-weight",
        type=float,
        default=None,
        help="Override OVERLAP_PENALTY_WEIGHT for test-time candidate re-ranking.",
    )
    parser.add_argument(
        "--comm-mode",
        type=str,
        default=None,
        choices=["periodic", "spiking", "spiking_learned", "random"],
        help="Broadcast scheduler: periodic | spiking | spiking_learned | random.",
    )
    parser.add_argument("--spike-threshold", type=float, default=None)
    parser.add_argument("--spike-leak", type=float, default=None)
    parser.add_argument("--spike-refractory", type=int, default=None)
    parser.add_argument("--spike-w-drift", type=float, default=None)
    parser.add_argument("--spike-w-cov", type=float, default=None)
    parser.add_argument("--random-comm-prob", type=float, default=None)
    parser.add_argument("--spike-gate-path", type=str, default=None)
    args = parser.parse_args()

    if args.communication_interval is not None:
        os.environ["BLMRE_COMM_INTERVAL"] = str(args.communication_interval)
    if args.overlap_penalty_weight is not None:
        os.environ["BLMRE_OVERLAP_PENALTY_WEIGHT"] = str(args.overlap_penalty_weight)
    _spike_env = {
        "BLMRE_COMM_MODE": args.comm_mode,
        "BLMRE_SPIKE_THRESHOLD": args.spike_threshold,
        "BLMRE_SPIKE_LEAK": args.spike_leak,
        "BLMRE_SPIKE_REFRACTORY": args.spike_refractory,
        "BLMRE_SPIKE_W_DRIFT": args.spike_w_drift,
        "BLMRE_SPIKE_W_COV": args.spike_w_cov,
        "BLMRE_RANDOM_COMM_PROB": args.random_comm_prob,
        "BLMRE_SPIKE_GATE_PATH": args.spike_gate_path,
    }
    for _k, _v in _spike_env.items():
        if _v is not None:
            os.environ[_k] = str(_v)
    if args.method_name:
        os.environ["BLMRE_METHOD_NAME"] = args.method_name
    elif args.communication_interval is not None or args.overlap_penalty_weight is not None:
        ci = int(args.communication_interval or os.environ.get("BLMRE_COMM_INTERVAL", "1"))
        ow = float(args.overlap_penalty_weight or os.environ.get("BLMRE_OVERLAP_PENALTY_WEIGHT", "0"))
        if ow > 0.0:
            _mn = f"OverlapPenalty_{ow}" if ci == 1 else "CommPlusOverlap"
        else:
            _mn = "Original" if ci == 1 else f"CommInterval_{ci}"
        os.environ["BLMRE_METHOD_NAME"] = _mn
    if args.experiment_name:
        os.environ["BLMRE_EXPERIMENT_NAME"] = args.experiment_name

    if (
        args.communication_interval is not None
        or args.overlap_penalty_weight is not None
        or args.experiment_name
        or args.method_name
        or any(v is not None for v in _spike_env.values())
    ):
        importlib.reload(tp)

    print(f"|#COMMUNICATION_INTERVAL: {tp.COMMUNICATION_INTERVAL}")
    print(f"|#OVERLAP_PENALTY_WEIGHT: {tp.OVERLAP_PENALTY_WEIGHT}")
    print(f"|#COMM_MODE: {tp.COMM_MODE}")
    print(f"|#METHOD_NAME: {tp.METHOD_NAME}")
    print(f"|#EXPERIMENT_NAME: {tp.EXPERIMENT_NAME}")

    ray.init()
    for i in range(NUM_RUN):
        run_test()