import os
import matplotlib.pyplot as plt
from skimage import io, color, util
from skimage.measure import block_reduce
from copy import deepcopy
import numpy as np

from sensor import sensor_work
from parameter import *
from test_parameter import *
from utils import *


class Env:
    def __init__(self, episode_index, plot=False, test=False):
        if test:
            N_AGENTS = TEST_N_AGENTS
        self.test = test
        self.episode_index = episode_index
        self.plot = plot
        self.map_path = None
        self.ground_truth, initial_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter

        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)  # meter

        self.sensor_range = SENSOR_RANGE  # meter
        _het = os.environ.get("BLMRE_HETERO_SRANGE", "")  # e.g. "10,30": per-robot heterogeneous sensing
        if _het:
            _r = [float(x) for x in _het.split(",")]
            self.sensor_ranges = [_r[i % len(_r)] for i in range(N_AGENTS)]
        else:
            self.sensor_ranges = [self.sensor_range] * N_AGENTS
        self.explored_rate = 0

        self.done = False

        self.robot_belief = sensor_work(initial_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        self.belief_info = Map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        
        self.ground_truth_info = Map_info(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        free, _ = get_local_node_coords(np.array([0.0, 0.0]), self.belief_info)
        _spawn = os.environ.get("BLMRE_SPAWN_CLUSTER", "")
        if _spawn == "clustered":  # seed cell + its N-1 nearest free cells: highly correlated initial views
            _seed = int(np.random.choice(free.shape[0]))
            _d = np.linalg.norm(free - free[_seed], axis=1)
            choice = np.argsort(_d)[:N_AGENTS]
        elif _spawn == "dispersed":  # farthest-point sampling: low initial correlation
            choice = [int(np.random.choice(free.shape[0]))]
            for _ in range(N_AGENTS - 1):
                _dmin = np.min(np.linalg.norm(free[:, None, :] - free[choice][None, :, :], axis=2), axis=1)
                choice.append(int(np.argmax(_dmin)))
            choice = np.array(choice)
        else:
            choice = np.random.choice(free.shape[0], N_AGENTS, replace=False)
        starts = free[choice]
        self.robot_locations = np.array(starts)

        robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        for _i, robot_cell in enumerate(robot_cells):
            self.robot_belief = sensor_work(robot_cell, self.sensor_ranges[_i] / self.cell_size, self.robot_belief,
                                            self.ground_truth)
        self.old_belief = deepcopy(self.robot_belief)
        self.global_frontiers = get_frontier_in_map(self.belief_info)

        if self.plot:
            self.frame_files = []


    def import_ground_truth(self, episode_index):
        # BLMRE_MAP_DIR lets training/eval point at disjoint map folders
        # (used for a leakage-free gate train/test split); default unchanged.
        map_dir = os.environ.get("BLMRE_MAP_DIR", "").strip()
        if not map_dir:
            map_dir = f'maps_test' if self.test else f'maps_medium'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)
        self.map_path = map_dir + '/' + map_list[map_index]
        # Robust read to uint8 grayscale in [0,255]. Modern skimage's as_gray
        # returns float[0,1] (astype(int) would zero it); convert explicitly.
        raw = io.imread(self.map_path)
        if raw.ndim == 3:
            if raw.shape[2] == 4:
                raw = color.rgba2rgb(raw)
            ground_truth = util.img_as_ubyte(color.rgb2gray(raw)).astype(int)
        else:
            ground_truth = raw.astype(int)

        ground_truth = block_reduce(ground_truth, 2, np.min)
        # Robot start marker is ~208 (grayscale rounding may give 207-209).
        rr, cc = np.nonzero((ground_truth >= 206) & (ground_truth <= 210))
        if rr.size > 10:
            robot_cell = np.array([cc[10], rr[10]])
        else:
            free = np.nonzero(ground_truth > 150)
            k = free[0].size // 2
            robot_cell = np.array([free[1][k], free[0][k]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])

    def update_robot_belief(self, robot_cell, agent_id=None):
        sr = self.sensor_ranges[agent_id] if (agent_id is not None and agent_id < len(self.sensor_ranges)) else self.sensor_range
        self.robot_belief = sensor_work(robot_cell, round(sr / self.cell_size), self.robot_belief,
                                        self.ground_truth)

    def calculate_reward(self):
        reward = 0

        global_frontiers = get_frontier_in_map(self.belief_info)
        if global_frontiers.shape[0] == 0:
            delta_num = self.global_frontiers.shape[0]
        else:
            global_frontiers = global_frontiers.reshape(-1, 2)
            frontiers_to_check = global_frontiers[:, 0] + global_frontiers[:, 1] * 1j
            pre_frontiers_to_check = self.global_frontiers[:, 0] + self.global_frontiers[:, 1] * 1j
            frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
            pre_frontiers_num = pre_frontiers_to_check.shape[0]
            delta_num = pre_frontiers_num - frontiers_num

        reward += delta_num / 40

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def check_done(self):
        if np.sum(self.ground_truth == 255) - np.sum(self.robot_belief == 255) <= 250:
            self.done = True

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoint, agent_id):
        self.evaluate_exploration_rate()
        self.robot_locations[agent_id] = next_waypoint
        reward = 0
        cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
        self.update_robot_belief(cell, agent_id)
        return reward
    
    def update_robot_location(self, next_waypoint, agent_id):
        self.robot_locations[agent_id] = next_waypoint


