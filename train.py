import argparse
import csv
import os
import random
import sys
from collections import deque, namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# SUMO setup
# ============================================================

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = (
        r"C:\Program Files (x86)\SUMO"
    )

SUMO_TOOLS = Path(
    os.environ["SUMO_HOME"]
) / "tools"

if str(SUMO_TOOLS) not in sys.path:
    sys.path.append(str(SUMO_TOOLS))

import traci
import sumolib


CONFIG_FILE = Path(
    r"maps\configuration.sumocfg"
)

TRAFFIC_LIGHT_ID = "J"


# ============================================================
# Network-specific lane groups
# ============================================================

HIGHWAY_INCOMING_LANES = [
    "W_J_0",
    "W_J_1",
    "W_J_2",
    "W_J_3",
    "E_J_0",
    "E_J_1",
    "E_J_2",
    "E_J_3",
]

CITY_INCOMING_LANES = [
    "N_J_0",
    "S_J_0",
]


# ============================================================
# Signal states extracted from your generated network
# ============================================================

HIGHWAY_GREEN_STATE = (
    "rrrGGGGGgrrrGGGGGg"
)

HIGHWAY_YELLOW_STATE = (
    "rrryyyyyyrrryyyyyy"
)

CITY_GREEN_STATE = (
    "GGgrrrrrrGGgrrrrrr"
)

CITY_YELLOW_STATE = (
    "yyyrrrrrryyyrrrrrr"
)


# ============================================================
# Environment settings
# ============================================================

STATE_SIZE = 8
ACTION_SIZE = 2

DECISION_INTERVAL = 5
YELLOW_DURATION = 3

MIN_GREEN_DURATION = 10
MAX_GREEN_DURATION = 60

QUEUE_REWARD_WEIGHT = 2.0
WAITING_REWARD_WEIGHT = 1.0

RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Replay memory
# ============================================================

Transition = namedtuple(
    "Transition",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "done",
    ],
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.memory.append(
            Transition(
                np.asarray(
                    state,
                    dtype=np.float32,
                ),
                int(action),
                float(reward),
                np.asarray(
                    next_state,
                    dtype=np.float32,
                ),
                bool(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(
            self.memory,
            batch_size,
        )

        states = np.stack(
            [item.state for item in batch]
        )

        actions = np.asarray(
            [item.action for item in batch],
            dtype=np.int64,
        )

        rewards = np.asarray(
            [item.reward for item in batch],
            dtype=np.float32,
        )

        next_states = np.stack(
            [item.next_state for item in batch]
        )

        dones = np.asarray(
            [item.done for item in batch],
            dtype=np.float32,
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )


# ============================================================
# DQN
# ============================================================

class DQN(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_size),
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.0005,
        gamma=0.99,
        batch_size=64,
        replay_capacity=100_000,
        target_update_interval=500,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.batch_size = batch_size

        self.target_update_interval = (
            target_update_interval
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.online_network = DQN(
            state_size,
            action_size,
        ).to(self.device)

        self.target_network = DQN(
            state_size,
            action_size,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(
            replay_capacity
        )

        self.learning_steps = 0

    def choose_action(
        self,
        state,
        epsilon,
    ):
        if random.random() < epsilon:
            return random.randrange(
                self.action_size
            )

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.online_network.eval()

        with torch.no_grad():
            q_values = self.online_network(
                state_tensor
            )

            action = int(
                q_values.argmax(
                    dim=1
                ).item()
            )

        self.online_network.train()

        return action

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def learn(self):
        if (
            len(self.replay_buffer)
            < self.batch_size
        ):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        current_q = self.online_network(
            states
        ).gather(
            1,
            actions,
        )

        with torch.no_grad():
            # Double-DQN action selection.
            next_actions = self.online_network(
                next_states
            ).argmax(
                dim=1,
                keepdim=True,
            )

            next_q = self.target_network(
                next_states
            ).gather(
                1,
                next_actions,
            )

            target_q = (
                rewards
                + self.gamma
                * next_q
                * (1.0 - dones)
            )

        loss = self.loss_function(
            current_q,
            target_q,
        )

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.learning_steps += 1

        if (
            self.learning_steps
            % self.target_update_interval
            == 0
        ):
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

            self.target_network.eval()

        return float(loss.item())

    def save(
        self,
        model_path,
        metadata=None,
    ):
        model_path = Path(model_path)

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "online_network":
                    self.online_network.state_dict(),
                "target_network":
                    self.target_network.state_dict(),
                "optimizer":
                    self.optimizer.state_dict(),
                "state_size":
                    self.state_size,
                "action_size":
                    self.action_size,
                "learning_steps":
                    self.learning_steps,
                "metadata":
                    metadata or {},
            },
            model_path,
        )

    def load(self, model_path):
        try:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
            )

        if (
            checkpoint["state_size"]
            != self.state_size
        ):
            raise ValueError(
                "Checkpoint state size does not "
                "match this network."
            )

        if (
            checkpoint["action_size"]
            != self.action_size
        ):
            raise ValueError(
                "Checkpoint action size does not "
                "match this network."
            )

        self.online_network.load_state_dict(
            checkpoint["online_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.online_network.eval()
        self.target_network.eval()

        self.learning_steps = checkpoint.get(
            "learning_steps",
            0,
        )

        return checkpoint.get(
            "metadata",
            {},
        )


# ============================================================
# SUMO traffic-light environment
# ============================================================

class HighwayCityEnvironment:
    def __init__(
        self,
        config_file,
        max_steps,
        use_gui=False,
        scale=1.0,
        seed=42,
        gui_delay=100,
    ):
        self.config_file = str(
            Path(config_file).resolve()
        )

        self.max_steps = max_steps
        self.use_gui = use_gui
        self.scale = scale
        self.seed = seed
        self.gui_delay = gui_delay

        self.current_green = 0
        self.pending_green = None

        self.green_elapsed = 0
        self.yellow_remaining = 0

        self.simulation_step = 0
        self.total_arrived = 0

        self.previous_total_waiting = 0.0

        self.started = False

    def _start_sumo(self):
        binary_name = (
            "sumo-gui"
            if self.use_gui
            else "sumo"
        )

        binary = sumolib.checkBinary(
            binary_name
        )

        command = [
            binary,
            "-c",
            self.config_file,
            "--seed",
            str(self.seed),
            "--scale",
            str(self.scale),
            "--start",
            "--quit-on-end",
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "-1",
            "--waiting-time-memory",
            "1000",
        ]

        if self.use_gui:
            command.extend(
                [
                    "--delay",
                    str(self.gui_delay),
                ]
            )

        traci.start(command)
        self.started = True

    def _validate_network(self):
        traffic_lights = list(
            traci.trafficlight.getIDList()
        )

        if TRAFFIC_LIGHT_ID not in traffic_lights:
            raise RuntimeError(
                f"Traffic light {TRAFFIC_LIGHT_ID!r} "
                f"was not found. Available lights: "
                f"{traffic_lights}"
            )

        state = (
            traci.trafficlight
            .getRedYellowGreenState(
                TRAFFIC_LIGHT_ID
            )
        )

        if len(state) != len(
            HIGHWAY_GREEN_STATE
        ):
            raise RuntimeError(
                f"Traffic-light state length is "
                f"{len(state)}, but this controller "
                f"expects {len(HIGHWAY_GREEN_STATE)}."
            )

        available_lanes = set(
            traci.lane.getIDList()
        )

        required_lanes = set(
            HIGHWAY_INCOMING_LANES
            + CITY_INCOMING_LANES
        )

        missing = (
            required_lanes
            - available_lanes
        )

        if missing:
            raise RuntimeError(
                f"Missing expected incoming lanes: "
                f"{sorted(missing)}"
            )

    def reset(self, seed=None):
        self.close()

        if seed is not None:
            self.seed = seed

        self.current_green = 0
        self.pending_green = None
        self.green_elapsed = 0
        self.yellow_remaining = 0

        self.simulation_step = 0
        self.total_arrived = 0
        self.previous_total_waiting = 0.0

        self._start_sumo()
        self._validate_network()

        traci.trafficlight.setRedYellowGreenState(
            TRAFFIC_LIGHT_ID,
            HIGHWAY_GREEN_STATE,
        )

        # Short warm-up.
        for _ in range(5):
            traci.simulationStep()

            self.simulation_step += 1
            self.green_elapsed += 1

            self.total_arrived += (
                traci.simulation
                .getArrivedNumber()
            )

        self.previous_total_waiting = (
            self._total_waiting_time()
        )

        return self.get_state()

    @staticmethod
    def _vehicle_count(lanes):
        return float(
            sum(
                traci.lane
                .getLastStepVehicleNumber(
                    lane
                )
                for lane in lanes
            )
        )

    @staticmethod
    def _queue_count(lanes):
        return float(
            sum(
                traci.lane
                .getLastStepHaltingNumber(
                    lane
                )
                for lane in lanes
            )
        )

    @staticmethod
    def _waiting_time(lanes):
        return float(
            sum(
                traci.lane.getWaitingTime(
                    lane
                )
                for lane in lanes
            )
        )

    def _total_waiting_time(self):
        return (
            self._waiting_time(
                HIGHWAY_INCOMING_LANES
            )
            + self._waiting_time(
                CITY_INCOMING_LANES
            )
        )

    def get_state(self):
        highway_vehicles = (
            self._vehicle_count(
                HIGHWAY_INCOMING_LANES
            )
        )

        city_vehicles = (
            self._vehicle_count(
                CITY_INCOMING_LANES
            )
        )

        highway_queue = self._queue_count(
            HIGHWAY_INCOMING_LANES
        )

        city_queue = self._queue_count(
            CITY_INCOMING_LANES
        )

        highway_wait = self._waiting_time(
            HIGHWAY_INCOMING_LANES
        )

        city_wait = self._waiting_time(
            CITY_INCOMING_LANES
        )

        # Normalization keeps DQN inputs within
        # numerically manageable ranges.
        state = np.asarray(
            [
                min(
                    highway_vehicles / 100.0,
                    1.0,
                ),
                min(
                    city_vehicles / 30.0,
                    1.0,
                ),
                min(
                    highway_queue / 100.0,
                    1.0,
                ),
                min(
                    city_queue / 30.0,
                    1.0,
                ),
                min(
                    highway_wait / 1000.0,
                    1.0,
                ),
                min(
                    city_wait / 1000.0,
                    1.0,
                ),
                float(self.current_green),
                min(
                    self.green_elapsed
                    / MAX_GREEN_DURATION,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

        return state

    def _start_yellow(self, target_green):
        if self.current_green == 0:
            yellow_state = (
                HIGHWAY_YELLOW_STATE
            )
        else:
            yellow_state = (
                CITY_YELLOW_STATE
            )

        traci.trafficlight.setRedYellowGreenState(
            TRAFFIC_LIGHT_ID,
            yellow_state,
        )

        self.pending_green = target_green
        self.yellow_remaining = (
            YELLOW_DURATION
        )

    def _activate_pending_green(self):
        if self.pending_green == 0:
            next_state = (
                HIGHWAY_GREEN_STATE
            )
        else:
            next_state = CITY_GREEN_STATE

        traci.trafficlight.setRedYellowGreenState(
            TRAFFIC_LIGHT_ID,
            next_state,
        )

        self.current_green = (
            self.pending_green
        )

        self.pending_green = None
        self.yellow_remaining = 0
        self.green_elapsed = 0

    def _request_action(self, action):
        target_green = int(action)

        if self.yellow_remaining > 0:
            return

        must_switch = (
            self.green_elapsed
            >= MAX_GREEN_DURATION
        )

        wants_switch = (
            target_green
            != self.current_green
        )

        if must_switch:
            target_green = (
                1 - self.current_green
            )
            wants_switch = True

        if not wants_switch:
            return

        if (
            self.green_elapsed
            < MIN_GREEN_DURATION
        ):
            return

        self._start_yellow(
            target_green
        )

    def _advance_signal_timer(self):
        if self.yellow_remaining > 0:
            self.yellow_remaining -= 1

            if self.yellow_remaining == 0:
                self._activate_pending_green()
        else:
            self.green_elapsed += 1

    def _calculate_reward(self):
        total_waiting = (
            self._total_waiting_time()
        )

        added_waiting = max(
            total_waiting
            - self.previous_total_waiting,
            0.0,
        )

        self.previous_total_waiting = (
            total_waiting
        )

        total_queue = (
            self._queue_count(
                HIGHWAY_INCOMING_LANES
            )
            + self._queue_count(
                CITY_INCOMING_LANES
            )
        )

        reward = -(
            WAITING_REWARD_WEIGHT
            * added_waiting
            + QUEUE_REWARD_WEIGHT
            * total_queue
        )

        return float(
            np.clip(
                reward,
                -2000.0,
                0.0,
            )
        )

    def step(self, action):
        self._request_action(action)

        total_reward = 0.0

        for _ in range(
            DECISION_INTERVAL
        ):
            traci.simulationStep()

            self.simulation_step += 1

            self.total_arrived += (
                traci.simulation
                .getArrivedNumber()
            )

            self._advance_signal_timer()

            total_reward += (
                self._calculate_reward()
            )

            if self.is_done():
                break

        next_state = self.get_state()
        done = self.is_done()
        info = self.get_metrics()

        return (
            next_state,
            total_reward,
            done,
            info,
        )

    def is_done(self):
        return (
            self.simulation_step
            >= self.max_steps
            or traci.simulation
                .getMinExpectedNumber()
            <= 0
        )

    def get_metrics(self):
        highway_queue = self._queue_count(
            HIGHWAY_INCOMING_LANES
        )

        city_queue = self._queue_count(
            CITY_INCOMING_LANES
        )

        highway_wait = self._waiting_time(
            HIGHWAY_INCOMING_LANES
        )

        city_wait = self._waiting_time(
            CITY_INCOMING_LANES
        )

        vehicles = list(
            traci.vehicle.getIDList()
        )

        if vehicles:
            mean_speed = float(
                np.mean(
                    [
                        traci.vehicle.getSpeed(
                            vehicle
                        )
                        for vehicle in vehicles
                    ]
                )
            )
        else:
            mean_speed = 0.0

        return {
            "step":
                self.simulation_step,
            "highway_queue":
                highway_queue,
            "city_queue":
                city_queue,
            "total_queue":
                highway_queue + city_queue,
            "highway_waiting":
                highway_wait,
            "city_waiting":
                city_wait,
            "total_waiting":
                highway_wait + city_wait,
            "active_vehicles":
                len(vehicles),
            "arrived_vehicles":
                self.total_arrived,
            "mean_speed":
                mean_speed,
            "current_green":
                (
                    "highway"
                    if self.current_green == 0
                    else "city"
                ),
        }

    def close(self):
        if self.started:
            try:
                traci.close()
            except Exception:
                pass

            self.started = False


# ============================================================
# Epsilon scheduling
# ============================================================

def calculate_epsilon(
    episode,
    start,
    end,
    decay,
):
    value = end + (
        start - end
    ) * np.exp(
        -episode / decay
    )

    return float(value)


# ============================================================
# Training
# ============================================================

def train(args):
    set_seeds(args.seed)

    Path("models").mkdir(
        exist_ok=True
    )

    Path("plots").mkdir(
        exist_ok=True
    )

    Path("outputs").mkdir(
        exist_ok=True
    )

    model_path = (
        Path("models")
        / f"{args.model_name}.pt"
    )

    metrics_path = (
        Path("outputs")
        / f"{args.model_name}_metrics.csv"
    )

    agent = DQNAgent(
        STATE_SIZE,
        ACTION_SIZE,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_interval=(
            args.target_update
        ),
    )

    print(
        f"PyTorch device: {agent.device}"
    )

    env = HighwayCityEnvironment(
        config_file=args.config,
        max_steps=args.steps,
        use_gui=args.gui,
        scale=args.scale,
        seed=args.seed,
        gui_delay=args.gui_delay,
    )

    best_reward = -float("inf")

    fieldnames = [
        "episode",
        "epsilon",
        "reward",
        "mean_loss",
        "max_total_queue",
        "final_total_waiting",
        "arrived_vehicles",
        "mean_speed",
    ]

    try:
        with metrics_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for episode in range(
                args.episodes
            ):
                state = env.reset(
                    args.seed + episode
                )

                epsilon = calculate_epsilon(
                    episode,
                    args.epsilon_start,
                    args.epsilon_end,
                    args.epsilon_decay,
                )

                done = False
                episode_reward = 0.0
                losses = []
                maximum_queue = 0.0
                info = {}

                while not done:
                    action = agent.choose_action(
                        state,
                        epsilon,
                    )

                    (
                        next_state,
                        reward,
                        done,
                        info,
                    ) = env.step(action)

                    agent.remember(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    )

                    loss = agent.learn()

                    if loss is not None:
                        losses.append(loss)

                    state = next_state
                    episode_reward += reward

                    maximum_queue = max(
                        maximum_queue,
                        info["total_queue"],
                    )

                mean_loss = (
                    float(np.mean(losses))
                    if losses
                    else 0.0
                )

                row = {
                    "episode":
                        episode + 1,
                    "epsilon":
                        epsilon,
                    "reward":
                        episode_reward,
                    "mean_loss":
                        mean_loss,
                    "max_total_queue":
                        maximum_queue,
                    "final_total_waiting":
                        info["total_waiting"],
                    "arrived_vehicles":
                        info["arrived_vehicles"],
                    "mean_speed":
                        info["mean_speed"],
                }

                writer.writerow(row)
                file.flush()

                print(
                    f"Episode "
                    f"{episode + 1:03d}/"
                    f"{args.episodes:03d} | "
                    f"Reward="
                    f"{episode_reward:11.1f} | "
                    f"Epsilon={epsilon:.3f} | "
                    f"Max queue="
                    f"{maximum_queue:6.1f} | "
                    f"Arrived="
                    f"{info['arrived_vehicles']} | "
                    f"Loss={mean_loss:.5f}"
                )

                if (
                    episode_reward
                    > best_reward
                ):
                    best_reward = (
                        episode_reward
                    )

                    agent.save(
                        model_path,
                        metadata={
                            "traffic_light":
                                TRAFFIC_LIGHT_ID,
                            "highway_lanes":
                                HIGHWAY_INCOMING_LANES,
                            "city_lanes":
                                CITY_INCOMING_LANES,
                            "scale":
                                args.scale,
                        },
                    )

                env.close()

    finally:
        env.close()

    plot_training(
        metrics_path,
        args.model_name,
    )

    print(
        f"\nBest model: "
        f"{model_path.resolve()}"
    )

    print(
        f"Metrics: "
        f"{metrics_path.resolve()}"
    )


# ============================================================
# Testing
# ============================================================

def test(args):
    model_path = (
        Path("models")
        / f"{args.model_name}.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    agent = DQNAgent(
        STATE_SIZE,
        ACTION_SIZE,
    )

    metadata = agent.load(
        model_path
    )

    print(
        f"Loaded model: {model_path}"
    )

    print(
        f"Metadata: {metadata}"
    )

    env = HighwayCityEnvironment(
        config_file=args.config,
        max_steps=args.steps,
        use_gui=True,
        scale=args.scale,
        seed=args.seed,
        gui_delay=args.gui_delay,
    )

    try:
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(
                state,
                epsilon=0.0,
            )

            (
                state,
                reward,
                done,
                info,
            ) = env.step(action)

            total_reward += reward

            if (
                info["step"] % 100 == 0
                or done
            ):
                print(
                    f"Step={info['step']:4d} | "
                    f"Green="
                    f"{info['current_green']:7s} | "
                    f"Highway queue="
                    f"{info['highway_queue']:5.0f} | "
                    f"City queue="
                    f"{info['city_queue']:5.0f} | "
                    f"Waiting="
                    f"{info['total_waiting']:8.1f}"
                )

        print("\nTest completed")
        print(
            f"Total reward: "
            f"{total_reward:.2f}"
        )

        for key, value in info.items():
            print(f"{key}: {value}")

    finally:
        env.close()


# ============================================================
# Plotting
# ============================================================

def plot_training(
    metrics_path,
    model_name,
):
    data = np.genfromtxt(
        metrics_path,
        delimiter=",",
        names=True,
    )

    if data.size == 0:
        return

    data = np.atleast_1d(data)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    axes[0, 0].plot(
        data["episode"],
        data["reward"],
    )

    axes[0, 0].set_title(
        "Episode reward"
    )

    axes[0, 1].plot(
        data["episode"],
        data["mean_loss"],
    )

    axes[0, 1].set_title(
        "Mean DQN loss"
    )

    axes[1, 0].plot(
        data["episode"],
        data["max_total_queue"],
    )

    axes[1, 0].set_title(
        "Maximum total queue"
    )

    axes[1, 1].plot(
        data["episode"],
        data["arrived_vehicles"],
    )

    axes[1, 1].set_title(
        "Arrived vehicles"
    )

    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(True, alpha=0.3)

    figure.tight_layout()

    output = (
        Path("plots")
        / f"{model_name}_training.png"
    )

    figure.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# CLI
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "DQN traffic-light controller for the "
            "highway-city intersection."
        )
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a model",
    )

    parser.add_argument(
        "-m",
        "--model-name",
        default="highway_dqn",
    )

    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=50,
    )

    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=3600,
    )

    parser.add_argument(
        "-c",
        "--config",
        default=str(CONFIG_FILE),
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--target-update",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show GUI during training",
    )

    parser.add_argument(
        "--gui-delay",
        type=int,
        default=100,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.train:
        train(arguments)
    else:
        test(arguments)