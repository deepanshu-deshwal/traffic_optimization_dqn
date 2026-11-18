import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


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


# Import the trained model and network definitions.
from train import (
    ACTION_SIZE,
    CITY_GREEN_STATE,
    CITY_INCOMING_LANES,
    CITY_YELLOW_STATE,
    DECISION_INTERVAL,
    DQNAgent,
    HIGHWAY_GREEN_STATE,
    HIGHWAY_INCOMING_LANES,
    HIGHWAY_YELLOW_STATE,
    MAX_GREEN_DURATION,
    MIN_GREEN_DURATION,
    STATE_SIZE,
    TRAFFIC_LIGHT_ID,
    YELLOW_DURATION,
)


ALL_INCOMING_LANES = (
    HIGHWAY_INCOMING_LANES
    + CITY_INCOMING_LANES
)


# ============================================================
# General helpers
# ============================================================

def start_sumo(
    config_file,
    scale,
    seed,
    label,
):
    binary = sumolib.checkBinary("sumo")

    command = [
        binary,
        "-c",
        str(Path(config_file).resolve()),
        "--scale",
        str(scale),
        "--seed",
        str(seed),
        "--no-step-log",
        "true",
        "--time-to-teleport",
        "-1",
        "--waiting-time-memory",
        "10000",
    ]

    traci.start(
        command,
        label=label,
    )

    return traci.getConnection(label)


def get_queue(connection, lanes):
    return float(
        sum(
            connection.lane
            .getLastStepHaltingNumber(lane)
            for lane in lanes
        )
    )


def get_lane_waiting(connection, lanes):
    return float(
        sum(
            connection.lane.getWaitingTime(lane)
            for lane in lanes
        )
    )


def get_state(
    connection,
    current_green,
    green_elapsed,
):
    highway_vehicles = sum(
        connection.lane
        .getLastStepVehicleNumber(lane)
        for lane in HIGHWAY_INCOMING_LANES
    )

    city_vehicles = sum(
        connection.lane
        .getLastStepVehicleNumber(lane)
        for lane in CITY_INCOMING_LANES
    )

    highway_queue = get_queue(
        connection,
        HIGHWAY_INCOMING_LANES,
    )

    city_queue = get_queue(
        connection,
        CITY_INCOMING_LANES,
    )

    highway_wait = get_lane_waiting(
        connection,
        HIGHWAY_INCOMING_LANES,
    )

    city_wait = get_lane_waiting(
        connection,
        CITY_INCOMING_LANES,
    )

    return np.asarray(
        [
            min(highway_vehicles / 100.0, 1.0),
            min(city_vehicles / 30.0, 1.0),
            min(highway_queue / 100.0, 1.0),
            min(city_queue / 30.0, 1.0),
            min(highway_wait / 1000.0, 1.0),
            min(city_wait / 1000.0, 1.0),
            float(current_green),
            min(
                green_elapsed
                / MAX_GREEN_DURATION,
                1.0,
            ),
        ],
        dtype=np.float32,
    )


# ============================================================
# Measurement collector
# ============================================================

class MetricsCollector:
    def __init__(self):
        self.queue_series = []
        self.highway_queue_series = []
        self.city_queue_series = []

        self.cumulative_waiting_series = []

        self.vehicle_waiting = {}
        self.vehicle_time_loss = {}

        self.completed_vehicle_ids = set()

        self.vehicle_speed_sum = 0.0
        self.vehicle_speed_samples = 0

        self.departed_vehicles = 0
        self.arrived_vehicles = 0

    def update(self, connection):
        highway_queue = get_queue(
            connection,
            HIGHWAY_INCOMING_LANES,
        )

        city_queue = get_queue(
            connection,
            CITY_INCOMING_LANES,
        )

        total_queue = (
            highway_queue
            + city_queue
        )

        self.highway_queue_series.append(
            highway_queue
        )

        self.city_queue_series.append(
            city_queue
        )

        self.queue_series.append(
            total_queue
        )

        self.departed_vehicles += (
            connection.simulation
            .getDepartedNumber()
        )

        arrived_ids = list(
            connection.simulation
            .getArrivedIDList()
        )

        self.arrived_vehicles += len(
            arrived_ids
        )

        for vehicle_id in arrived_ids:
            self.completed_vehicle_ids.add(
                vehicle_id
            )

        active_vehicle_ids = list(
            connection.vehicle.getIDList()
        )

        for vehicle_id in active_vehicle_ids:
            waiting_time = (
                connection.vehicle
                .getAccumulatedWaitingTime(
                    vehicle_id
                )
            )

            time_loss = (
                connection.vehicle
                .getTimeLoss(vehicle_id)
            )

            speed = (
                connection.vehicle
                .getSpeed(vehicle_id)
            )

            # Keep the latest value last observed for
            # every vehicle.
            self.vehicle_waiting[vehicle_id] = (
                float(waiting_time)
            )

            self.vehicle_time_loss[vehicle_id] = (
                float(time_loss)
            )

            self.vehicle_speed_sum += float(
                speed
            )

            self.vehicle_speed_samples += 1

        completed_waiting = sum(
            self.vehicle_waiting.get(
                vehicle_id,
                0.0,
            )
            for vehicle_id
            in self.completed_vehicle_ids
        )

        self.cumulative_waiting_series.append(
            float(completed_waiting)
        )

    def summarize(self):
        completed_waiting_values = [
            self.vehicle_waiting.get(
                vehicle_id,
                0.0,
            )
            for vehicle_id
            in self.completed_vehicle_ids
        ]

        completed_time_loss_values = [
            self.vehicle_time_loss.get(
                vehicle_id,
                0.0,
            )
            for vehicle_id
            in self.completed_vehicle_ids
        ]

        total_waiting = float(
            sum(completed_waiting_values)
        )

        total_time_loss = float(
            sum(completed_time_loss_values)
        )

        mean_waiting = (
            total_waiting
            / self.arrived_vehicles
            if self.arrived_vehicles > 0
            else 0.0
        )

        mean_time_loss = (
            total_time_loss
            / self.arrived_vehicles
            if self.arrived_vehicles > 0
            else 0.0
        )

        mean_speed = (
            self.vehicle_speed_sum
            / self.vehicle_speed_samples
            if self.vehicle_speed_samples > 0
            else 0.0
        )

        return {
            "departed_vehicles":
                int(self.departed_vehicles),

            "arrived_vehicles":
                int(self.arrived_vehicles),

            "unfinished_vehicles":
                int(
                    self.departed_vehicles
                    - self.arrived_vehicles
                ),

            "total_waiting_time_seconds":
                total_waiting,

            "mean_waiting_time_seconds":
                mean_waiting,

            "total_time_loss_seconds":
                total_time_loss,

            "mean_time_loss_seconds":
                mean_time_loss,

            "average_queue_vehicles":
                float(
                    np.mean(
                        self.queue_series
                    )
                )
                if self.queue_series
                else 0.0,

            "maximum_queue_vehicles":
                float(
                    np.max(
                        self.queue_series
                    )
                )
                if self.queue_series
                else 0.0,

            "average_highway_queue":
                float(
                    np.mean(
                        self.highway_queue_series
                    )
                )
                if self.highway_queue_series
                else 0.0,

            "average_city_queue":
                float(
                    np.mean(
                        self.city_queue_series
                    )
                )
                if self.city_queue_series
                else 0.0,

            "mean_speed_mps":
                float(mean_speed),

            "mean_speed_kph":
                float(mean_speed * 3.6),
        }


# ============================================================
# Fixed-time controller
# ============================================================

def run_fixed_controller(args):
    print("\nRunning fixed-time controller...")

    connection = start_sumo(
        config_file=args.config,
        scale=args.scale,
        seed=args.seed,
        label="fixed-controller",
    )

    collector = MetricsCollector()

    try:
        for _ in range(args.steps):
            connection.simulationStep()
            collector.update(connection)

    finally:
        connection.close()

    results = collector.summarize()

    return results, collector


# ============================================================
# RL controller
# ============================================================

def run_rl_controller(args):
    print("\nRunning RL controller...")

    model_path = (
        Path("models")
        / f"{args.model_name}.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: "
            f"{model_path.resolve()}"
        )

    agent = DQNAgent(
        STATE_SIZE,
        ACTION_SIZE,
    )

    metadata = agent.load(model_path)

    print(f"Loaded model: {model_path}")
    print(f"Model metadata: {metadata}")
    print(f"Device: {agent.device}")

    connection = start_sumo(
        config_file=args.config,
        scale=args.scale,
        seed=args.seed,
        label="rl-controller",
    )

    collector = MetricsCollector()

    current_green = 0
    pending_green = None

    green_elapsed = 0
    yellow_remaining = 0

    current_action = 0

    connection.trafficlight.setRedYellowGreenState(
        TRAFFIC_LIGHT_ID,
        HIGHWAY_GREEN_STATE,
    )

    try:
        for step in range(args.steps):
            # Select a new action every decision interval.
            if (
                step % DECISION_INTERVAL == 0
                and yellow_remaining == 0
            ):
                state = get_state(
                    connection,
                    current_green,
                    green_elapsed,
                )

                requested_action = (
                    agent.choose_action(
                        state,
                        epsilon=0.0,
                    )
                )

                must_switch = (
                    green_elapsed
                    >= MAX_GREEN_DURATION
                )

                wants_switch = (
                    requested_action
                    != current_green
                )

                if must_switch:
                    requested_action = (
                        1 - current_green
                    )

                    wants_switch = True

                if (
                    wants_switch
                    and green_elapsed
                    >= MIN_GREEN_DURATION
                ):
                    if current_green == 0:
                        yellow_state = (
                            HIGHWAY_YELLOW_STATE
                        )
                    else:
                        yellow_state = (
                            CITY_YELLOW_STATE
                        )

                    connection.trafficlight.setRedYellowGreenState(
                        TRAFFIC_LIGHT_ID,
                        yellow_state,
                    )

                    pending_green = (
                        requested_action
                    )

                    yellow_remaining = (
                        YELLOW_DURATION
                    )

            connection.simulationStep()

            if yellow_remaining > 0:
                yellow_remaining -= 1

                if yellow_remaining == 0:
                    if pending_green == 0:
                        green_state = (
                            HIGHWAY_GREEN_STATE
                        )
                    else:
                        green_state = (
                            CITY_GREEN_STATE
                        )

                    connection.trafficlight.setRedYellowGreenState(
                        TRAFFIC_LIGHT_ID,
                        green_state,
                    )

                    current_green = (
                        pending_green
                    )

                    current_action = (
                        current_green
                    )

                    pending_green = None
                    green_elapsed = 0

            else:
                green_elapsed += 1

            collector.update(connection)

    finally:
        connection.close()

    results = collector.summarize()

    return results, collector


# ============================================================
# Percentage calculations
# ============================================================

def lower_is_better_improvement(
    fixed_value,
    rl_value,
):
    if fixed_value == 0:
        return None

    return (
        100.0
        * (fixed_value - rl_value)
        / fixed_value
    )


def higher_is_better_improvement(
    fixed_value,
    rl_value,
):
    if fixed_value == 0:
        return None

    return (
        100.0
        * (rl_value - fixed_value)
        / fixed_value
    )


def calculate_improvements(
    fixed_results,
    rl_results,
):
    return {
        "waiting_time_reduction_percent":
            lower_is_better_improvement(
                fixed_results[
                    "mean_waiting_time_seconds"
                ],
                rl_results[
                    "mean_waiting_time_seconds"
                ],
            ),

        "total_waiting_reduction_percent":
            lower_is_better_improvement(
                fixed_results[
                    "total_waiting_time_seconds"
                ],
                rl_results[
                    "total_waiting_time_seconds"
                ],
            ),

        "average_queue_reduction_percent":
            lower_is_better_improvement(
                fixed_results[
                    "average_queue_vehicles"
                ],
                rl_results[
                    "average_queue_vehicles"
                ],
            ),

        "maximum_queue_reduction_percent":
            lower_is_better_improvement(
                fixed_results[
                    "maximum_queue_vehicles"
                ],
                rl_results[
                    "maximum_queue_vehicles"
                ],
            ),

        "time_loss_reduction_percent":
            lower_is_better_improvement(
                fixed_results[
                    "mean_time_loss_seconds"
                ],
                rl_results[
                    "mean_time_loss_seconds"
                ],
            ),

        "mean_speed_improvement_percent":
            higher_is_better_improvement(
                fixed_results[
                    "mean_speed_kph"
                ],
                rl_results[
                    "mean_speed_kph"
                ],
            ),

        "throughput_improvement_percent":
            higher_is_better_improvement(
                fixed_results[
                    "arrived_vehicles"
                ],
                rl_results[
                    "arrived_vehicles"
                ],
            ),
    }


# ============================================================
# Save output files
# ============================================================

def save_summary_csv(
    fixed_results,
    rl_results,
    output_path,
):
    rows = []

    for metric in fixed_results:
        rows.append(
            {
                "metric": metric,
                "fixed_time":
                    fixed_results[metric],
                "reinforcement_learning":
                    rl_results[metric],
            }
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "fixed_time",
                "reinforcement_learning",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def save_time_series_csv(
    fixed_collector,
    rl_collector,
    output_path,
):
    number_of_steps = min(
        len(fixed_collector.queue_series),
        len(rl_collector.queue_series),
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "simulation_step",
                "fixed_total_queue",
                "rl_total_queue",
                "fixed_highway_queue",
                "rl_highway_queue",
                "fixed_city_queue",
                "rl_city_queue",
                "fixed_cumulative_waiting",
                "rl_cumulative_waiting",
            ]
        )

        for index in range(number_of_steps):
            writer.writerow(
                [
                    index + 1,

                    fixed_collector
                    .queue_series[index],

                    rl_collector
                    .queue_series[index],

                    fixed_collector
                    .highway_queue_series[index],

                    rl_collector
                    .highway_queue_series[index],

                    fixed_collector
                    .city_queue_series[index],

                    rl_collector
                    .city_queue_series[index],

                    fixed_collector
                    .cumulative_waiting_series[index],

                    rl_collector
                    .cumulative_waiting_series[index],
                ]
            )


# ============================================================
# Report-ready plots
# ============================================================

def create_plots(
    fixed_results,
    rl_results,
    fixed_collector,
    rl_collector,
    output_path,
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    controllers = [
        "Fixed-Time",
        "RL Controller",
    ]

    colors = [
        "#D9534F",
        "#2E86C1",
    ]

    # Plot 1: Mean waiting time
    waiting_values = [
        fixed_results[
            "mean_waiting_time_seconds"
        ],
        rl_results[
            "mean_waiting_time_seconds"
        ],
    ]

    axes[0, 0].bar(
        controllers,
        waiting_values,
        color=colors,
    )

    axes[0, 0].set_title(
        "Mean Waiting Time per Completed Vehicle"
    )

    axes[0, 0].set_ylabel(
        "Waiting time (seconds)"
    )

    # Plot 2: Average and maximum queues
    x_positions = np.arange(2)
    width = 0.35

    average_queues = [
        fixed_results[
            "average_queue_vehicles"
        ],
        rl_results[
            "average_queue_vehicles"
        ],
    ]

    maximum_queues = [
        fixed_results[
            "maximum_queue_vehicles"
        ],
        rl_results[
            "maximum_queue_vehicles"
        ],
    ]

    axes[0, 1].bar(
        x_positions - width / 2,
        average_queues,
        width,
        label="Average queue",
        color="#5DADE2",
    )

    axes[0, 1].bar(
        x_positions + width / 2,
        maximum_queues,
        width,
        label="Maximum queue",
        color="#AF7AC5",
    )

    axes[0, 1].set_xticks(
        x_positions,
        controllers,
    )

    axes[0, 1].set_ylabel(
        "Halted vehicles"
    )

    axes[0, 1].set_title(
        "Queue-Length Comparison"
    )

    axes[0, 1].legend()

    # Plot 3: Queue over time
    fixed_queue = np.asarray(
        fixed_collector.queue_series
    )

    rl_queue = np.asarray(
        rl_collector.queue_series
    )

    # Smooth with a 60-step moving average.
    smoothing_window = 60

    kernel = np.ones(
        smoothing_window
    ) / smoothing_window

    fixed_smoothed = np.convolve(
        fixed_queue,
        kernel,
        mode="same",
    )

    rl_smoothed = np.convolve(
        rl_queue,
        kernel,
        mode="same",
    )

    axes[1, 0].plot(
        fixed_smoothed,
        label="Fixed-Time",
        color=colors[0],
    )

    axes[1, 0].plot(
        rl_smoothed,
        label="RL Controller",
        color=colors[1],
    )

    axes[1, 0].set_title(
        "Total Queue Over Simulation"
    )

    axes[1, 0].set_xlabel(
        "Simulation step"
    )

    axes[1, 0].set_ylabel(
        "Halted vehicles, 60-step mean"
    )

    axes[1, 0].legend()

    # Plot 4: Throughput and speed
    throughput_values = [
        fixed_results[
            "arrived_vehicles"
        ],
        rl_results[
            "arrived_vehicles"
        ],
    ]

    speed_values = [
        fixed_results[
            "mean_speed_kph"
        ],
        rl_results[
            "mean_speed_kph"
        ],
    ]

    throughput_axis = axes[1, 1]

    throughput_bars = (
        throughput_axis.bar(
            x_positions - width / 2,
            throughput_values,
            width,
            label="Arrived vehicles",
            color="#58D68D",
        )
    )

    throughput_axis.set_ylabel(
        "Arrived vehicles"
    )

    throughput_axis.set_xticks(
        x_positions,
        controllers,
    )

    speed_axis = (
        throughput_axis.twinx()
    )

    speed_bars = speed_axis.bar(
        x_positions + width / 2,
        speed_values,
        width,
        label="Mean speed",
        color="#F5B041",
    )

    speed_axis.set_ylabel(
        "Mean speed (km/h)"
    )

    throughput_axis.set_title(
        "Throughput and Mean Speed"
    )

    handles_1, labels_1 = (
        throughput_axis
        .get_legend_handles_labels()
    )

    handles_2, labels_2 = (
        speed_axis
        .get_legend_handles_labels()
    )

    throughput_axis.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="best",
    )

    for axis in axes.flat:
        axis.grid(
            True,
            axis="y",
            alpha=0.25,
        )

    figure.suptitle(
        "Fixed-Time vs Reinforcement-Learning Traffic Control",
        fontsize=15,
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Console report
# ============================================================

def print_report(
    fixed_results,
    rl_results,
    improvements,
):
    print("\n" + "=" * 70)
    print("FIXED-TIME VS REINFORCEMENT-LEARNING COMPARISON")
    print("=" * 70)

    print(
        "\nMean waiting time per completed vehicle:"
    )

    print(
        f"  Fixed-Time: "
        f"{fixed_results['mean_waiting_time_seconds']:.2f} s"
    )

    print(
        f"  RL:         "
        f"{rl_results['mean_waiting_time_seconds']:.2f} s"
    )

    reduction = improvements[
        "waiting_time_reduction_percent"
    ]

    if reduction is not None:
        print(
            f"  Reduction:  {reduction:.2f}%"
        )

    print("\nAverage queue:")

    print(
        f"  Fixed-Time: "
        f"{fixed_results['average_queue_vehicles']:.2f} vehicles"
    )

    print(
        f"  RL:         "
        f"{rl_results['average_queue_vehicles']:.2f} vehicles"
    )

    queue_reduction = improvements[
        "average_queue_reduction_percent"
    ]

    if queue_reduction is not None:
        print(
            f"  Reduction:  {queue_reduction:.2f}%"
        )

    print("\nMaximum queue:")

    print(
        f"  Fixed-Time: "
        f"{fixed_results['maximum_queue_vehicles']:.0f}"
    )

    print(
        f"  RL:         "
        f"{rl_results['maximum_queue_vehicles']:.0f}"
    )

    print("\nMean vehicle speed:")

    print(
        f"  Fixed-Time: "
        f"{fixed_results['mean_speed_kph']:.2f} km/h"
    )

    print(
        f"  RL:         "
        f"{rl_results['mean_speed_kph']:.2f} km/h"
    )

    print("\nThroughput:")

    print(
        f"  Fixed-Time: "
        f"{fixed_results['arrived_vehicles']} vehicles"
    )

    print(
        f"  RL:         "
        f"{rl_results['arrived_vehicles']} vehicles"
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model-name",
        default="highway_dqn_100",
    )

    parser.add_argument(
        "-c",
        "--config",
        default=r"maps\configuration.sumocfg",
    )

    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=3600,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    Path("outputs").mkdir(
        exist_ok=True
    )

    Path("plots").mkdir(
        exist_ok=True
    )

    (
        fixed_results,
        fixed_collector,
    ) = run_fixed_controller(args)

    (
        rl_results,
        rl_collector,
    ) = run_rl_controller(args)

    improvements = calculate_improvements(
        fixed_results,
        rl_results,
    )

    results = {
        "experiment": {
            "model": args.model_name,
            "steps": args.steps,
            "scale": args.scale,
            "seed": args.seed,
        },
        "fixed_time": fixed_results,
        "reinforcement_learning":
            rl_results,
        "improvements_percent":
            improvements,
    }

    json_path = (
        Path("outputs")
        / "controller_comparison.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    summary_csv_path = (
        Path("outputs")
        / "controller_comparison.csv"
    )

    save_summary_csv(
        fixed_results,
        rl_results,
        summary_csv_path,
    )

    time_series_path = (
        Path("outputs")
        / "controller_time_series.csv"
    )

    save_time_series_csv(
        fixed_collector,
        rl_collector,
        time_series_path,
    )

    plot_path = (
        Path("plots")
        / "fixed_vs_rl_comparison.png"
    )

    create_plots(
        fixed_results,
        rl_results,
        fixed_collector,
        rl_collector,
        plot_path,
    )

    print_report(
        fixed_results,
        rl_results,
        improvements,
    )

    print("\nFiles created:")

    print(
        f"  {json_path.resolve()}"
    )

    print(
        f"  {summary_csv_path.resolve()}"
    )

    print(
        f"  {time_series_path.resolve()}"
    )

    print(
        f"  {plot_path.resolve()}"
    )


if __name__ == "__main__":
    main()