import argparse
import time
from pathlib import Path

import serial

from train import (
    ACTION_SIZE,
    CITY_GREEN_STATE,
    CITY_YELLOW_STATE,
    CONFIG_FILE,
    DQNAgent,
    HIGHWAY_GREEN_STATE,
    HIGHWAY_YELLOW_STATE,
    STATE_SIZE,
    HighwayCityEnvironment,
)


class ArduinoOutput:
    def __init__(
        self,
        port,
        baudrate=115200,
    ):
        self.connection = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=0.2,
            write_timeout=1.0,
        )

        self.last_command = None

        # Opening USB serial usually resets an Arduino Uno.
        time.sleep(2.0)

        self.send(
            "H:R,C:R",
            force=True,
        )

        print(
            f"Arduino connected on {port} "
            f"at {baudrate} baud."
        )

    def send(
        self,
        command,
        force=False,
    ):
        if (
            not force
            and command == self.last_command
        ):
            return

        message = (
            command + "\n"
        ).encode("utf-8")

        self.connection.write(message)
        self.connection.flush()

        self.last_command = command

        print(f"Arduino <- {command}")

    def close(self):
        if self.connection is not None:
            try:
                self.send(
                    "H:R,C:R",
                    force=True,
                )
                time.sleep(0.05)
            finally:
                self.connection.close()
                self.connection = None


class HardwareEnvironment(
    HighwayCityEnvironment
):
    """
    Extends the existing SUMO environment without changing
    train.py. Every SUMO signal change is mirrored to Arduino.
    """

    def __init__(
        self,
        *args,
        arduino,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.arduino = arduino

    def reset(self, seed=None):
        state = super().reset(seed)

        # The base environment starts with highway green.
        self.arduino.send("H:G,C:R")

        return state

    def _start_yellow(
        self,
        target_green,
    ):
        # current_green identifies the phase that is ending.
        if self.current_green == 0:
            self.arduino.send("H:Y,C:R")
        else:
            self.arduino.send("H:R,C:Y")

        super()._start_yellow(
            target_green
        )

    def _activate_pending_green(self):
        target_green = self.pending_green

        super()._activate_pending_green()

        if target_green == 0:
            self.arduino.send("H:G,C:R")
        else:
            self.arduino.send("H:R,C:G")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained SUMO DQN controller and "
            "mirror its traffic lights to Arduino."
        )
    )

    parser.add_argument(
        "--arduino-port",
        required=True,
        help="Arduino COM port, for example COM5",
    )

    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
    )

    parser.add_argument(
        "-m",
        "--model-name",
        default="highway_dqn_100",
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
        "--gui-delay",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    model_path = (
        Path("models")
        / f"{args.model_name}.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: "
            f"{model_path.resolve()}"
        )

    arduino = ArduinoOutput(
        port=args.arduino_port,
        baudrate=args.baudrate,
    )

    agent = DQNAgent(
        STATE_SIZE,
        ACTION_SIZE,
    )

    metadata = agent.load(model_path)

    print(f"Loaded model: {model_path}")
    print(f"Model metadata: {metadata}")
    print(f"PyTorch device: {agent.device}")

    environment = HardwareEnvironment(
        config_file=CONFIG_FILE,
        max_steps=args.steps,
        use_gui=True,
        scale=args.scale,
        seed=args.seed,
        gui_delay=args.gui_delay,
        arduino=arduino,
    )

    try:
        state = environment.reset(
            seed=args.seed
        )

        done = False
        total_reward = 0.0

        while not done:
            # No exploration during hardware demonstration.
            action = agent.choose_action(
                state,
                epsilon=0.0,
            )

            (
                state,
                reward,
                done,
                info,
            ) = environment.step(action)

            total_reward += reward

            if (
                info["step"] % 100 == 0
                or done
            ):
                print(
                    f"Step={info['step']:4d} | "
                    f"Green={info['current_green']:7s} | "
                    f"Highway queue="
                    f"{info['highway_queue']:5.0f} | "
                    f"City queue="
                    f"{info['city_queue']:5.0f} | "
                    f"Waiting="
                    f"{info['total_waiting']:8.1f}"
                )

        print("\nHardware demonstration completed.")
        print(f"Total reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\nHardware demonstration stopped.")

    finally:
        environment.close()
        arduino.close()


if __name__ == "__main__":
    main()
