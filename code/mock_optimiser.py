import os
import time
import logging
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import bluesky.plan_stubs as bps
from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
from bluesky.run_engine import RunEngine
from bluesky.callbacks.tiled_writer import TiledWriter
from bluesky.utils import plan

from tiled.client import from_uri
from tiled.server import SimpleTiledServer

from blop.ax import Agent, RangeDOF, Objective
import warnings

import blop
print(blop.__version__)

logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="ax")

CHECKPOINT = "mock_agent_checkpoint.json"

# Mock device classes
class AlwaysSuccessfulStatus(Status):

    def add_callback(self, callback) -> None:
        callback(self)

    def exception(self, timeout=0.0):
        return None

    @property
    def done(self) -> bool:
        return True

    @property
    def success(self) -> bool:
        return True


class ReadableSignal(Readable, HasHints, HasParent):

    def __init__(self, name: str) -> None:
        self._name = name
        self._value = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def hints(self) -> Hints:
        return {"fields": [self._name], "dimensions": [], "gridding": "rectilinear"}

    @property
    def parent(self) -> Any | None:
        return None

    def read(self):
        return {self._name: {"value": self._value, "timestamp": time.time()}}

    def describe(self):
        return {self._name: {"source": self._name, "dtype": "number", "shape": []}}


class MovableSignal(ReadableSignal, NamedMovable):

    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name)
        self._value: float = initial_value

    def set(self, value: float) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()


# Tiled / RunEngine setup
RE = RunEngine({})

if 'tiled_server' not in globals() or not hasattr(tiled_server, 'uri'):
    tiled_server = SimpleTiledServer()
    tiled_client = from_uri(tiled_server.uri)
    tiled_writer = TiledWriter(tiled_client)
    print("Created new Tiled server and RunEngine")

RE.subscribe(tiled_writer)

# Spectrum simulator
x_peak = 13.0
y_peak = 44.0
theta1_peak = 37.0
sx = 5.0
sy = 5.0
stheta1 = 15.0
A0 = 1000.0
peak_width = 50.0
E0 = 650.0
noise_factor = 0

energy_range = (649.5, 650.5)


def create_dummy_spectrum_generator(
    x_peak=x_peak,
    y_peak=y_peak,
    theta1_peak=theta1_peak,
    sx=sx,
    sy=sy,
    stheta1=stheta1,
    A0=A0,
    peak_width=peak_width,
    E0=E0,
    noise_factor=noise_factor,
):
    energy_grid = np.linspace(0, 1024, 2048)

    def generate(x, y, theta1):
        A = A0 * np.exp(
            -((x - x_peak) ** 2 / (2 * sx ** 2)
              + (y - y_peak) ** 2 / (2 * sy ** 2)
              + (theta1 - theta1_peak) ** 2 / (2 * stheta1 ** 2))
        )
        peak_shape = np.exp(-((energy_grid - E0) ** 2) / (2 * peak_width ** 2))
        spectrum = A * peak_shape
        if noise_factor > 0:
            spectrum += np.random.normal(0, noise_factor, size=len(energy_grid))
        return energy_grid, spectrum

    return generate


spectrum_generator = create_dummy_spectrum_generator()


# Mock actuators and sensor
x_motor = MovableSignal("x", initial_value=0.0)
y_motor = MovableSignal("y", initial_value=0.0)
theta_motor = MovableSignal("theta", initial_value=0.0)

# BLOP requires at least one sensor even though the evaluation function reads directly from tiled rather than from the sensor itself
dummy_sensor = ReadableSignal("dummy_sensor")


# Acquisition plan: move all motors simultaneously then read
# trying to use low-level plan stubs to avoid opening a nested run inside BLOP's
# outer run wrapper -- bp.list_scan cannot be used here for that reason


# @plan
# def simultaneous_acquire(suggestions, actuators, sensors=None, **kwargs):
#     if sensors is None:
#         sensors = []

#     readables = []
#     for s in sensors:
#         if hasattr(s, "read"):
#             readables.append(s)

#     for suggestion in suggestions:
#         move_args = []
#         for actuator in actuators:
#             move_args.extend([actuator, suggestion[actuator.name]])
#         yield from bps.mv(*move_args)


# Evaluation function
# reads the last run from tiled since the installed BLOP version does not
# pass the uid correctly to the evaluation function


def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
    outcomes = []

    for suggestion in suggestions:
        idx = suggestion["_id"]
        x = suggestion["x"]
        y = suggestion["y"]
        theta = suggestion["theta"]

        energy_grid, spectrum = spectrum_generator(x, y, theta)

        mask = (energy_grid >= energy_range[0]) & (energy_grid <= energy_range[1])
        integrated_amplitude = np.trapezoid(spectrum[mask], energy_grid[mask])

        outcomes.append({"peak_amplitude": integrated_amplitude, "_id": idx})

    return outcomes


# Main plan
def blop_peak_scan(iterations: int = 15):

    dofs = [
        RangeDOF(actuator=x_motor, bounds=(0, 50), parameter_type="float"),
        RangeDOF(actuator=y_motor, bounds=(0, 50), parameter_type="float"),
        RangeDOF(actuator=theta_motor, bounds=(0, 90), parameter_type="float"),
    ]

    objectives = [Objective(name="peak_amplitude", minimize=False)]

    motors = [dof.actuator.name for dof in dofs]

    # checkpoint: resume if available, otherwise start fresh
    if os.path.exists(CHECKPOINT):
        print("Checkpoint found, resuming from saved model.")
        agent = Agent.from_checkpoint(
            CHECKPOINT,
            sensors=[dummy_sensor],
            actuators=[x_motor, y_motor, theta_motor],
            evaluation_function=evaluation_function,
            # acquisition_plan=simultaneous_acquire,
        )
        agent._readable_cache = {}
    else:
        print("No checkpoint found, creating new agent.")
        agent = Agent(
            sensors=[dummy_sensor],
            dofs=dofs,
            objectives=objectives,
            evaluation_function=evaluation_function,
            # acquisition_plan=simultaneous_acquire,
            checkpoint_path=CHECKPOINT,
            name="mock-peak-scan",
        )

    # live plot setup

    # plt.ion()
    # fig_live, ax_live = plt.subplots(figsize=(7, 4))
    # ax_live.set_xlabel("Trial")
    # ax_live.set_ylabel("Peak amplitude")
    # ax_live.set_title("live optimization progress")
    # ax_live.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    # def update_live_plot():
    #     summary = agent.ax_client.summarize()
    #     if len(summary) == 0:
    #         return
    #     signal = summary["peak_amplitude"].values
    #     trials = np.arange(len(signal))
    #     best_so_far = np.maximum.accumulate(signal)
    #     ax_live.cla()
    #     ax_live.set_xlabel("Trial")
    #     ax_live.set_ylabel("Peak amplitude")
    #     ax_live.set_title("Optimisation in progress")
    #     ax_live.grid(True, alpha=0.3)
    #     ax_live.plot(trials, signal, marker="o", color="darkgreen", label="signal")
    #     ax_live.plot(trials, best_so_far, linestyle="--", color="darkorange", label="best so far")
    #     ax_live.legend(loc="lower right")
    #     fig_live.canvas.draw()
    #     fig_live.canvas.flush_events()

    # subscribe live plot callback to RE stop documents

    # def on_stop(name, doc):
    #     if name == "stop":
    #         update_live_plot()
    # token = RE.subscribe(on_stop)


    yield from agent.optimize(iterations=iterations)

    # unsubscribe and close live plot

    # RE.unsubscribe(token)
    # plt.ioff()
    # plt.close(fig_live)

    # save checkpoint
    agent.checkpoint()
    print(f"Checkpoint saved to {CHECKPOINT}")

    # final history plot
    
    # def plot_optimization_history():
    #     summary = agent.ax_client.summarize()
    #     trial_index = summary.index
    #     signal = summary["peak_amplitude"].values
    #     best_so_far = np.maximum.accumulate(signal)
    #     n_rows = 2 + len(motors)
    #     fig = plt.figure(figsize=(10, 3 * n_rows))
    #     gs = gridspec.GridSpec(n_rows, 1, hspace=0.5)
    #     ax0 = fig.add_subplot(gs[0])
    #     ax0.plot(trial_index, signal, marker="o", linestyle="-", color="darkgreen")
    #     ax0.set_ylabel("Peak amplitude")
    #     ax0.set_title("Signal per trial")
    #     ax0.grid(True, alpha=0.3)
    #     ax1 = fig.add_subplot(gs[1])
    #     ax1.plot(trial_index, best_so_far, marker=".", linestyle="-", color="darkorange")
    #     ax1.set_ylabel("Best signal so far")
    #     ax1.set_title("Convergence")
    #     ax1.grid(True, alpha=0.3)
    #     for i, motor_name in enumerate(motors):
    #         ax = fig.add_subplot(gs[2 + i])
    #         positions = summary[motor_name].values
    #         sc = ax.scatter(trial_index, positions, c=signal, cmap="viridis", s=60, zorder=3)
    #         ax.plot(trial_index, positions, color="gray", linewidth=0.8, alpha=0.5)
    #         plt.colorbar(sc, ax=ax, label="signal")
    #         ax.set_ylabel(motor_name)
    #         ax.set_title(f"{motor_name} position per trial")
    #         ax.grid(True, alpha=0.3)
    #     ax.set_xlabel("Trial index")
    #     plt.suptitle("Optimisation history", fontsize=13, y=1.01)
    #     plt.show()
    # plot_optimization_history()

    # move to best position and return
    summary = agent.ax_client.summarize()
    print(summary[["x", "y", "theta", "peak_amplitude"]].to_string())

    best_idx = summary["peak_amplitude"].idxmax()
    best_x = summary.loc[best_idx, "x"]
    best_y = summary.loc[best_idx, "y"]
    best_theta = summary.loc[best_idx, "theta"]
    best_eval = summary.loc[best_idx, "peak_amplitude"]

    best_positions = {name: summary.loc[best_idx, name] for name in motors}


    print(f"Best x: {best_x}")
    print(f"Best y: {best_y}")
    print(f"Best theta: {best_theta}")
    print(f"Best evaluation value: {best_eval}")

    for motor_name, value in best_positions.items():
        motor = {"x": x_motor, "y": y_motor, "theta": theta_motor}[motor_name]
        motor.set(value)

    return best_eval, best_positions




if __name__ == "__main__":

    # live plot figure must be created in the main thread before RE runs
    # plt.ion()
    # fig_live, ax_live = plt.subplots(figsize=(7, 4))
    # ax_live.set_xlabel("Trial")
    # ax_live.set_ylabel("Peak amplitude")
    # ax_live.set_title("Optimisation in progress")
    # ax_live.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    result = RE(blop_peak_scan(iterations=30))
    tiled_server.close()