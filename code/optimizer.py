import numpy as np
import os
from blop.ax import Agent, Objective, RangeDOF
from bluesky_queueserver import parameter_annotation_decorator

from beamlinetools.beamline_config.base import db
from beamlinetools.beamline_config.beamline import devices_dictionary
from beamlinetools.devices.manipulator import SISSY1Manipulator, SISSY1ManipulatorAxis
from beamlinetools.devices.peak_controller import PeakController

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.utils import plan


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


CHECKPOINT = "sissy1_agent_checkpoint.json"
#path to checkpoint file json at SISSY1 control machine


def get_optimization_method(
    energy_range: tuple[float, float],
    peak_spectrum_x_dkey: str,
    peak_spectrum_y_dkey: str,
): 
    def evaluation_function(
        uid: str, suggestions: list[dict], *args, **kwargs
    ) -> list[dict]:
        # for the integration of the spectrum in the range
        # i'd like the spectrum counts from the IOC as the PV, The Eval function can be just THAT for blop to maximise.
        uid = str(list(db.keys())[-1])
        run = db[uid]
        outcomes = []
        spectrum_data = np.array(run["primary"]["data"][peak_spectrum_x_dkey][-1])

        energy_grid = np.array(run["primary"]["data"][peak_spectrum_y_dkey][-1])
        for suggestion in suggestions:
            idx = suggestion["_id"]

            spectrum = np.array(spectrum_data)

            mask = (energy_grid >= energy_range[0]) & (energy_grid <= energy_range[1])

            if np.sum(mask) == 0:
                print(f"No data points in energy range {energy_range}")
                integrated_signal = 0.0
            else:
                energy_in_range = energy_grid[mask]
                spectrum_in_range = spectrum[mask]
                integrated_signal = np.trapezoid(spectrum_in_range, energy_in_range)

            outcomes.append({"integrated_signal": float(integrated_signal), "_id": idx})

        return outcomes

    return evaluation_function


'''
BLOP's default acquisition plan is to move to the suggested point sequentially
rather than simultaneously-independently, then read
'''
@plan
def simultaneous_acquire(suggestions, actuators, sensors=None, **kwargs):
    if sensors is None:
        sensors = []

    readables = []
    for s in sensors:
        if hasattr(s, "read"):
            readables.append(s)
    md = {"blop_suggestions": suggestions}

    def per_step(detectors, step, pos_cache):  # step is a dict of {motor: value}
        move_args = []
        for motor, position in step.items():
            move_args.extend([motor, position])
        yield from bps.mv(*move_args)
        yield from bps.trigger_and_read(list(detectors) + list(step.keys()))

    # build list_scan args
    plan_args = []
    for actuator in actuators:
        plan_args.append(actuator)
        plan_args.append([s[actuator.name] for s in suggestions])

    return (
        yield from bp.list_scan(
            readables,
            *plan_args,
            per_step=per_step,
            md=md,
            **kwargs,
        )
    )


@parameter_annotation_decorator(
    {
        "parameters": {
            "iterations": {},
            "energy_start": {"max": 10000, "min": 0},
            "energy_stop": {"max": 10000, "min": 0},
            "use_x": {"default": False},
            "x_start": {"default": 0.0, "min": -4000, "max": 4000},
            "x_end": {"default": 0.0, "min": -4000, "max": 4000},
            "use_y": {"default": False},
            "y_start": {"default": 0.0, "min": -4000, "max": 4000},
            "y_end": {"default": 0.0, "min": -4000, "max": 4000},
            "use_z": {"default": False},
            "z_start": {"default": 0.0, "min": -4000, "max": 4000},
            "z_end": {"default": 0.0, "min": -4000, "max": 4000},
            "use_r1": {"default": False},
            "r1_start": {"default": 0.0, "min": -90, "max": 90},
            "r1_end": {"default": 0.0, "min": -90, "max": 90},
            "use_r2": {"default": False},
            "r2_start": {"default": 0.0, "min": -90, "max": 90},
            "r2_end": {"default": 0.0, "min": -90, "max": 90},
        }
    }
)
def blop_peak_scan(
    iterations: int,
    energy_start: float,
    energy_stop: float,
    use_x: bool = False,
    x_start: float = 0.0,
    x_end: float = 0.0,
    use_y: bool = False,
    y_start: float = 0.0,
    y_end: float = 0.0,
    use_z: bool = False,
    z_start: float = 0.0,
    z_end: float = 0.0,
    use_r1: bool = False,
    r1_start: float = 0.0,
    r1_end: float = 0.0,
    use_r2: bool = False,
    r2_start: float = 0.0,
    r2_end: float = 0.0,
):


    peak: PeakController = devices_dictionary.get("peak_controller")
    manipulator: SISSY1Manipulator = devices_dictionary.get("manipulator")
    if peak is None or manipulator is None:
        raise RuntimeError(
            "Required devices 'peak' or 'manipulator' are not available."
        )

    evaluation_function = get_optimization_method(
        energy_range=(energy_start, energy_stop),
        peak_spectrum_x_dkey=peak.analyser.spectrum.data.name,
        peak_spectrum_y_dkey=peak.analyser.spectrum.axis.name,
    )

    # Ensure that the bounds are correctly ordered.
    (x_start, x_end) = (min(x_start, x_end), max(x_start, x_end))
    (y_start, y_end) = (min(y_start, y_end), max(y_start, y_end))
    (z_start, z_end) = (min(z_start, z_end), max(z_start, z_end))
    (r1_start, r1_end) = (min(r1_start, r1_end), max(r1_start, r1_end))
    (r2_start, r2_end) = (min(r2_start, r2_end), max(r2_start, r2_end))

    motors, dofs = [], []

    if use_x:
        motors.append(manipulator.x.name)
        dofs.append(
            RangeDOF(
                movable=manipulator.x, bounds=(x_start, x_end), parameter_type="float"
            )
        )
        print("Using x with bounds:", (x_start, x_end))
    if use_y:
        motors.append(manipulator.y.name)
        dofs.append(
            RangeDOF(
                movable=manipulator.y, bounds=(y_start, y_end), parameter_type="float"
            )
        )
        print("Using y with bounds:", (y_start, y_end))
    if use_z:
        motors.append(manipulator.z.name)
        dofs.append(
            RangeDOF(
                movable=manipulator.z, bounds=(z_start, z_end), parameter_type="float"
            )
        )
        print("Using z with bounds:", (z_start, z_end))
    if use_r1:
        motors.append(manipulator.r1.name)
        dofs.append(
            RangeDOF(
                movable=manipulator.r1,
                bounds=(r1_start, r1_end),
                parameter_type="float",
            )
        )
        print("Using r1 with bounds:", (r1_start, r1_end))
    if use_r2:
        motors.append(manipulator.r2.name)
        dofs.append(
            RangeDOF(
                movable=manipulator.r2,
                bounds=(r2_start, r2_end),
                parameter_type="float",
            )
        )
        print("Using r2 with bounds:", (r2_start, r2_end))

    objectives = [
        Objective(name="integrated_signal", minimize=False),
    ]

    if os.path.exists(CHECKPOINT):
        print("Loading agent from checkpoint.")
        agent = Agent.from_checkpoint(
            CHECKPOINT,
            sensors=[peak],
            actuators=[dof.actuator for dof in dofs],
            evaluation_function=evaluation_function,
            acquisition_plan=simultaneous_acquire,
        )
        print(f"Loaded agent with {len(agent.ax_client._experiment.trials)} previous trials.")
    else:
        print("No checkpoint found, creating new agent.")
        agent = Agent(
            sensors=[peak],
            dofs=dofs,
            objectives=objectives,
            evaluation_function=evaluation_function,
            acquisition_plan=simultaneous_acquire,
            checkpoint_path=CHECKPOINT,
            name="peak-sample-map",
            description="Get peak to find the best position for the sample given the energy range specified.",
        )
        print("Created new agent with no previous trials.")

    '''
    live plotting setup to visualize optimization progress,
    integrated siganal and best so far -- as a fucntion of trial number
    '''


    plt.ion()
    fig_live, ax_live = plt.subplots(figsize=(7, 4))
    ax_live.set_xlabel("Trials")
    ax_live.set_ylabel("Integrated signal")
    ax_live.set_title("live optimization progress")
    ax_live.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    def update_live_plot():
        summary = agent.ax_client.summarize()
        if len(summary) == 0:
            return
        signal = summary["integrated_signal"].values
        trials = np.arange(len(signal))
        best_so_far = np.maximum.accumulate(signal)  # overlay the best so far curve to see convergence

        ax_live.cla()
        ax_live.set_xlabel("Trials")
        ax_live.set_ylabel("Integrated signal")
        ax_live.set_title("live optimization progress")
        ax_live.grid(True, alpha=0.3)
        ax_live.plot(trials, signal, marker="o", color="blue", label="signal")
        ax_live.plot(trials, best_so_far, linestyle="--", color="darkorange", label="best so far")
        ax_live.legend(loc="lower right")
        fig_live.canvas.draw()
        fig_live.canvas.flush_events()

    '''
    workaround for the actuator bug, run the optimization loop,
    if motor already at 0 - skip,
    yielding messages back to the caller after each suggestion and after each run completion (so the live plot can be updated)
    '''

    for message in agent.optimize(iterations=iterations):
        if message.command == "set" and message.args == (0.0,):
            if (
                isinstance(message.obj, SISSY1ManipulatorAxis)
                and message.obj.readback.get() == 0.0
            ):
                continue
            else:
                yield message
        else:
            yield message

        # we want to update the live plot after every run completion, which corresponds to a "close_run" message from the agent
        if message.command == "close_run":
            update_live_plot()

    plt.ioff()
    plt.close(fig_live)

    # final checkpoint to save the state of the agent after optimization is complete
    agent.checkpoint()


    # final history plot 
    def plot_optimization_history(agent, motors):
        summary = agent.ax_client.summarize()
        trial_index = summary.index
        signal = summary["integrated_signal"].values
        best_so_far = np.maximum.accumulate(signal)

        n_rows = 2 + len(motors)
        fig = plt.figure(figsize=(10, 3 * n_rows))
        gs = gridspec.GridSpec(n_rows, 1, hspace=0.5)

        ax0 = fig.add_subplot(gs[0])
        ax0.plot(trial_index, signal, marker="o", linestyle="-", color="steelblue")
        ax0.set_ylabel("Integrated signal")
        ax0.set_title("Signal per trial")
        ax0.grid(True, alpha=0.3)

        ax1 = fig.add_subplot(gs[1])
        ax1.plot(trial_index, best_so_far, marker=".", linestyle="-", color="darkorange")
        ax1.set_ylabel("Best signal so far")
        ax1.set_title("Convergence")
        ax1.grid(True, alpha=0.3)

        for i, motor_name in enumerate(motors):
            ax = fig.add_subplot(gs[2 + i])
            positions = summary[motor_name].values
            sc = ax.scatter(trial_index, positions, c=signal, cmap="viridis", s=60, zorder=3)
            ax.plot(trial_index, positions, color="gray", linewidth=0.8, alpha=0.5)
            plt.colorbar(sc, ax=ax, label="signal")
            ax.set_ylabel(motor_name)
            ax.set_title(f"{motor_name} position per trial")
            ax.grid(True, alpha=0.3)

        ax.set_xlabel("Trial index")
        plt.suptitle("Optimisation history", fontsize=13, y=1.01)
        plt.show()

    plot_optimization_history(agent, motors)

    # move to best and return
    summary = agent.ax_client.summarize()
    best_idx = summary["integrated_signal"].idxmax()
    best_signal = summary.loc[best_idx, "integrated_signal"]
    best_postions = {name: summary.loc[best_idx, name] for name in motors}

    for motor_name, value in best_postions.items():
        getattr(manipulator, motor_name.split("_")[-1]).set(value, wait=True)

    return best_signal, best_postions
