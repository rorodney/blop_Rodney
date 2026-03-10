import numpy as np
from blop.ax import Agent, Objective, RangeDOF
from bluesky_queueserver import parameter_annotation_decorator

from beamlinetools.beamline_config.base import db
from beamlinetools.beamline_config.beamline import devices_dictionary
from beamlinetools.devices.manipulator import SISSY1Manipulator, SISSY1ManipulatorAxis
from beamlinetools.devices.peak_controller import PeakController

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.utils import plan

LAST_AGENT: Agent | None = None


def get_optimization_method(
    energy_range: tuple[float, float],
    peak_spectrum_x_dkey: str,
    peak_spectrum_y_dkey: str,
):
    def evaluation_function(
        uid: str, suggestions: list[dict], *args, **kwargs
    ) -> list[dict]:
        # for the integration of the spectrum in the range

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


@plan
def simultaneous_acquire(suggestions, actuators, sensors=None, **kwargs):
    if sensors is None:
        sensors = []

    readables = [s for s in sensors if hasattr(s, "read")]
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

    print("Creating new agent.")
    agent = Agent(
        sensors=[peak],                          
        dofs=dofs,
        objectives=objectives,
        evaluation_function=evaluation_function, 
        acquisition_plan=simultaneous_acquire,  
        name="peak-sample-map",
        description="Get peak to find the best position for the sample given the energy range specified.",
    )
    agent.dofs

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

    summary = agent.ax_client.summarize()
    table = summary[motors + ["integrated_signal"]]

    best_idx = summary["integrated_signal"].idxmax()
    best_signal = summary.loc[best_idx, "integrated_signal"]
    best_postions = {name: summary.loc[best_idx, name] for name in motors}

    for motor_name, value in best_postions.items():
        getattr(manipulator, motor_name.split("_")[-1]).set(value, wait=True)
    return best_signal, best_postions