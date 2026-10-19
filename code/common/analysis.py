"""Per-episode phase timings and failure categories shared by the dataset
inspection, student evaluation, and failure analysis."""

import numpy as np

LIFT_Z = 0.045      # box counts as lifted once its centre is 1.5 cm above the rest height
GRASP_DIST = 0.02   # gripper site within 2 cm of the box centre
NEAR_MISS_DIST = 0.10

# Ordered from earliest to latest stage of the task.
CATEGORIES = ["success", "no_grasp", "grasp_no_lift", "dropped", "near_miss", "wrong_place",
              "out_of_bounds"]


def first_true(mask):
    """Index of the first True along the last axis, or -1."""
    idx = np.argmax(mask, axis=-1)
    return np.where(mask.any(axis=-1), idx, -1)


def phase_steps(out):
    """Control step (1-based, -1 = never) at which the gripper first reaches the
    box and the box is first lifted. `out` is a rollout dict or demo dataset."""
    box, grip = out["box_pos"][:, 1:], out["gripper_pos"][:, 1:]
    grasp = first_true(np.linalg.norm(box - grip, axis=-1) < GRASP_DIST)
    lift = first_true(box[..., 2] > LIFT_Z)
    return np.where(grasp >= 0, grasp + 1, -1), np.where(lift >= 0, lift + 1, -1)


def failure_category(out):
    """One of CATEGORIES per episode.

    no_grasp       the gripper never gets within GRASP_DIST of the box
    grasp_no_lift  reaches the box but never lifts it
    dropped        lifts the box but it is back on the table at the end
    near_miss      holds the box at the end, 5-10 cm from the target
    wrong_place    holds the box at the end, >= 10 cm from the target
    out_of_bounds  the episode terminated early (box left the workspace)
    """
    grasp, lift = phase_steps(out)
    final_z = out["box_pos"][:, -1, 2]
    early = out["dones"].any(axis=1)
    cats = []
    for i in range(len(grasp)):
        if out["success"][i]:
            c = "success"
        elif early[i]:
            c = "out_of_bounds"
        elif grasp[i] < 0:
            c = "no_grasp"
        elif lift[i] < 0:
            c = "grasp_no_lift"
        elif final_z[i] <= LIFT_Z:
            c = "dropped"
        elif out["final_dist"][i] < NEAR_MISS_DIST:
            c = "near_miss"
        else:
            c = "wrong_place"
        cats.append(c)
    return np.array(cats)

