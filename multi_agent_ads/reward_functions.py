"""
File: reward_functions.py

Description:
  Reward function utilities for multi-agent RL in CARLA.
  Each agent gets an individual reward that depends on:
    - Collision occurrence (boolean or collision severity)
    - Speed (encouraging or penalizing certain speeds)
    - Lane deviation (distance from center line or desired path)
    - TTC (Time To Collision) and PET (Post-Encroachment Time)
      as additional safety metrics

Usage in environment step():
  from utils.reward_functions import compute_multi_agent_rewards

  # after collecting per-agent data:
  collisions = [...]
  speeds = [...]
  lane_deviations = [...]
  ttcs = [...]
  pets = [...]

  rewards_dict = compute_multi_agent_rewards(
      collisions=collisions,
      speeds=speeds,
      lane_deviations=lane_deviations,
      ttcs=ttcs,
      pets=pets
  )
  # then you can return rewards_dict in step() or sum them for a shared reward if desired.
"""

import numpy as np


def penalize_ttc(ttc: float, threshold: float = 3.0, factor: float = -1.0) -> float:
    """
    Time-To-Collision penalty.
    If ttc < 0 or extremely small => heavy penalty
    If 0 < ttc < threshold => mild to moderate penalty
    If ttc >= threshold => no penalty
    :param ttc: time to collision in seconds (estimated)
    :param threshold: below this, penalize
    :param factor: negative scale factor for penalty
    :return: penalty value (negative or zero)
    """
    if ttc < 0.0:
        # negative ttc usually implies collision or invalid
        return factor * threshold  # strong penalty
    elif ttc < threshold:
        # linear penalty from ttc=0 to ttc=threshold
        # e.g. if ttc=0 => factor * threshold
        # if ttc=2 => factor*(threshold-2)
        return factor * (threshold - ttc)
    else:
        return 0.0


def penalize_pet(pet: float, threshold: float = 2.0, factor: float = -0.5) -> float:
    """
    Post-Encroachment Time (PET) penalty or partial reward.
    - If pet is small => high risk => negative
    - If pet is large => no penalty
    :param pet: post-encroachment time in seconds
    :param threshold: below this => penalize
    :param factor: negative scale factor
    :return: penalty value
    """
    if pet < 0.0:
        # negative PET might indicate overlap or collision event
        return factor * (abs(pet) + threshold)
    elif pet < threshold:
        return factor * (threshold - pet)
    else:
        return 0.0


def compute_agent_reward(
        collision: bool,
        speed: float,
        lane_deviation: float,
        ttc: float,
        pet: float,
        desired_speed: float = 10.0,
        collision_penalty: float = -10.0,
        speed_factor: float = 0.1,
        lane_dev_factor: float = -0.05,
        ttc_threshold: float = 3.0,
        ttc_factor: float = -1.0,
        pet_threshold: float = 2.0,
        pet_factor: float = -0.5
) -> float:
    """
    Compute the reward for a single agent in one time step.
    :param collision: whether the agent has collided in this step
    :param speed: current speed (m/s) or chosen unit
    :param lane_deviation: distance from center of lane (m)
    :param ttc: time to collision
    :param pet: post-encroachment time
    :param desired_speed: speed at which we get the highest (speed-based) reward
    :param collision_penalty: fixed penalty for collision
    :param speed_factor: coefficient for speed-based reward
    :param lane_dev_factor: coefficient for lane deviation penalty
    :param ttc_threshold: threshold below which TTC is penalized
    :param ttc_factor: negative factor for TTC penalty
    :param pet_threshold: threshold below which PET is penalized
    :param pet_factor: negative factor for PET penalty
    :return: single scalar reward
    """

    # 1) Collision penalty
    coll_term = collision_penalty if collision else 0.0

    # 2) Speed reward (encourage to drive near desired_speed, cap at desired_speed)
    #    For example: reward ~ speed_factor * min(speed, desired_speed)
    speed_term = speed_factor * min(speed, desired_speed)

    # 3) Lane deviation penalty
    #    e.g. lane_dev_factor < 0, so bigger lane_deviation => bigger negative
    lane_term = lane_dev_factor * lane_deviation

    # 4) TTC penalty
    ttc_term = 0.0
    if ttc_factor != 0.0:
        ttc_term = penalize_ttc(ttc, threshold=ttc_threshold, factor=ttc_factor)

    # 5) PET penalty
    pet_term = 0.0
    if pet_factor != 0.0:
        pet_term = penalize_pet(pet, threshold=pet_threshold, factor=pet_factor)

    total_reward = coll_term + speed_term + lane_term + ttc_term + pet_term
    return total_reward


def compute_multi_agent_rewards(
        collisions,
        speeds,
        lane_deviations,
        ttcs,
        pets,
        desired_speed=10.0,
        collision_penalty=-10.0,
        speed_factor=0.1,
        lane_dev_factor=-0.05,
        ttc_threshold=3.0,
        ttc_factor=-1.0,
        pet_threshold=2.0,
        pet_factor=-0.5
) -> dict:
    """
    For a multi-agent scenario, compute each agent's reward individually
    and return a dictionary { "agent_0": r0, "agent_1": r1, ... }.

    :param collisions: list/array of bool, one per agent
    :param speeds: list/array of float, one per agent
    :param lane_deviations: list/array of float, one per agent
    :param ttcs: list/array of float, one per agent
    :param pets: list/array of float, one per agent
    :return: reward_dict = { "agent_i": reward_i, ... }

    Example usage in environment step():
      collisions = [False, True, ...]
      speeds = [5.0, 8.2, ...]
      lane_devs = [0.3, 0.1, ...]
      ttcs = [2.5, 3.1, ...]
      pets = [1.5, 2.2, ...]
      rewards = compute_multi_agent_rewards(
          collisions, speeds, lane_devs, ttcs, pets
      )
    """
    num_agents = len(collisions)
    reward_dict = {}
    for i in range(num_agents):
        agent_id = f"agent_{i}"
        r = compute_agent_reward(
            collision=collisions[i],
            speed=speeds[i],
            lane_deviation=lane_deviations[i],
            ttc=ttcs[i],
            pet=pets[i],
            desired_speed=desired_speed,
            collision_penalty=collision_penalty,
            speed_factor=speed_factor,
            lane_dev_factor=lane_dev_factor,
            ttc_threshold=ttc_threshold,
            ttc_factor=ttc_factor,
            pet_threshold=pet_threshold,
            pet_factor=pet_factor
        )
        reward_dict[agent_id] = r
    return reward_dict


# ---------------------
# Example usage snippet
# ---------------------
if __name__ == "__main__":
    # A quick test/demonstration
    # Suppose we have 2 agents with the following states:
    collisions_demo = [False, True]
    speeds_demo = [7.5, 12.3]  # m/s
    lane_devs_demo = [0.2, 1.0]  # m
    ttcs_demo = [2.0, 1.5]  # s
    pets_demo = [3.0, 1.0]  # s

    multi_rewards = compute_multi_agent_rewards(
        collisions=collisions_demo,
        speeds=speeds_demo,
        lane_deviations=lane_devs_demo,
        ttcs=ttcs_demo,
        pets=pets_demo,
        desired_speed=10.0
    )
    print("Demo multi-agent rewards:", multi_rewards)
    # e.g. => { "agent_0": 0.XX, "agent_1": -X.XX }
