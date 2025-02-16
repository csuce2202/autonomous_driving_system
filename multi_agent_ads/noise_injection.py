"""
File: noise_injection.py

Description:
  Utility functions/classes to inject noise or random disturbances into:
    - Actions (e.g., Gaussian noise)
    - Observations (e.g., sensor noise)
    - Vehicle / environment parameters (e.g., random friction, mass variations)

Usage:
  In your environment or agent code, you can do:
    from utils.noise_injection import ActionNoise, ObservationNoise

    noise = ActionNoise(sigma=0.02)
    noisy_action = noise.apply(original_action)

    obs_noise = ObservationNoise(sigma=0.05)
    noisy_obs = obs_noise.apply(original_observation)

  Or define more advanced logic for parameter randomization.

Notes:
  - This is a skeleton; adapt or extend for your actual research needs (e.g., Ornstein-Uhlenbeck noise).
"""

import numpy as np


class ActionNoise:
    """
    Inject Gaussian noise into continuous actions (e.g. throttle, steer, brake).
    """

    def __init__(self, sigma=0.01):
        """
        :param sigma: standard deviation of Gaussian noise
        """
        self.sigma = sigma

    def apply(self, action: np.ndarray) -> np.ndarray:
        """
        :param action: shape (action_dim,) in [-1,1] or [0,1], etc.
        :return: action + gaussian noise
        """
        noisy_action = action + np.random.randn(*action.shape) * self.sigma
        # optionally clip to valid range
        return np.clip(noisy_action, -1.0, 1.0)


class ObservationNoise:
    """
    Inject noise into observation vectors, simulating sensor uncertainties.
    """

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def apply(self, observation: np.ndarray) -> np.ndarray:
        """
        :param observation: shape (obs_dim,)
        :return: noisy observation
        """
        noisy_obs = observation + np.random.randn(*observation.shape) * self.sigma
        return noisy_obs


class ParameterDisturbance:
    """
    Randomize environment or vehicle parameters each reset or periodically,
    e.g., friction, mass, wind, etc.
    This is a placeholder for advanced domain randomization.
    """

    def __init__(self, friction_range=(0.8, 1.2), mass_range=(0.9, 1.1)):
        """
        :param friction_range: possible range for friction multiplier
        :param mass_range: possible range for mass multiplier
        """
        self.friction_range = friction_range
        self.mass_range = mass_range

    def sample_friction(self):
        return np.random.uniform(self.friction_range[0], self.friction_range[1])

    def sample_mass(self):
        return np.random.uniform(self.mass_range[0], self.mass_range[1])

    def apply_to_vehicle(self, vehicle):
        """
        Example: pseudo-code for adjusting friction or mass on a vehicle.
        Requires custom logic for Carla or your simulator, if available.
        """
        friction_factor = self.sample_friction()
        mass_factor = self.sample_mass()
        # For demonstration. In reality you'd have to
        # retrieve vehicle physics/control parameters from Carla and override them.
        # E.g. in Carla, you might do something with `vehicle.get_physics_control()`
        # to set new mass or friction. For now, let's just log or pass.
        print(
            f"[ParameterDisturbance] Applying friction={friction_factor:.3f}, mass={mass_factor:.3f} to vehicle {vehicle.id}.")
        # (You can also do advanced domain randomization with weather, sensor noise, etc.)
