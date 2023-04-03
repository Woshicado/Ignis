import numpy as np

standard = {
    "name": "standard",
    "cam_dist": 500,
    "theta_steps": 20,          # equidistant
    "theta_max": np.pi,         # inclusive
    "phi_max": 2.0 * np.pi,     # exclusive
    "phi_steps": [(0.05, 1),    # Percentage of 1 corresponds to `pi / 2`
                  (0.1, 2),     # Order by increasing percentage
                  (0.3, 4),     # format: (percentage of `pi / 2`, num_samples)
                  (0.6, 8),     # Higher percentage -> closer to x-y-plane
                  (0.8, 12)],
    "spp": 32,
    "sun_theta_max": np.pi,     # inclusive
    "sun_theta_steps": 10,
    "sun_phi_max": 2.0 * np.pi, # exclusive
    "sun_phi_steps": [(0.05, 1),
                      (0.1, 1),
                      (0.3, 1),
                      (0.6, 1),
                      (0.8, 1)],
    "sun_irradiance": 1.0
}
