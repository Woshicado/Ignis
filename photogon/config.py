import numpy as np

standard = {
    "name": "standard",
    "cam_dist": 300,
    "theta_steps": 50,          # equidistant
    "theta_max": np.pi / 2,     # inclusive
    "phi_max": 2.0 * np.pi,     # exclusive
    "phi_steps": [(0.05, 20),    # Percentage of 1 corresponds to `pi / 2`
                  (0.1, 40),     # Order by increasing percentage
                  (0.3, 80),     # format: (percentage of `pi / 2`, num_samples)
                  (0.6, 160),     # Higher percentage -> closer to x-y-plane
                  (0.8, 320)],
    "spp": 32,
    "sun_theta_max": np.pi / 2, # inclusive
    "sun_theta_steps": 10,
    "sun_phi_max": 2.0 * np.pi, # exclusive
    "sun_phi_steps": [(0.05, 1),
                      (0.1, 1),
                      (0.3, 3),
                      (0.6, 3),
                      (0.8, 3)],
    "sun_irradiance": 1.0
}
