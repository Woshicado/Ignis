from datetime import datetime
import numpy as np

FILE_HEADER = "#Synthetic BRDF data created in ignis\n" \
             f"#created at { datetime.now().strftime('%Y-%m-%d %H:%M:%S') }\n" \
              "#format: theta\tphi\tBRDF\n"

standard = {
    "name": "standard",
    "cam_dist": 500,        # equidistant
    "theta_steps": 50,      # equidistant
    "theta_max": np.pi,     # inclusive
    "phi_max": 2.0 * np.pi,   # exclusive
    "phi_steps": [(0.05, 8),    # Percentage of 1 corresponds to `pi / 2`
                  (0.1, 16),    # Order by increasing percentage
                  (0.3, 32),    # format: (percentage of `pi / 2`, num_samples)
                  (0.6, 64),
                  (0.8, 128)],
    "spp": 32
}
