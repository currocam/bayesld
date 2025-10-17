import numpy as np
import pickle

# Load the pickle file
data = np.load("analytical_data.npz", allow_pickle=True)

# Extract mean values for each scenario
output = {}

for scenario_name, scenario_data in data.items():
    scenario_data = scenario_data.item()
    # Extract simulation means
    sims_mean = np.asarray([x["mean"] for x in scenario_data["sims"]]).mean(axis=0)

    # Extract prediction means
    predictions_mean = scenario_data["idata"].LD.mean(axis=1).values

    # Extract left and right from first simulation (same for all)
    left = scenario_data["sims"][0]["left"]
    right = scenario_data["sims"][0]["right"]

    # Store in output dictionary
    output[f"{scenario_name}_sims_mean"] = sims_mean
    output[f"{scenario_name}_predictions_mean"] = predictions_mean
    output[f"{scenario_name}_left"] = left
    output[f"{scenario_name}_right"] = right

# Save as NPZ file
np.savez("analytical_data_means.npz", **output)
print("Saved means to analytical_data_means.npz")
