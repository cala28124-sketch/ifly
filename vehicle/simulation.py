import matplotlib.pyplot as plt
from flyvis import NetworkView
from flyvis.analysis.moving_bar_responses import plot_angular_tuning

# 1. Load a single network using NetworkView
print("Loading network...")
network = NetworkView("flow/0000/000")

# 2. Run the simulation 
print("Running moving edge simulation...")
stims_and_resps = network.moving_edge_responses()

# 3. Generate the plot
print("Generating plot...")
fig, ax = plt.subplots()
plot_angular_tuning(stims_and_resps, ax=ax)

# 4. Save the plot to a file instead of displaying it inline
plt.savefig("angular_tuning_output.png")
print("Saved plot to angular_tuning_output.png")