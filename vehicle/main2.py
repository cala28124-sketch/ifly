import cv2
import torch
import numpy as np
import pandas as pd
from flyvis import NetworkView
from flyvis.datasets.rendering import BoxEye

# 1. Initialize the network and eye renderer
print("Loading network view...")
network = NetworkView("flow/0000/000")
receptors = BoxEye()

# 2. Extract and recover the raw PyTorch model
print("Recovering core PyTorch model...")
checkpoint_container = network.network()
core_model = checkpoint_container.recover()

# 3. Locate the looming detectors via expanding motion
print("Locating global motion detectors...")

raw_types = network.connectome.nodes["type"][:]
cell_types = [
    val.decode('utf-8') if isinstance(val, bytes) else str(val) 
    for val in raw_types
]

# Grab every T4 and T5 cell (covering all 4 cardinal directions)
target_indices = [
    i for i, ctype in enumerate(cell_types) 
    if ctype.startswith("T4") or ctype.startswith("T5")
]
print(f"Monitoring {len(target_indices)} cells for looming expansion...")

# 4. Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

BUFFER_SIZE = 10
frame_buffer = []

print("Starting webcam feed. Press 'q' to quit.")

# 5. Main Simulation Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Pre-process the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0

    
    
    frame_buffer.append(normalized)
    
    if len(frame_buffer) > BUFFER_SIZE:
        frame_buffer.pop(0)
        
    # Run the simulation once the buffer is full
    if len(frame_buffer) == BUFFER_SIZE:
        video_tensor = torch.tensor(np.array([frame_buffer]), dtype=torch.float32)
        hex_stimulus = receptors(video_tensor)
        
        # Pad the visual input to match the 45,669 total neurons
        total_neurons = 45669
        num_receptors = hex_stimulus.shape[-1]
        padded_stimulus = torch.nn.functional.pad(hex_stimulus, (0, total_neurons - num_receptors))
        
        with torch.no_grad():
            responses = core_model(padded_stimulus, dt=0.01)
            
        # Extract the global motion data
        brain_state = responses[0] 
        target_activity = brain_state[:, 0, target_indices] 
        
        # Calculate the mean activation of the T4/T5 cells for each frame
        mean_activity_over_time = target_activity.mean(dim=1) 
        
        # The model starts at 0 volts on frame 1 and drops to -1.22.
        # Isolate the last 4 frames where the voltage has settled to analyze real motion.
        settled_activity = mean_activity_over_time[-4:]
        
        # Calculate the mathematical fluctuation (max - min) in that settled window
        activation_spike = settled_activity.max().item() - settled_activity.min().item()
        
        # Trigger threshold to catch sudden motion spikes
        TRIGGER_THRESHOLD = 0.4001
        
        if activation_spike < TRIGGER_THRESHOLD:
            print(f"Collision! (Spike: {activation_spike:.4f})")
        else:
            print(f"Stable. (Spike: {activation_spike:.4f})")
        
    # Display the live feed
    cv2.imshow("Webcam Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()