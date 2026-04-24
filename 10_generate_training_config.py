# In[]
"""
10_generate_training_config.py

This script generates a JSON configuration file for training the machine learning model.
The generated config includes parameters for data loading, peak detection, training
hyperparameters, and hardware device settings.

Output:
    icequake_train_config.json
"""

import json
import os

def main():
    # Define the training configuration dictionary
    # Values are aligned with our newly curated 200Hz, 2001-sample dataset.
    config = {
        "peak_detection": {
            "sampling_rate": 200,
            "height": 0.5,
            "distance": 100
        },
        "data": {
            "dataset_name": "final_curated_seisbench_data",
            "sampling_rate": 200,
            "window_len": 1001,      # Our traces are exactly 1001 samples long (5 seconds)
            "samples_before": 1000,  # Arrival is centered at sample 1000
            "windowlen_large": 2000,
            "sample_fraction": 1.0
        },
        "training": {
            "batch_size": 4,
            "num_workers": 2,
            "learning_rate": 0.01,
            "epochs": 50,
            "patience": 5,
            "loss_weights": [
                0.01,  # Noise
                0.4,   # P-wave
                0.59   # S-wave
            ],
            "optimization": {
                "mixed_precision": True,
                "gradient_accumulation_steps": 1,
                "pin_memory": True,
                "prefetch_factor": 2,
                "persistent_workers": True
            }
        },
        "device": {
            "use_cuda": True,
            "device_id": 0
        }
    }

    output_file = "icequake_train_config.json"
    
    # Write the dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Training configuration successfully generated: {output_file}")

if __name__ == "__main__":
    main()

# %%
