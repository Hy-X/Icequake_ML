#In[]
"""
09_visualize_curated_final_dataset.py

This script verifies and visualizes the final SeisBench-compatible dataset.
It loads the dataset from 'final_curated_seisbench_data', prints out metadata
statistics, and generates a plot of a sample waveform showing the 3 components
(Z, N, E) and the phase arrival labels (P and/or S).

Usage:
    python 09_visualize_curated_final_dataset.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seisbench.data as sbd

def main():
    # Path to the final dataset directory
    dataset_path = 'final_curated_seisbench_data'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return
    
    print(f"Loading SeisBench dataset from '{dataset_path}'...")
    
    # Load the dataset
    dataset = sbd.WaveformDataset(dataset_path)
    
    print("\n" + "="*50)
    print("DATASET METADATA SUMMARY")
    print("="*50)
    
    # Show metadata columns
    print("Available columns:")
    for col in dataset.metadata.columns:
        print(f"  - {col}")
        
    # Display the first few rows of metadata
    print("\nFirst 15 rows of metadata:")
    with pd.option_context('display.max_columns', None, 'display.width', 1000):
        print(dataset.metadata.head(15))
    
    # Dataset Summary statistics
    print("\nDataset Summary:")
    print(f"  Stations: {dataset.metadata['station_code'].nunique()}")
    print(f"  Networks: {dataset.metadata['station_network_code'].nunique()}")
    
    if 'trace_p_arrival_sample' in dataset.metadata.columns:
        print(f"  Traces with P-picks: {dataset.metadata['trace_p_arrival_sample'].notna().sum()}")
    else:
        print("  Traces with P-picks: 0")
        
    if 'trace_s_arrival_sample' in dataset.metadata.columns:
        print(f"  Traces with S-picks: {dataset.metadata['trace_s_arrival_sample'].notna().sum()}")
    else:
        print("  Traces with S-picks: 0")
        
    if 'trace_snr_db' in dataset.metadata.columns and not dataset.metadata['trace_snr_db'].isna().all():
        print(f"\nSNR Statistics:")
        print(f"  Mean: {dataset.metadata['trace_snr_db'].mean():.2f} dB")
        print(f"  Range: {dataset.metadata['trace_snr_db'].min():.2f} - {dataset.metadata['trace_snr_db'].max():.2f} dB")
    
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    # Visualize traces
    num_examples = min(3, len(dataset))
    
    # Color scheme requested
    colors = ['#344e41', '#588157', '#a3b18a']
    components = ['Z (Vertical)', 'N (North)', 'E (East)']
    
    for trace_idx in range(num_examples):
        waveforms = dataset.get_waveforms(trace_idx)
        metadata = dataset.metadata.iloc[trace_idx]
        
        sr = metadata.get('trace_sampling_rate_hz', 100.0)
        npts = metadata.get('trace_npts', waveforms.shape[1])
        
        # Create time axis
        time_axis = np.arange(npts) / sr
        
        event_id = metadata.get('source_id', 'Unknown_Event')
        station_code = metadata.get('station_code', 'Unknown_Station')
        network_code = metadata.get('station_network_code', 'Unknown_Network')
        start_time = metadata.get('trace_start_time', 'Unknown_Time')
        
        p_sample = metadata.get('trace_p_arrival_sample', np.nan)
        s_sample = metadata.get('trace_s_arrival_sample', np.nan)
        
        p_time = p_sample / sr if pd.notna(p_sample) else np.nan
        s_time = s_sample / sr if pd.notna(s_sample) else np.nan
        sp_time = s_time - p_time if (pd.notna(p_time) and pd.notna(s_time)) else np.nan
        
        # Print Trace Information
        print(f"\n{'='*60}")
        print("TRACE INFORMATION")
        print(f"{'='*60}")
        print(f"Station:        {network_code}.{station_code}")
        print(f"Event ID:       {event_id}")
        print(f"Date/Time:      {start_time}")
        print(f"Sampling Rate:  {sr} Hz")
        print("\nPhase Arrivals:")
        if pd.notna(p_sample):
            print(f"  P-wave:       sample {int(p_sample)} ({p_time:.2f} s)")
        else:
            print("  P-wave:       None")
            
        if pd.notna(s_sample):
            print(f"  S-wave:       sample {int(s_sample)} ({s_time:.2f} s)")
        else:
            print("  S-wave:       None")
            
        if pd.notna(sp_time):
            print(f"  S-P time:     {sp_time:.2f} s")
        print(f"{'='*60}")
        
        # Create visualization plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        fig.suptitle(f"Event: {event_id} | Station: {network_code}.{station_code} | Start: {start_time} | SR: {sr} Hz", fontsize=14)
        
        for i, (ax, comp, color) in enumerate(zip(axes, components, colors)):
            ax.plot(time_axis, waveforms[i], color=color, linewidth=1.5, label=comp)
            
            # Add P pick if available
            if pd.notna(p_time):
                ax.axvline(x=p_time, color='red', linestyle='--', label='P Arrival' if i == 0 else "_nolegend_")
                
            # Add S pick if available
            if pd.notna(s_time):
                ax.axvline(x=s_time, color='blue', linestyle='--', label='S Arrival' if i == 0 else "_nolegend_")
                
            ax.set_ylabel(f"{comp}", fontsize=10)
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.6)
            
        axes[-1].set_xlabel("Time (seconds)")
        
        output_img = f"final_dataset_sample_{trace_idx}.png"
        plt.savefig(output_img, dpi=300)
        plt.close(fig)
        print(f"Saved sample visualization to {output_img}")
    
    print("\nVisualization complete. The dataset is ready for machine learning tasks!")

if __name__ == "__main__":
    main()

# %%
