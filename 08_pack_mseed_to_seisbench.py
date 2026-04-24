# In[]

#!/usr/bin/env python3
"""
Pack trimmed ML datasets into SeisBench HDF5 and CSV format.
"""

import numpy as np
import pandas as pd
import json
import glob
import os
from pathlib import Path
from obspy import read
from datetime import datetime, timedelta

import seisbench.data as sbd

input_dir = 'trimmed_and_consistent_mseed'
output_dir = 'final_curated_seisbench_data'

def assign_dataset_splits(n_events, split_ratios=None, random_seed=42):
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'dev': 0.15, 'test': 0.15}
    
    np.random.seed(random_seed)
    
    n_train = int(np.floor(n_events * split_ratios['train']))
    n_dev = int(np.floor(n_events * split_ratios['dev']))
    n_test = n_events - n_train - n_dev
    
    splits = ['train'] * n_train + ['dev'] * n_dev + ['test'] * n_test
    np.random.shuffle(splits)
    
    return splits

def map_channel_to_component(chan):
    if chan.endswith('Z'): return 'Z'
    if chan.endswith('N') or chan.endswith('2'): return 'N'
    if chan.endswith('E') or chan.endswith('1'): return 'E'
    return None

def main():
    print("=" * 70)
    print("Packing to SeisBench Format")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    waveforms_path = os.path.join(output_dir, 'waveforms.hdf5')
    
    json_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    print(f"Found {len(json_files)} metadata JSON files.")
    
    if len(json_files) == 0:
        print("No JSON files found!")
        return
        
    splits = assign_dataset_splits(len(json_files))
    
    traces_written = 0
    traces_failed = 0
    
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        writer.data_format = {
            'dimension_order': 'CW',  # Channel, Width (samples)
            'component_order': 'ZNE',  # Vertical, North, East
            'measurement': 'velocity',
            'unit': 'counts',
            'instrument_response': 'not restituted',
        }
        
        for i, json_path in enumerate(json_files):
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                
            event_id = metadata['event_id']
            station = metadata['station']
            network = metadata['network']
            start_time = metadata['start_time']
            sample_rate = metadata['sample_rate']
            
            # Load waveforms
            data_dict = {}
            valid = True
            
            # The JSON files dictionary contains the mseed filenames
            for mseed_file, channel in zip(metadata['files']['mseed'], metadata['channels']):
                full_path = os.path.join(input_dir, mseed_file)
                if not os.path.exists(full_path):
                    valid = False
                    break
                    
                st = read(full_path)
                comp = map_channel_to_component(channel)
                if comp:
                    data_dict[comp] = st[0].data
            
            if not valid or 'Z' not in data_dict or 'N' not in data_dict or 'E' not in data_dict:
                traces_failed += 1
                continue
                
            # Stack into ZNE order
            data_3c = np.vstack([data_dict['Z'], data_dict['N'], data_dict['E']])
            
            # Build trace metadata for SeisBench
            trace_metadata = {
                'trace_name_original': f"{network}.{station}.{start_time}",
                'station_network_code': network,
                'station_code': station,
                'trace_channel': metadata['channels'][0][:2], # e.g. 'HH'
                'trace_sampling_rate_hz': sample_rate,
                'trace_npts': data_3c.shape[1],
                'trace_start_time': start_time,
                'source_id': event_id,
                'split': splits[i]
            }
            
            if 'p_arrival_sample' in metadata:
                trace_metadata['trace_p_arrival_sample'] = metadata['p_arrival_sample']
                trace_metadata['trace_p_status'] = 'manual'
                trace_metadata['trace_p_weight'] = 1.0
                
            if 's_arrival_sample' in metadata:
                trace_metadata['trace_s_arrival_sample'] = metadata['s_arrival_sample']
                trace_metadata['trace_s_status'] = 'manual'
                trace_metadata['trace_s_weight'] = 1.0
                
            # Write trace
            try:
                writer.add_trace(trace_metadata, data_3c)
                traces_written += 1
            except Exception as e:
                print(f"Failed writing {station}: {e}")
                traces_failed += 1
                
            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{len(json_files)}")

    print(f"\n✓ Completed packing {traces_written} traces ({traces_failed} failed).")
    
    # Verify
    print("\nVerifying dataset integrity...")
    try:
        dataset = sbd.WaveformDataset(output_dir)
        print(f"✓ Dataset loaded successfully with SeisBench")
        print(f"✓ Number of traces: {len(dataset)}")
        print(f"✓ Metadata loaded: {len(dataset.metadata)} rows")
        
        df = dataset.metadata
        print(f"\nDataset splits:")
        for split_name in ['train', 'dev', 'test']:
            count = (df['split'] == split_name).sum()
            print(f"  {split_name}: {count} traces")
            
    except Exception as e:
        print(f"✗ Verification failed: {e}")

if __name__ == "__main__":
    main()

# %%
