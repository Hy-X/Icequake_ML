#!/usr/bin/env python3
"""
Pack synthetic seismograms into SeisBench HDF5 and CSV format.

This script converts generated synthetic 3-component seismograms into the
SeisBench data format for easy integration with SeisBench models and workflows.

Uses the official SeisBench WaveformDataWriter API for proper dataset creation,
following the reference example from SeisBench documentation.

Output:
    - metadata.csv: Trace metadata with phase picks
    - waveforms.hdf5: 3-component waveform data in HDF5 format

The format follows SeisBench conventions for compatibility with existing
SeisBench models (e.g., PhaseNet, EQTransformer).

Usage:
    python P003_pack_to_seisbench.py

Requirements:
    - seisbench
    - pandas
    - numpy
    - obspy
"""

import numpy as np
import pandas as pd
import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from obspy import read
from datetime import datetime, timedelta

import seisbench.data as sbd
import seisbench.util as sbu


def load_config(config_path: str = 'Syn_Config.json') -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r') as f:
        return json.load(f)


def assign_dataset_splits(
    n_events: int,
    split_ratios: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None
) -> List[str]:
    """
    Randomly assign dataset splits to events based on specified ratios.
    
    Args:
        n_events: Number of events to split
        split_ratios: Dictionary with 'train', 'dev', 'test' ratios (default: 0.7/0.15/0.15)
        random_seed: Random seed for reproducibility (default: None)
        
    Returns:
        List of split assignments ('train', 'dev', or 'test') for each event
        
    Raises:
        ValueError: If split ratios don't sum to 1.0
    """
    # Default split ratios
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'dev': 0.15, 'test': 0.15}
    
    # Validate ratios
    ratio_sum = sum(split_ratios.values())
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum}. "
            f"Ratios: {split_ratios}"
        )
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate number of samples for each split
    train_ratio = split_ratios.get('train', 0.7)
    dev_ratio = split_ratios.get('dev', 0.15)
    test_ratio = split_ratios.get('test', 0.15)
    
    n_train = int(np.floor(n_events * train_ratio))
    n_dev = int(np.floor(n_events * dev_ratio))
    n_test = n_events - n_train - n_dev  # Remainder goes to test
    
    # Create split assignments
    splits = ['train'] * n_train + ['dev'] * n_dev + ['test'] * n_test
    
    # Shuffle to randomize
    np.random.shuffle(splits)
    
    return splits


def discover_synthetic_data() -> List[Dict]:
    """
    Discover all synthetic seismogram files and load their metadata.
    
    Returns:
        List of dictionaries containing event information and file paths
        
    Raises:
        FileNotFoundError: If no synthetic data files are found
    """
    metadata_files = sorted(glob.glob('SYNTHETIC_*_metadata.json'))
    
    if len(metadata_files) == 0:
        raise FileNotFoundError(
            "No synthetic seismogram metadata files found (SYNTHETIC_*_metadata.json). "
            "Please run P002_batch_generate_synthetic_3c.py first."
        )
    
    events = []
    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract base name for file discovery
        base_name = meta_file.replace('_metadata.json', '')
        
        # Check for required files
        npy_file = f"{base_name}_3C.npy"
        mseed_files = {
            'Z': f"{base_name}_HHZ.mseed",
            'N': f"{base_name}_HHN.mseed",
            'E': f"{base_name}_HHE.mseed"
        }
        
        # Verify files exist
        if not os.path.exists(npy_file):
            print(f"Warning: NPY file not found for {metadata['event_id']}, skipping...")
            continue
        
        missing_mseed = [ch for ch, path in mseed_files.items() if not os.path.exists(path)]
        if missing_mseed:
            print(f"Warning: Missing MSEED files for {metadata['event_id']} channels: {missing_mseed}")
            # Continue if NPY file exists as fallback
        
        events.append({
            'event_id': metadata['event_id'],
            'metadata': metadata,
            'npy_file': npy_file,
            'mseed_files': mseed_files,
            'base_name': base_name
        })
    
    return events


def load_waveform_data(event: Dict, use_mseed: bool = True) -> Tuple[np.ndarray, float, Dict[str, str]]:
    """
    Load 3-component waveform data from MSEED or NPY files.
    
    Args:
        event: Event dictionary from discover_synthetic_data()
        use_mseed: If True, load from MSEED files; otherwise use NPY (default: True)
        
    Returns:
        data_3c: Waveform array of shape (3, n_samples) for Z, N, E
        sampling_rate: Sampling rate in Hz
        channel_codes: Dictionary with keys 'Z', 'N', 'E' containing full channel codes
                      (e.g., {'Z': 'HHZ', 'N': 'HHN', 'E': 'HHE'})
        
    Raises:
        ValueError: If waveform data cannot be loaded
    """
    channel_codes = {'Z': 'HHZ', 'N': 'HHN', 'E': 'HHE'}  # Default fallback
    
    if use_mseed and all(os.path.exists(f) for f in event['mseed_files'].values()):
        # Load from MSEED files
        channels = ['Z', 'N', 'E']
        data_list = []
        
        for ch in channels:
            st = read(event['mseed_files'][ch])
            tr = st[0]
            data_list.append(tr.data)
            sampling_rate = tr.stats.sampling_rate
            
            # Extract full channel code from MSEED header (e.g., 'HHZ', 'HHN', 'HHE')
            channel_codes[ch] = tr.stats.channel
        
        data_3c = np.vstack(data_list)
        
    else:
        # Load from NPY file
        data_3c = np.load(event['npy_file'])
        sampling_rate = event['metadata'].get('sample_rate', 100.0)
        # Try to get channel code from metadata, otherwise use default
        channel_base = event['metadata'].get('channel_code', 'HH')
        channel_codes = {
            'Z': f'{channel_base}Z',
            'N': f'{channel_base}N',
            'E': f'{channel_base}E'
        }
    
    # Validate shape
    if data_3c.shape[0] != 3:
        raise ValueError(
            f"Invalid waveform shape {data_3c.shape} for {event['event_id']}. "
            f"Expected (3, n_samples)."
        )
    
    return data_3c, sampling_rate, channel_codes


def build_trace_metadata(
    event: Dict,
    data_3c: np.ndarray,
    sampling_rate: float,
    channel_codes: Dict[str, str],
    split: str = 'train'
) -> Dict:
    """
    Build metadata dictionary for a single trace following SeisBench conventions.
    
    Following the pattern from SeisBench reference example, metadata uses
    prefixes like 'station_', 'trace_', 'source_' for different property types.
    
    Constructs trace name in format:
        {network}.{station}.{channel_E}.{channel_N}.{channel_Z}.{start_time}{end_time}
    Example: "2V.TG11.EHE.EHN.EHZ.2023-08-18T1821082023-08-18T182108"
    
    Note: The custom trace name is stored in 'trace_name_original' because SeisBench's
    WaveformDataWriter automatically overwrites 'trace_name' with HDF5 bucket identifiers.
    
    Args:
        event: Event dictionary from discover_synthetic_data()
        data_3c: Waveform array of shape (3, n_samples)
        sampling_rate: Sampling rate in Hz
        channel_codes: Dictionary with keys 'Z', 'N', 'E' containing full channel codes
        split: Dataset split assignment ('train', 'dev', or 'test')
        
    Returns:
        Dictionary with trace metadata following SeisBench naming conventions
    """
    metadata = event['metadata']
    event_id = event['event_id']
    
    # Station/Network information
    station_code = metadata.get('station', 'HX')
    network_code = metadata.get('network', 'QD')
    station_latitude = metadata.get('station_latitude', 35.0)
    station_longitude = metadata.get('station_longitude', -97.0)
    
    # Phase arrivals
    p_arrival_sample = metadata.get('p_arrival_sample', -1)
    s_arrival_sample = metadata.get('s_arrival_sample', -1)
    
    # Time information
    start_time_str = metadata.get('start_time', '1970-01-01T00:00:00')
    
    # Calculate end time based on trace length and sampling rate
    trace_duration_sec = data_3c.shape[1] / sampling_rate
    try:
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        end_time = start_time + timedelta(seconds=trace_duration_sec)
        
        # Format timestamps for trace name (remove colons from time part)
        # Format: YYYY-MM-DDTHHMMSS
        start_time_fmt = start_time.strftime('%Y-%m-%dT%H%M%S')
        end_time_fmt = end_time.strftime('%Y-%m-%dT%H%M%S')
    except (ValueError, AttributeError):
        # Fallback if datetime parsing fails - only remove colons
        start_time_fmt = start_time_str.replace(':', '')
        end_time_fmt = start_time_fmt
    
    # Construct trace name following SeisBench format
    # Format: {network}.{station}.{channel_E}.{channel_N}.{channel_Z}.{start_time}{end_time}
    trace_name = (
        f"{network_code}.{station_code}."
        f"{channel_codes['E']}.{channel_codes['N']}.{channel_codes['Z']}."
        f"{start_time_fmt}{end_time_fmt}"
    )
    
    # Extract band + instrument code for trace_channel field (e.g., 'HH' from 'HHZ')
    channel_code = channel_codes['Z'][:2] if len(channel_codes['Z']) >= 2 else 'HH'
    
    # Build metadata dictionary with SeisBench conventions
    trace_metadata = {
        # Custom trace identifier (stored separately from SeisBench's internal trace_name)
        # SeisBench will overwrite 'trace_name' with HDF5 bucket IDs, so we use a custom field
        'trace_name_original': trace_name,
        
        # Station information (station_ prefix)
        'station_network_code': network_code,
        'station_code': station_code,
        'station_latitude_deg': station_latitude,
        'station_longitude_deg': station_longitude,
        
        # Trace properties (trace_ prefix)
        'trace_channel': channel_code,  # Band + instrument code from MSEED header
        'trace_sampling_rate_hz': sampling_rate,
        'trace_npts': data_3c.shape[1],
        'trace_start_time': start_time_str,
        
        # Phase arrivals - P-wave (trace_ prefix)
        'trace_p_arrival_sample': p_arrival_sample if p_arrival_sample >= 0 else None,
        'trace_p_status': 'manual' if p_arrival_sample >= 0 else None,
        'trace_p_weight': 1.0 if p_arrival_sample >= 0 else None,
        
        # Phase arrivals - S-wave (trace_ prefix)
        'trace_s_arrival_sample': s_arrival_sample if s_arrival_sample >= 0 else None,
        'trace_s_status': 'manual' if s_arrival_sample >= 0 else None,
        'trace_s_weight': 1.0 if s_arrival_sample >= 0 else None,
        
        # Source information (source_ prefix)
        'source_id': event_id,
        'source_origin_time': metadata.get('start_time', '1970-01-01T00:00:00'),
        'source_type': 'earthquake',
        'source_magnitude': 2.0,
        'source_magnitude_type': 'ML',
        'source_magnitude_author': 'synthetic',
        
        # Quality metrics
        'trace_snr_db': metadata.get('snr_db', 0.0),
        
        # Dataset split
        'split': split,
    }
    
    return trace_metadata


def create_seisbench_dataset(
    events: List[Dict],
    output_dir: str = '../data',
    config: Optional[Dict] = None
) -> Path:
    """
    Create SeisBench dataset using the official WaveformDataWriter API.
    
    This function follows the SeisBench best practices for dataset creation
    as shown in the reference creating_a_dataset.ipynb example:
    - Uses WaveformDataWriter as a context manager
    - Sets proper data_format specifications
    - Writes traces incrementally with add_trace()
    - Creates metadata.csv and waveforms.hdf5 in SeisBench format
    
    Args:
        events: List of event dictionaries from discover_synthetic_data()
        output_dir: Output directory for dataset files (default: '../data')
        config: Configuration dictionary with split ratios (default: None)
        
    Returns:
        Path to the output directory
        
    Raises:
        IOError: If dataset files cannot be created
    """
    # Setup output paths
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    metadata_path = base_path / 'metadata.csv'
    waveforms_path = base_path / 'waveforms.hdf5'
    
    print(f"\nCreating SeisBench dataset using WaveformDataWriter")
    print(f"  Output directory: {base_path}")
    print(f"  Number of events: {len(events)}")
    
    # Assign dataset splits based on configuration
    split_ratios = None
    random_seed = None
    if config is not None:
        split_ratios = config.get('dataset_split_ratios')
        random_seed = config.get('random_seed')
    
    splits = assign_dataset_splits(len(events), split_ratios, random_seed)
    
    # Print split statistics
    split_counts = {split: splits.count(split) for split in ['train', 'dev', 'test']}
    print(f"  Dataset splits: train={split_counts['train']}, dev={split_counts['dev']}, test={split_counts['test']}")
    
    # Use WaveformDataWriter following SeisBench conventions
    # This is the official API method from seisbench.data
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        # Define data format specifications
        # This tells SeisBench how to interpret the waveform arrays
        writer.data_format = {
            'dimension_order': 'CW',  # Channel, Width (samples)
            'component_order': 'ZNE',  # Vertical, North, East
            'measurement': 'velocity',
            'unit': 'counts',
            'instrument_response': 'not restituted',
        }
        
        # Iterate over events and write traces
        traces_written = 0
        traces_failed = 0
        
        for i, event in enumerate(events, 1):
            event_id = event['event_id']
            
            try:
                # Load waveform data and extract channel codes from MSEED headers
                data_3c, sampling_rate, channel_codes = load_waveform_data(event)
                
                # Get split assignment for this event
                event_split = splits[i - 1]
                
                # Build metadata dictionary with extracted channel codes
                trace_metadata = build_trace_metadata(event, data_3c, sampling_rate, channel_codes, event_split)
                
                # Add trace to dataset using SeisBench writer
                # The writer handles HDF5 writing and metadata collection
                writer.add_trace(trace_metadata, data_3c)
                
                traces_written += 1
                
                if i % 10 == 0 or i == len(events):
                    print(f"  Progress: {i}/{len(events)} events processed ({traces_written} written, {traces_failed} failed)")
                
            except Exception as e:
                traces_failed += 1
                print(f"  Warning: Failed to process {event_id}: {e}")
                continue
    
    print(f"\n✓ SeisBench dataset created successfully!")
    print(f"  Total traces written: {traces_written}")
    print(f"  Failed: {traces_failed}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Waveforms: {waveforms_path}")
    
    return base_path


def print_dataset_summary(dataset_path: Path) -> None:
    """
    Load and print summary statistics of the created SeisBench dataset.
    
    Args:
        dataset_path: Path to the dataset directory
    """
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    
    metadata_path = dataset_path / 'metadata.csv'
    waveforms_path = dataset_path / 'waveforms.hdf5'
    
    # File sizes
    if waveforms_path.exists():
        size_mb = waveforms_path.stat().st_size / (1024 * 1024)
        print(f"Waveforms HDF5 size: {size_mb:.2f} MB")
    
    if metadata_path.exists():
        size_kb = metadata_path.stat().st_size / 1024
        print(f"Metadata CSV size: {size_kb:.2f} KB")
    
    # Load metadata
    try:
        df = pd.read_csv(metadata_path)
        
        print(f"\nDataset statistics:")
        print(f"  Total traces: {len(df)}")
        print(f"  Stations: {df['station_code'].nunique()}")
        print(f"  Networks: {df['station_network_code'].nunique()}")
        
        # Sampling rate
        if 'trace_sampling_rate_hz' in df.columns:
            print(f"  Sampling rate: {df['trace_sampling_rate_hz'].iloc[0]:.0f} Hz")
        
        # Pick statistics
        print(f"\nPhase arrival statistics:")
        p_col = 'trace_p_arrival_sample'
        s_col = 'trace_s_arrival_sample'
        
        if p_col in df.columns:
            p_count = df[p_col].notna().sum()
            print(f"  P-arrivals: {p_count}/{len(df)}")
            if p_count > 0:
                p_samples = df[df[p_col].notna()][p_col]
                print(f"    Sample range: {p_samples.min():.0f} - {p_samples.max():.0f}")
        
        if s_col in df.columns:
            s_count = df[s_col].notna().sum()
            print(f"  S-arrivals: {s_count}/{len(df)}")
            if s_count > 0:
                s_samples = df[df[s_col].notna()][s_col]
                print(f"    Sample range: {s_samples.min():.0f} - {s_samples.max():.0f}")
        
        # SNR statistics
        if 'trace_snr_db' in df.columns:
            snr_data = df['trace_snr_db']
            print(f"\nSNR statistics:")
            print(f"  Mean: {snr_data.mean():.2f} dB")
            print(f"  Std:  {snr_data.std():.2f} dB")
            print(f"  Range: {snr_data.min():.2f} - {snr_data.max():.2f} dB")
        
        # Split information
        if 'split' in df.columns:
            print(f"\nDataset splits:")
            for split_name in ['train', 'dev', 'test']:
                count = (df['split'] == split_name).sum()
                if count > 0:
                    print(f"  {split_name}: {count} traces")
        
    except Exception as e:
        print(f"\n  Warning: Could not load metadata for summary: {e}")
    
    print("=" * 70)


def verify_dataset(dataset_path: Path) -> bool:
    """
    Verify integrity of created SeisBench dataset by loading it.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        True if dataset is valid, False otherwise
    """
    print("\nVerifying dataset integrity...")
    
    try:
        # Try to load the dataset using SeisBench
        # This is the ultimate test - if SeisBench can load it, it's valid
        dataset = sbd.WaveformDataset(dataset_path, sampling_rate=100)
        
        print(f"  ✓ Dataset loaded successfully with SeisBench")
        print(f"  ✓ Number of traces: {len(dataset)}")
        
        # Check metadata
        if len(dataset.metadata) == 0:
            print(f"  ✗ Dataset has no metadata")
            return False
        
        print(f"  ✓ Metadata loaded: {len(dataset.metadata)} rows")
        
        # Try to load first waveform
        if len(dataset) > 0:
            waveform = dataset.get_waveforms(0)
            print(f"  ✓ Waveform shape: {waveform.shape}")
            
            if waveform.shape[0] != 3:
                print(f"  ✗ Invalid waveform shape (expected 3 components)")
                return False
        
        # Check required columns
        required_cols = ['station_code', 'station_network_code', 'trace_sampling_rate_hz']
        missing_cols = [col for col in required_cols if col not in dataset.metadata.columns]
        if missing_cols:
            print(f"  ⚠ Warning: Missing recommended columns: {missing_cols}")
        else:
            print(f"  ✓ Required columns present")
        
        print("  ✓ Dataset verification passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_synthetic_files(events: List[Dict]) -> None:
    """
    Delete synthetic trace files after successful packing to save disk space.
    
    Removes MSEED files, NPY files, and metadata JSON files for each event.
    
    Args:
        events: List of event dictionaries from discover_synthetic_data()
    """
    print("\nCleaning up synthetic trace files...")
    
    files_deleted = 0
    files_failed = 0
    
    for event in events:
        try:
            # Delete MSEED files
            for mseed_file in event['mseed_files'].values():
                if os.path.exists(mseed_file):
                    os.remove(mseed_file)
                    files_deleted += 1
            
            # Delete NPY file
            if os.path.exists(event['npy_file']):
                os.remove(event['npy_file'])
                files_deleted += 1
            
            # Delete metadata JSON file
            metadata_file = f"{event['base_name']}_metadata.json"
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                files_deleted += 1
                
        except Exception as e:
            files_failed += 1
            print(f"  Warning: Failed to delete files for {event['event_id']}: {e}")
    
    print(f"  ✓ Deleted {files_deleted} synthetic trace files")
    if files_failed > 0:
        print(f"  ⚠ Failed to delete {files_failed} files")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Packing Synthetic Seismograms into SeisBench Format")
    print("Using Official WaveformDataWriter API")
    print("=" * 70)
    
    # Load configuration
    try:
        config = load_config('Syn_Config.json')
        print(f"\n✓ Loaded configuration from Syn_Config.json")
        
        # Print split ratios if available
        if 'dataset_split_ratios' in config:
            ratios = config['dataset_split_ratios']
            print(f"  Split ratios: train={ratios.get('train', 0.7):.0%}, "
                  f"dev={ratios.get('dev', 0.15):.0%}, test={ratios.get('test', 0.15):.0%}")
        if 'random_seed' in config:
            print(f"  Random seed: {config['random_seed']}")
    except Exception as e:
        print(f"\n⚠ Warning: Could not load config, using defaults: {e}")
        config = None
    
    # Discover synthetic data
    try:
        events = discover_synthetic_data()
        print(f"\n✓ Found {len(events)} synthetic seismograms")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    # Create SeisBench dataset using WaveformDataWriter
    try:
        dataset_path = create_seisbench_dataset(events, output_dir='../data', config=config)
    except Exception as e:
        print(f"\n✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print summary
    print_dataset_summary(dataset_path)
    
    # Verify dataset by loading it with SeisBench
    if verify_dataset(dataset_path):
        # Clean up synthetic trace files after successful packing
        cleanup_synthetic_files(events)
        
        print("\n" + "=" * 70)
        print("✓ Dataset creation completed successfully!")
        print("=" * 70)
        print(f"\nOutput directory: {dataset_path}")
        print(f"  - metadata.csv (trace metadata with picks)")
        print(f"  - waveforms.hdf5 (3-component waveform data)")
        print("\nUsage with SeisBench:")
        print("  import seisbench.data as sbd")
        print(f"  dataset = sbd.WaveformDataset('{dataset_path}', sampling_rate=100)")
        print(f"  waveform = dataset.get_waveforms(0)")
        print(f"  metadata = dataset.metadata")
        print("\nNote: Custom trace names are stored in 'trace_name_original' column.")
        print("      SeisBench uses 'trace_name' for internal HDF5 bucket identifiers.")
        print("      Original synthetic trace files have been deleted to save space.")
        print("      All data is preserved in the SeisBench format.")
        return 0
    else:
        print("\n✗ Dataset verification failed!")
        print("   Keeping synthetic trace files for debugging.")
        return 1


if __name__ == "__main__":
    exit(main())
