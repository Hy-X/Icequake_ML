# In[]

import obspy
import glob
import os

# Define source and destination directories
waveform_dir = 'top_300_raw_cut_waveforms/'
output_dir = 'unpack_top_300_miniseed_raw/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Find all .m files
file_pattern = os.path.join(waveform_dir, '*.m')
m_files = glob.glob(file_pattern)

if not m_files:
    print(f"No .m files found in {waveform_dir}. Please check the path.")
else:
    print(f"Found {len(m_files)} .m files. Starting extraction...")
    
    total_traces_extracted = 0
    
    # Process each file
    for file_path in m_files:
        try:
            # Read the .m file into an ObsPy Stream
            st = obspy.read(file_path)
            
            # Extract each trace and save as an individual .mseed file
            for tr in st:
                # Extract metadata, handling empty networks just in case
                network = tr.stats.network if tr.stats.network else "XX"
                station = tr.stats.station
                channel = tr.stats.channel
                
                # Format timestamps to avoid invalid filename characters (like colons)
                # Format: YYYYMMDDTHHMMSS
                start_time_str = tr.stats.starttime.strftime('%Y%m%dT%H%M%S')
                end_time_str = tr.stats.endtime.strftime('%Y%m%dT%H%M%S')
                
                # Construct the clean filename
                # Example: 2E.TJTJ.HHZ.20200724T212502.20200724T212555.mseed
                filename = f"{network}.{station}.{channel}.{start_time_str}.{end_time_str}.mseed"
                output_path = os.path.join(output_dir, filename)
                
                # Write to the new file using the MSEED format
                tr.write(output_path, format="MSEED")
                total_traces_extracted += 1
                
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    print(f"\nSuccess! Extracted {total_traces_extracted} individual MiniSEED files into '{output_dir}'.")

# In[]

## This is to verify the extracted mseed files

# Read an original file (dynamically grabbing the first one available)
orig_files = glob.glob('top_300_raw_cut_waveforms/*.m')
if orig_files:
    orig_file = orig_files[0]
    orig = obspy.read(orig_file)
    orig_tr = orig[0]
    
    print(f"Verifying using original file: {os.path.basename(orig_file)}")

    # Construct the exact filename we expect for this trace
    network = orig_tr.stats.network if orig_tr.stats.network else "XX"
    station = orig_tr.stats.station
    channel = orig_tr.stats.channel
    start_time_str = orig_tr.stats.starttime.strftime('%Y%m%dT%H%M%S')
    end_time_str = orig_tr.stats.endtime.strftime('%Y%m%dT%H%M%S')
    expected_filename = f"{network}.{station}.{channel}.{start_time_str}.{end_time_str}.mseed"
    
    unpacked_path = os.path.join('unpack_top_300_miniseed_raw', expected_filename)

    if os.path.exists(unpacked_path):
        unp = obspy.read(unpacked_path)
        unp_tr = unp[0]
        
        print('\n--- ORIGINAL METADATA ---')
        for k, v in orig_tr.stats.items():
            print(f'{k}: {v}')
            
        print('\n--- UNPACKED MINISEED METADATA ---')
        for k, v in unp_tr.stats.items():
            print(f'{k}: {v}')
    else:
        print(f"\nCould not find expected unpacked file: {unpacked_path}")
else:
    print("No original .m files found to verify.")

# In[]:

import numpy as np

try:
    # Verify Metadata (using core keys)
    keys_to_check = ['network', 'station', 'channel', 'starttime', 'endtime', 'sampling_rate', 'npts']
    metadata_match = all(orig_tr.stats[k] == unp_tr.stats[k] for k in keys_to_check)
    
    # Verify Data (the actual waveform numpy arrays)
    data_match = np.array_equal(orig_tr.data, unp_tr.data)
    
    print("\n--- VERIFICATION SUMMARY ---")
    print(f"Metadata Match: {'✅' if metadata_match else '❌'}")
    print(f"Waveform Data Match: {'✅' if data_match else '❌'}")
except NameError:
    print("Cannot verify: Please ensure the previous cell ran successfully and found the traces.")

# %%
