# In[]:

import os
import glob
import csv
import numpy as np
from obspy import read, UTCDateTime

mseed_dir = 'selected_quick_migrate_mseed'
csv_file = 'filtered_picks_organized.csv'

print(f"Loading pick times from {csv_file}...")

# 1. Load picks from CSV
# We store them in a dictionary grouped by station for fast lookup
picks_by_station = {}
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sta = row['station']
        try:
            pt = UTCDateTime(row['pick_time'])
            if sta not in picks_by_station:
                picks_by_station[sta] = []
            picks_by_station[sta].append(pt)
        except Exception:
            continue

# 2. Process the MiniSEED files
mseed_files = glob.glob(os.path.join(mseed_dir, '*.mseed'))
print(f"Found {len(mseed_files)} MiniSEED files to check in '{mseed_dir}'.\n")

# Setup a nice formatted table
print(f"{'MiniSEED Filename':<48} | {'Length (sec)':<12} | {'Samples':<8} | {'Pick Diff (sec)':<15}")
print("-" * 90)

all_lengths = []
all_samples = []
all_pick_diffs = []
files_processed = 0

for filepath in mseed_files:
    filename = os.path.basename(filepath)
    
    try:
        # Read the trace
        st = read(filepath)
        tr = st[0]
        
        sta = tr.stats.station
        start = tr.stats.starttime
        end = tr.stats.endtime
        
        # Calculate trace lengths
        length_sec = end - start
        samples = tr.stats.npts
        
        all_lengths.append(length_sec)
        all_samples.append(samples)
        
        # Find any picks that fall within this trace's recording window
        valid_picks = [pt for pt in picks_by_station.get(sta, []) if start <= pt <= end]
        
        # Calculate the difference between starttime and pick_time
        if valid_picks:
            diff_secs = [pt - start for pt in valid_picks]
            all_pick_diffs.extend(diff_secs)
            diff_str = ", ".join([f"{d:.2f}" for d in diff_secs])
        else:
            diff_str = "No Pick Found"
            
        # Print the first 20 files as a visual check
        if files_processed < 20:
            print(f"{filename:<48} | {length_sec:<12.2f} | {samples:<8} | {diff_str:<15}")
            
        files_processed += 1
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")

print("-" * 90)
if files_processed > 20:
    print(f"... and {files_processed - 20} more files checked silently.")

# Print overall summary statistics
if files_processed > 0:
    print(f"\n--- COMPREHENSIVE OVERALL SUMMARY ---")
    
    print("\n[ Trace Lengths (seconds) ]")
    print(f"  Mean: {np.mean(all_lengths):.2f}")
    print(f"  Min:  {np.min(all_lengths):.2f}")
    print(f"  Max:  {np.max(all_lengths):.2f}")
    print(f"  Std:  {np.std(all_lengths):.2f}")
    
    print("\n[ Trace Samples ]")
    print(f"  Mean: {np.mean(all_samples):.0f}")
    print(f"  Min:  {np.min(all_samples):.0f}")
    print(f"  Max:  {np.max(all_samples):.0f}")
    print(f"  Std:  {np.std(all_samples):.0f}")
    
    if all_pick_diffs:
        print("\n[ Pick Differences (seconds from trace start) ]")
        print(f"  Mean: {np.mean(all_pick_diffs):.2f}")
        print(f"  Min:  {np.min(all_pick_diffs):.2f}")
        print(f"  Max:  {np.max(all_pick_diffs):.2f}")
        print(f"  Std:  {np.std(all_pick_diffs):.2f}")
    else:
        print("\n[ Pick Differences ]")
        print("  No matching picks found to calculate stats.")

# %%
