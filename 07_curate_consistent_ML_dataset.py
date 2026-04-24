# In[]

import os
import glob
import csv
import json
from collections import defaultdict
from obspy import read, UTCDateTime

csv_file = 'filtered_picks_organized.csv'
input_dir = 'selected_quick_migrate_mseed'
output_dir = 'trimmed_and_consistent_mseed'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# 1. Group picks by (event_id, station) to find the absolute earliest arrival
print("Parsing CSV to find local arrival times for each event/station...")
# Dictionary structure: station_event_picks[(event_id, station)] = {'P': time, 'S': time}
station_event_picks = defaultdict(dict)

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        event_id = row['event_id']
        station = row['station']
        phase = row['phase'] # Usually 'P' or 'S'
        try:
            pick_time = UTCDateTime(row['pick_time'])
            station_event_picks[(event_id, station)][phase] = pick_time
        except Exception:
            continue

print(f"Identified {len(station_event_picks)} unique station-event pairs.")

# 2. Index the input MiniSEED files by station for fast lookup
print(f"Indexing input files in '{input_dir}'...")
input_files = glob.glob(os.path.join(input_dir, '*.mseed'))
mseed_index = defaultdict(list)

for filepath in input_files:
    basename = os.path.basename(filepath)
    parts = basename.split('.')
    if len(parts) >= 6:
        station = parts[1]
        try:
            start_time = UTCDateTime(parts[3])
            end_time = UTCDateTime(parts[4])
            mseed_index[station].append({
                'filepath': filepath,
                'network': parts[0],
                'station': station,
                'channel': parts[2],
                'start_time': start_time,
                'end_time': end_time
            })
        except Exception:
            pass

# 3. Slice the traces and generate JSON metadata
print(f"Slicing traces into exactly 10-second windows and writing JSON...")
success_count = 0
not_found_count = 0

# We use a set to avoid slicing the exact same window multiple times
processed_windows = set()

for (event_id, station), picks in station_event_picks.items():
    # Find the earliest pick for this station-event to serve as the reference time
    ref_time = min(picks.values())
    window_start = ref_time - 5.0
    window_end = ref_time + 5.0
    
    found_covering_file = False
    saved_channels = []
    saved_files = []
    network = "XX"
    sample_rate = 200.0
    
    if station in mseed_index:
        for file_info in mseed_index[station]:
            # Does this file cover the entire 10s window we need?
            if file_info['start_time'] <= window_start and file_info['end_time'] >= window_end:
                found_covering_file = True
                
                # Format timestamps for new filename
                w_start_str = window_start.strftime('%Y%m%dT%H%M%S')
                w_end_str = window_end.strftime('%Y%m%dT%H%M%S')
                
                new_filename = f"{file_info['network']}.{file_info['station']}.{file_info['channel']}.{w_start_str}.{w_end_str}.mseed"
                new_filepath = os.path.join(output_dir, new_filename)
                
                # Check for duplicates
                if new_filename not in processed_windows:
                    try:
                        # Read, slice, and write!
                        st = read(file_info['filepath'])
                        tr = st[0]
                        tr_sliced = tr.slice(starttime=window_start, endtime=window_end)
                        
                        # Only save if the slice contains data
                        if tr_sliced.stats.npts > 0:
                            tr_sliced.write(new_filepath, format="MSEED")
                            processed_windows.add(new_filename)
                            
                            saved_channels.append(file_info['channel'])
                            saved_files.append(new_filename)
                            network = file_info['network']
                            sample_rate = tr_sliced.stats.sampling_rate
                            success_count += 1
                    except Exception as e:
                        print(f"Failed to slice {file_info['filepath']}: {e}")
                        
    if not found_covering_file:
        not_found_count += 1
    else:
        # Generate JSON metadata for this station-event pair
        if saved_files:
            # Sort channels to ensure order: HH1, HH2, HHZ or HHE, HHN, HHZ
            channel_order = {"1": 0, "E": 0, "2": 1, "N": 1, "Z": 2}
            def sort_key(pair):
                chan = pair[0]
                return channel_order.get(chan[-1], 99)
                
            paired = list(zip(saved_channels, saved_files))
            paired.sort(key=sort_key)
            saved_channels = [p[0] for p in paired]
            saved_files = [p[1] for p in paired]
            
            json_filename = f"{network}.{station}.{w_start_str}.{w_end_str}.json"
            json_filepath = os.path.join(output_dir, json_filename)
            
            metadata = {
                "event_id": event_id,
                "station": station,
                "network": network,
                "start_time": window_start.strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z',
                "sample_rate": sample_rate,
                "duration": 10.0,
            }
            
            if 'P' in picks:
                p_relative = picks['P'] - window_start
                metadata["p_arrival_sample"] = int(p_relative * sample_rate)
                metadata["p_arrival_time"] = p_relative
                
            if 'S' in picks:
                s_relative = picks['S'] - window_start
                metadata["s_arrival_sample"] = int(s_relative * sample_rate)
                metadata["s_arrival_time"] = s_relative
                
            metadata["channels"] = saved_channels
            metadata["files"] = {
                "mseed": saved_files
            }
            
            # Write JSON file
            with open(json_filepath, 'w') as json_file:
                json.dump(metadata, json_file, indent=4)

print("\n--- SUMMARY ---")
print(f"Successfully sliced and saved {success_count} consistent 10s traces.")
print(f"Successfully generated JSON metadata files alongside the traces.")
print(f"Could not find covering .mseed file for {not_found_count} station-events.")

# %%
