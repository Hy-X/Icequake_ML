# In[ ]
import os
import glob
import csv
import shutil
from collections import defaultdict
from obspy import UTCDateTime

mseed_dir = 'unpack_top_300_miniseed_raw'
csv_file = 'filtered_picks_organized.csv'
output_dir = 'selected_quick_migrate_mseed'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# 1. Index the available MiniSEED files
# Filename format: network.station.channel.start_time.end_time.mseed
# Example: 2E.TJTJ.HHZ.20200724T212502.20200724T212555.mseed
print(f"Indexing MiniSEED files in {mseed_dir}/ ...")
mseed_files = glob.glob(os.path.join(mseed_dir, '*.mseed'))

# We use a dictionary to group file information by station name
mseed_index = defaultdict(list)

for filepath in mseed_files:
    basename = os.path.basename(filepath)
    parts = basename.split('.')
    
    # Ensure the filename matches our expected format
    if len(parts) >= 6:
        station = parts[1]
        start_str = parts[3]
        end_str = parts[4]
        
        try:
            start_time = UTCDateTime(start_str)
            end_time = UTCDateTime(end_str)
            
            # Add this file to the station's list in our index
            mseed_index[station].append({
                'filepath': filepath,
                'basename': basename,
                'start_time': start_time,
                'end_time': end_time
            })
        except Exception as e:
            print(f"Could not parse timestamps in {basename}: {e}")

print(f"Indexed {len(mseed_files)} files across {len(mseed_index)} unique stations.")

# 2. Parse CSV and copy matching files
print(f"\nScanning {csv_file} for wave arrival matches...")
copied_files = set() # To prevent copying the same file twice
match_count = 0
not_found_count = 0
row_count = 0

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_count += 1
        station = row['station']
        
        try:
            pick_time = UTCDateTime(row['pick_time'])
        except Exception:
            print(f"Row {row_count}: Could not parse pick_time {row['pick_time']}")
            continue
            
        found_match = False
        
        # Look up this station in our index
        if station in mseed_index:
            for file_info in mseed_index[station]:
                # Check if the wave arrived while this file was recording
                if file_info['start_time'] <= pick_time <= file_info['end_time']:
                    found_match = True
                    
                    # Check deduplication set to avoid rewriting identical files
                    if file_info['basename'] not in copied_files:
                        src = file_info['filepath']
                        dst = os.path.join(output_dir, file_info['basename'])
                        
                        # Use copy2 to preserve file metadata (timestamps)
                        shutil.copy2(src, dst)
                        copied_files.add(file_info['basename'])
                        match_count += 1
                        
        if not found_match:
            not_found_count += 1

print("\n--- SUMMARY ---")
print(f"Total CSV rows checked: {row_count}")
print(f"Total rows where no covering .mseed was found: {not_found_count}")
print(f"Total unique .mseed files successfully copied: {match_count}")

# In[]:

print("--- VERIFICATION ---")

# 1. Check directory
dir_exists = os.path.exists(output_dir)
print(f"Output Directory Exists: {'✅' if dir_exists else '❌'}")

if dir_exists:
    # 2. Check file count
    actual_files = glob.glob(os.path.join(output_dir, '*.mseed'))
    count_match = len(actual_files) == match_count
    print(f"File Count Matches Expected ({match_count}): {'✅' if count_match else '❌'}")
    
    # 3. Validate a copied file
    if actual_files:
        try:
            import obspy
            # Read the first copied file
            test_st = obspy.read(actual_files[0])
            is_valid = len(test_st) > 0 and len(test_st[0].data) > 0
            print(f"Copied Files are Valid MiniSEED: {'✅' if is_valid else '❌'}")
        except Exception as e:
            print(f"Copied Files are Valid MiniSEED: ❌ (Error: {e})")
else:
    print("Cannot perform further verification: Output directory is missing.")

# In[]:

print("\n--- VISUAL VERIFICATION (3 EXAMPLES) ---")

try:
    examples_shown = 0
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if examples_shown >= 3:
                break
                
            station = row['station']
            try:
                pick_time = UTCDateTime(row['pick_time'])
            except Exception:
                continue
                
            print(f"\nExample {examples_shown + 1}:")
            print(f"  CSV Info Used:")
            print(f"    - Station:   {station}")
            print(f"    - Pick Time: {pick_time}")
            
            if station in mseed_index:
                print(f"  Available MiniSEED files for {station} (Showing first 3 of {len(mseed_index[station])}):")
                for file_info in mseed_index[station][:3]:
                    print(f"    - {file_info['basename']}")
                    print(f"        [Start: {file_info['start_time']} | End: {file_info['end_time']}]")
                
                print(f"  Selected & Copied Files (Where Start <= Pick <= End):")
                found_any = False
                for file_info in mseed_index[station]:
                    if file_info['start_time'] <= pick_time <= file_info['end_time']:
                        print(f"    ✅ {file_info['basename']}")
                        found_any = True
                
                if not found_any:
                    print("    ❌ None matched the pick time.")
            else:
                print(f"  ❌ No files available for station {station} in index.")
                
            examples_shown += 1
except NameError:
    print("Cannot run visual verification: Please ensure the previous cells ran successfully.")

# %%
