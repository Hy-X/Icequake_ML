# In[]:

import obspy
import glob
import os

# Define the path to the raw waveform files
# Note: You may need to change this if your files are in a different location!
waveform_dir = 'top_300_raw_cut_waveforms/'
file_pattern = os.path.join(waveform_dir, '*.m')

print(f"Searching for waveform files in: {waveform_dir}")

try:
    # Find all .m files
    m_files = glob.glob(file_pattern)
    
    if not m_files:
        print("No .m files found. Please check the directory path.")
    else:
        print(f"Successfully found {len(m_files)} .m files!")
        print("Examining the first 3 files...\n")
        
        # Loop through a couple of files to check them out
        for i, file_path in enumerate(m_files[:3], start=1):
            print(f"--- File {i}: {os.path.basename(file_path)} ---")
            
            try:
                # Read the waveform file using obspy
                st = obspy.read(file_path)
                
                print(f"Total Traces inside file: {len(st)}")
                
                # Inspect each trace inside the file
                for j, tr in enumerate(st, start=1):
                    print(f"  Trace {j} Info:")
                    print(f"    - Station Code:  {tr.stats.station}")
                    print(f"    - Network Code:  {tr.stats.network}")
                    print(f"    - Channel:       {tr.stats.channel}")
                    print(f"    - Start Time:    {tr.stats.starttime}")
                    print(f"    - End Time:      {tr.stats.endtime}")
                    print(f"    - Sampling Rate: {tr.stats.sampling_rate} Hz")
                    print(f"    - Data Points:   {tr.stats.npts}")
                    print(f"    - Total Length:  {tr.stats.endtime - tr.stats.starttime:.2f} seconds")
                    print(f"    - Full Stats Dictionary:\n {tr.stats}")
                    print("")
                    
            except Exception as e:
                print(f"  Error reading this specific file: {e}\n")
                
except PermissionError:
    print(f"Permission Denied: You do not have access to read from '{waveform_dir}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %%
