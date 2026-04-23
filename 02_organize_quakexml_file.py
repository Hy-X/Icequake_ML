# In[]

import obspy
import csv
from collections import defaultdict

# Define the path to the XML file
xml_file = "filtered_events.xml"

try:
    # Read the XML file
    cat = obspy.read_events(xml_file)

    # Print basic information
    print(f"Number of events found: {len(cat)}")
    
    csv_filename = "filtered_picks_organized.csv"
    csvfile = open(csv_filename, mode='w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['event_id', 'origin_time', 'latitude', 'longitude', 'depth_m', 'station', 'phase', 'pick_time', 'method_id'])
    print(f"Exporting data to {csv_filename}...")

    # Loop through events to inspect data
    for i, event in enumerate(cat, start=1):
        event_id = str(event.resource_id).replace("smi:local/", "")
        origin_time = str(event.origins[0].time) if event.origins else ""
        lat = str(event.origins[0].latitude) if event.origins else ""
        lon = str(event.origins[0].longitude) if event.origins else ""
        depth = str(event.origins[0].depth) if event.origins else ""
        
        print(f"\n--- Event {i} ---")
        print(f"Event ID: {event_id}")
        
        if event.origins:
            print(f"Origin Time: {origin_time}")
            print(f"Latitude: {lat}")
            print(f"Longitude: {lon}")
            print(f"Depth: {depth}")
        
        # Organize picks by station and phase
        # Dictionary structure: { 'station_code': { 'P': [pick1, pick2], 'S': [pick1] } }
        organized_picks = defaultdict(lambda: defaultdict(list))
        
        if event.picks:
            for pick in event.picks:
                station = pick.waveform_id.station_code if pick.waveform_id else "Unknown"
                phase = pick.phase_hint
                organized_picks[station][phase].append(pick)
            
            print("Selected Picks (Prioritizing 'smi:local/modelled'):")
            
            # Select the best pick for each station and phase
            for station, phases in organized_picks.items():
                for phase, picks in phases.items():
                    selected_pick = None
                    
                    if len(picks) > 1:
                        # If there are multiple picks, look for the 'modelled' one
                        for p in picks:
                            method = str(p.method_id) if p.method_id else ""
                            if "modelled" in method:
                                selected_pick = p
                                break
                                
                        # Fallback to autopick if 'modelled' wasn't found
                        if not selected_pick:
                            for p in picks:
                                method = str(p.method_id) if p.method_id else ""
                                if "autopick" in method:
                                    selected_pick = p
                                    break
                        
                        # Fallback to the first one if neither was found
                        if not selected_pick:
                            selected_pick = picks[0]
                    else:
                        # If there's only one pick, just use it
                        selected_pick = picks[0]
                        
                    method = str(selected_pick.method_id).replace("smi:local/", "") if selected_pick.method_id else "Unknown"
                    print(f"  - Station: {station}, Phase: {phase}, Time: {selected_pick.time}, Method ID: {method}")
                    writer.writerow([event_id, origin_time, lat, lon, depth, station, phase, str(selected_pick.time), method])
        else:
            print("No picks found.")
        
        # Stop after 3 events to prevent flooding the console
        #if i >= 3:
        #    print("\n... stopping after 3 events.")
        #    break
    
    csvfile.close()
    print(f"\nSuccessfully exported data to {csv_filename}")
            
except Exception as e:
    print(f"Error reading file: {e}")

# %%
