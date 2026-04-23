# In[]

import obspy
from collections import defaultdict

# Define the path to the XML file
xml_file = "filtered_events.xml"

try:
    # Read the XML file
    cat = obspy.read_events(xml_file)

    # Print basic information
    print(f"Number of events found: {len(cat)}")

    # Loop through events to inspect data
    for i, event in enumerate(cat, start=1):
        print(f"\n--- Event {i} ---")
        print(f"Event ID: {event.resource_id}")
        
        if event.origins:
            print(f"Origin Time: {event.origins[0].time}")
        
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
                        
                        # Fallback to the first one if 'modelled' wasn't found
                        if not selected_pick:
                            selected_pick = picks[0]
                    else:
                        # If there's only one pick, just use it
                        selected_pick = picks[0]
                        
                    method = str(selected_pick.method_id) if selected_pick.method_id else "Unknown"
                    print(f"  - Station: {station}, Phase: {phase}, Time: {selected_pick.time}, Method ID: {method}")
        else:
            print("No picks found.")
        
        # Stop after 3 events to prevent flooding the console
        #if i >= 3:
        #    print("\n... stopping after 3 events.")
        #    break
            
except Exception as e:
    print(f"Error reading file: {e}")

# %%
