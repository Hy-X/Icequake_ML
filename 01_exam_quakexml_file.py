
# In[ ]

#  read quakexml
# output the columns 

import obspy

# Define the path to your XML file
xml_file = "filtered_events.xml"

try:
    # Read the XML file
    cat = obspy.read_events(xml_file)

    # Print basic information
    print(f"Number of events found: {len(cat)}")

    # Loop through events to inspect data
    for i, event in enumerate(cat, start=1):
        print(f"\n--- Event {i} ---")
        
        # Get event details
        print(f"Event ID: {event.resource_id}")
        if event.origins:
            print(f"Origin Time: {event.origins[0].time}")
            print(f"Latitude: {event.origins[0].latitude}")
            print(f"Longitude: {event.origins[0].longitude}")
            print(f"Depth: {event.origins[0].depth}")
            
        if event.magnitudes:
            print(f"Magnitude: {event.magnitudes[0].mag}")
        else:
            print(f"Magnitude: Not available")
        
        # Print associated picks if available
        if event.picks:
            print("Picks:")
            for pick in event.picks:
                station = pick.waveform_id.station_code if pick.waveform_id else "Unknown"
                method = str(pick.method_id) if pick.method_id else "Unknown"
                print(f"  - Station: {station}, Phase: {pick.phase_hint}, Time: {pick.time}, Method ID: {method}")
        
        #if i >= 3:
        #    
        #    print("\n... stopping after 3 events.")
        #    break
            
except Exception as e:
    print(f"Error reading file: {e}")



# In[ ]:
