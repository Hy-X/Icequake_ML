import argparse
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict

def parse_quakeml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # QuakeML namespaces
    namespaces = {
        'q': 'http://quakeml.org/xmlns/quakeml/1.2',
        'bed': 'http://quakeml.org/xmlns/bed/1.2'
    }
    
    events_data = []
    
    # Extract events
    for event in root.findall('.//bed:event', namespaces):
        event_dict = {}
        
        # Get source_id
        public_id = event.get('publicID')
        # Extract the local part after 'smi:local/' if it exists
        if public_id and 'smi:local/' in public_id:
            source_id = public_id.split('smi:local/')[-1]
        else:
            source_id = public_id
        
        event_dict['source_id'] = source_id
        
        # Get preferred origin (or first origin)
        origin = event.find('bed:origin', namespaces)
        if origin is not None:
            time_elem = origin.find('.//bed:time/bed:value', namespaces)
            if time_elem is not None:
                event_dict['source_origin_time'] = time_elem.text
                
            lat_elem = origin.find('.//bed:latitude/bed:value', namespaces)
            if lat_elem is not None:
                event_dict['source_latitude_deg'] = float(lat_elem.text)
                
            lon_elem = origin.find('.//bed:longitude/bed:value', namespaces)
            if lon_elem is not None:
                event_dict['source_longitude_deg'] = float(lon_elem.text)
                
            depth_elem = origin.find('.//bed:depth/bed:value', namespaces)
            if depth_elem is not None:
                # QuakeML depth is usually in meters, convert to km
                event_dict['source_depth_km'] = float(depth_elem.text) / 1000.0
                
        # Get picks
        picks_by_station = defaultdict(dict)
        for pick in event.findall('.//bed:pick', namespaces):
            phase_elem = pick.find('bed:phaseHint', namespaces)
            time_elem = pick.find('.//bed:time/bed:value', namespaces)
            waveform_elem = pick.find('bed:waveformID', namespaces)
            
            if phase_elem is not None and time_elem is not None and waveform_elem is not None:
                phase = phase_elem.text
                time = time_elem.text
                station = waveform_elem.get('stationCode')
                
                picks_by_station[station][phase] = time
                
        event_dict['picks'] = picks_by_station
        events_data.append(event_dict)
        
    return events_data

def export_to_seisbench_csv(events_data, output_csv):
    # Seisbench metadata columns we will export
    columns = [
        'trace_name',
        'source_id',
        'source_origin_time',
        'source_latitude_deg',
        'source_longitude_deg',
        'source_depth_km',
        'station_code',
        'trace_p_arrival_time',
        'trace_s_arrival_time'
    ]
    
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        
        for event in events_data:
            source_id = event.get('source_id', '')
            
            for station, phases in event.get('picks', {}).items():
                trace_name = f"{source_id}_{station}"
                
                row = {
                    'trace_name': trace_name,
                    'source_id': source_id,
                    'source_origin_time': event.get('source_origin_time', ''),
                    'source_latitude_deg': event.get('source_latitude_deg', ''),
                    'source_longitude_deg': event.get('source_longitude_deg', ''),
                    'source_depth_km': event.get('source_depth_km', ''),
                    'station_code': station,
                    'trace_p_arrival_time': phases.get('P', ''),
                    'trace_s_arrival_time': phases.get('S', '')
                }
                writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert QuakeML to SeisBench metadata.csv')
    parser.add_argument('--input', '-i', type=str, default='filtered_events.xml', help='Input QuakeML file')
    parser.add_argument('--output', '-o', type=str, default='metadata.csv', help='Output metadata.csv file')
    args = parser.parse_args()
    
    print(f"Parsing {args.input}...")
    events = parse_quakeml(args.input)
    print(f"Found {len(events)} events.")
    
    export_to_seisbench_csv(events, args.output)
    print(f"Successfully exported metadata to {args.output}")
