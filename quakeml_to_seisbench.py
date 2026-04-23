import argparse
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict
from datetime import datetime

def parse_quakeml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    namespaces = {
        'q': 'http://quakeml.org/xmlns/quakeml/1.2',
        'bed': 'http://quakeml.org/xmlns/bed/1.2'
    }
    
    events_data = []
    
    for event in root.findall('.//bed:event', namespaces):
        event_dict = {}
        
        public_id = event.get('publicID')
        if public_id and 'smi:local/' in public_id:
            source_id = public_id.split('smi:local/')[-1]
        else:
            source_id = public_id
        
        event_dict['source_id'] = source_id
        
        origin = event.find('bed:origin', namespaces)
        if origin is not None:
            time_elem = origin.find('.//bed:time/bed:value', namespaces)
            if time_elem is not None:
                event_dict['source_origin_time'] = time_elem.text
                
            lat_elem = origin.find('.//bed:latitude/bed:value', namespaces)
            if lat_elem is not None:
                event_dict['source_latitude'] = float(lat_elem.text)
                
            lon_elem = origin.find('.//bed:longitude/bed:value', namespaces)
            if lon_elem is not None:
                event_dict['source_longitude'] = float(lon_elem.text)
                
            depth_elem = origin.find('.//bed:depth/bed:value', namespaces)
            if depth_elem is not None:
                event_dict['source_depth_km'] = float(depth_elem.text) / 1000.0
                
        picks_by_station = defaultdict(dict)
        for pick in event.findall('.//bed:pick', namespaces):
            phase_elem = pick.find('bed:phaseHint', namespaces)
            time_elem = pick.find('.//bed:time/bed:value', namespaces)
            waveform_elem = pick.find('bed:waveformID', namespaces)
            method_elem = pick.find('bed:methodID', namespaces)
            
            if phase_elem is not None and time_elem is not None and waveform_elem is not None:
                phase = phase_elem.text
                time = time_elem.text
                station = waveform_elem.get('stationCode')
                network = waveform_elem.get('networkCode', '')
                
                method = method_elem.text if method_elem is not None else ""
                if 'smi:local/' in method:
                    method = method.split('smi:local/')[-1]
                
                picks_by_station[station][phase] = {
                    'time': time,
                    'network': network,
                    'status': method
                }
                
        event_dict['picks'] = picks_by_station
        events_data.append(event_dict)
        
    return events_data

def calculate_travel_time(pick_time_str, origin_time_str):
    try:
        pick_time = datetime.fromisoformat(pick_time_str.replace('Z', '+00:00'))
        origin_time = datetime.fromisoformat(origin_time_str.replace('Z', '+00:00'))
        return round((pick_time - origin_time).total_seconds(), 6)
    except Exception:
        return ""

def export_to_seisbench_csv(events_data, output_csv):
    columns = [
        'network_code', 'receiver_code', 'receiver_type', 'receiver_latitude',
        'receiver_longitude', 'receiver_elevation_m', 'p_arrival_sample',
        'p_status', 'p_weight', 'p_travel_sec', 's_arrival_sample', 's_status',
        's_weight', 'source_id', 'source_origin_time',
        'source_origin_uncertainty_sec', 'source_latitude', 'source_longitude',
        'source_error_sec', 'source_gap_deg', 'source_horizontal_uncertainty_km',
        'source_depth_km', 'source_depth_uncertainty_km', 'source_magnitude',
        'source_magnitude_type', 'source_magnitude_author',
        'source_mechanism_strike_dip_rake', 'source_distance_deg',
        'source_distance_km', 'back_azimuth_deg', 'snr_db', 'coda_end_sample',
        'trace_start_time', 'trace_category', 'trace_name',
        # Extra columns appended to prevent data loss
        's_travel_sec', 'p_arrival_time', 's_arrival_time'
    ]
    
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        
        for event in events_data:
            source_id = event.get('source_id', '')
            source_origin_time = event.get('source_origin_time', '')
            
            for station, phases in event.get('picks', {}).items():
                trace_name = f"{source_id}_{station}"
                
                p_data = phases.get('P', {})
                s_data = phases.get('S', {})
                
                network = p_data.get('network', s_data.get('network', ''))
                
                p_arrival_time = p_data.get('time', '')
                p_travel_sec = calculate_travel_time(p_arrival_time, source_origin_time) if p_arrival_time and source_origin_time else ''
                
                s_arrival_time = s_data.get('time', '')
                s_travel_sec = calculate_travel_time(s_arrival_time, source_origin_time) if s_arrival_time and source_origin_time else ''
                
                row = {
                    'network_code': network,
                    'receiver_code': station,
                    'receiver_type': '',
                    'receiver_latitude': '',
                    'receiver_longitude': '',
                    'receiver_elevation_m': '',
                    'p_arrival_sample': '',
                    'p_status': p_data.get('status', ''),
                    'p_weight': 1 if p_arrival_time else 0,
                    'p_travel_sec': p_travel_sec,
                    's_arrival_sample': '',
                    's_status': s_data.get('status', ''),
                    's_weight': 1 if s_arrival_time else 0,
                    'source_id': source_id,
                    'source_origin_time': source_origin_time,
                    'source_origin_uncertainty_sec': '',
                    'source_latitude': event.get('source_latitude', ''),
                    'source_longitude': event.get('source_longitude', ''),
                    'source_error_sec': '',
                    'source_gap_deg': '',
                    'source_horizontal_uncertainty_km': '',
                    'source_depth_km': event.get('source_depth_km', ''),
                    'source_depth_uncertainty_km': '',
                    'source_magnitude': '',
                    'source_magnitude_type': '',
                    'source_magnitude_author': '',
                    'source_mechanism_strike_dip_rake': '',
                    'source_distance_deg': '',
                    'source_distance_km': '',
                    'back_azimuth_deg': '',
                    'snr_db': '',
                    'coda_end_sample': '',
                    'trace_start_time': '',
                    'trace_category': '',
                    'trace_name': trace_name,
                    's_travel_sec': s_travel_sec,
                    'p_arrival_time': p_arrival_time,
                    's_arrival_time': s_arrival_time
                }
                writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert QuakeML to SeisBench metadata.csv')
    parser.add_argument('--input', '-i', type=str, default='filtered_events.xml', help='Input QuakeML file')
    parser.add_argument('--output', '-o', type=str, default='metadata.csv', help='Output metadata.csv file')
    args = parser.parse_args()
    
    events = parse_quakeml(args.input)
    export_to_seisbench_csv(events, args.output)
