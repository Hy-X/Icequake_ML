# Icequake_ML

This repository is intended for Icequake ML with quick migrate results. It includes tools to convert QuakeML outputs into a SeisBench compatible format for machine learning workflows.

## QuakeML to SeisBench CSV Converter

The script `quakeml_to_seisbench.py` reads a QuakeML event catalog (`filtered_events.xml`) and generates a `metadata.csv` file that is compatible with SeisBench. It extracts event origins and phase arrivals, grouping them into individual waveform traces for model training or evaluation.

### Usage

```bash
python quakeml_to_seisbench.py --input filtered_events.xml --output metadata.csv
```
