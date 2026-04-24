# Icequake_ML

This repository is intended for Icequake ML with quick migrate results. It includes tools to explore, clean, and convert QuakeML outputs into formats compatible with machine learning workflows (like SeisBench).

## 1. Examining the Raw Data

Before cleaning your data, it's helpful to see what is inside the raw QuakeML file.

*   **`01_exam_quakexml_file.py`**: A simple script that reads the QuakeML catalog (e.g., `filtered_events.xml`) and prints out the basic event details (origin time, location, magnitude) and raw, unfiltered wave arrivals (picks) to the console.
*   **`01_exam_quakexml_file_explained.ipynb`**: A beginner-friendly Jupyter Notebook that walks through the `01_exam_quakexml_file.py` script step-by-step. Perfect for college students or anyone new to `obspy`.

## 2. Organizing and Filtering Picks

Raw earthquake data often contains redundant picks for the exact same wave (e.g., an automatic computer guess vs a human-reviewed model).

*   **`02_organize_quakexml_file.py`**: This script reads the events, groups the wave arrivals by station and wave phase (P or S), and strictly filters them to select the single *best* pick for that station. It prioritizes `modelled` picks, falls back to `autopick`, and otherwise takes the first available pick. It exports this cleaned dataset to `filtered_picks_organized.csv`.
*   **`02_organize_quakexml_file_explained.ipynb`**: A companion Jupyter Notebook that breaks down the logic and priority-filtering of this script into easy-to-understand steps.

## 3. Inspecting Raw Waveform Files (.m)

In addition to the catalog metadata, we also process the actual physical waveform recordings (saved as `.m` files).

*   **`ref_read_m_file.py`**: A reference script demonstrating how to find `.m` files by their origin timestamp, read them using `obspy`, apply filters, calculate Signal-to-Noise Ratios (SNR), and plot spectrograms.
*   **`03_check_raw_m_file.py`**: A utility script that searches a directory for `.m` files and prints out a detailed summary of the trace metadata within them (Station, Network, Channel, Sampling Rate, Total Length).
*   **`03_check_raw_m_file_explained.ipynb`**: A companion Jupyter Notebook that explains the process of reading and extracting metadata from these raw waveform files step-by-step.

## 4. QuakeML to SeisBench CSV Converter

*   **`quakeml_to_seisbench.py`**: Reads a QuakeML event catalog and generates a `metadata.csv` file that is fully compatible with SeisBench. It extracts event origins and phase arrivals, formatting them exactly as required for model training or evaluation.

### Usage for SeisBench Conversion

```bash
python quakeml_to_seisbench.py --input filtered_events.xml --output metadata.csv
```
