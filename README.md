# Icequake_ML

This repository contains a full end-to-end pipeline for training seismic phase detection ML models on icequake data. Starting from raw QuakeML catalog files and `.m` waveform archives, the pipeline curates a SeisBench-compatible dataset and trains a PhaseNet model to identify P and S wave arrivals.

## Pipeline Overview

```
Raw Data (.m / QuakeML) → Picks Filtering → MiniSEED Export → Quality Control
       → Curation (ZNE windows) → SeisBench HDF5 → Training (PhaseNet)
```

---

## 1. Examining the Raw Catalog

*   **`01_exam_quakexml_file.py`**: Reads the QuakeML catalog (e.g., `filtered_events.xml`) and prints event details (origin time, location, magnitude) and raw phase picks to the console.
*   **`01_exam_quakexml_file_explained.ipynb`**: Beginner-friendly companion notebook walking through the script step-by-step.

## 2. Organizing and Filtering Picks

Raw data often contains redundant picks for the same wave (e.g., automatic vs. human-reviewed).

*   **`02_organize_quakexml_file.py`**: Groups picks by station and phase (P/S), filters to a single best pick per station (prioritising `modelled` → `autopick`), and exports to `filtered_picks_organized.csv`.
*   **`02_organize_quakexml_file_explained.ipynb`**: Companion notebook explaining the priority-filtering logic.

## 3. Inspecting Raw Waveform Files (.m)

*   **`ref_read_m_file.py`**: Reference script demonstrating `.m` file reading, filtering, SNR calculation, and spectrogram plotting with `obspy`.
*   **`03_check_raw_m_file.py`**: Prints a detailed metadata summary (Station, Network, Channel, Sampling Rate, Length) for all `.m` files in a directory.
*   **`03_check_raw_m_file_explained.ipynb`**: Companion notebook explaining `.m` file inspection.

## 4. Exporting MiniSEED Files

*   **`04_export_mseed_files.py`**: Unpacks each trace from `.m` archive files into individual standard MiniSEED files under `unpack_top_300_miniseed_raw/`, with a metadata + waveform verification step.
*   **`04_export_mseed_files_explained.ipynb`**: Companion notebook explaining the unpacking and verification logic.

## 5. Matching MiniSEED Files to QuakeMigrate Picks

*   **`05_find_quakemigrate_mseed_files.py`**: Cross-references `filtered_picks_organized.csv` with the MiniSEED archive, locates the correct waveform for each event-station pair, and copies matches into `selected_quick_migrate_mseed/`.
*   **`05_find_quakemigrate_mseed_files_explained.ipynb`**: Companion notebook explaining the file-matching logic.

## 6. Checking Selected Traces

*   **`06_check_selected_traces.py`**: Quality-control script that verifies trace lengths, sampling rates, and pick-time accuracy across all matched MiniSEED files.
*   **`06_check_selected_traces_explained.ipynb`**: Companion notebook explaining the QC checks.

## 7. Curating a Consistent ML Dataset

*   **`07_curate_consistent_ML_dataset.py`**: Slices a fixed-length **10-second window centered on the P-wave arrival** for each event-station pair, enforces strict **ZNE component ordering**, and saves trimmed MiniSEED files and JSON metadata sidecar files to `trimmed_and_consistent_mseed/`.
*   **`07_curate_consistent_ML_dataset_explained.ipynb`**: Companion notebook explaining windowing, resampling, and metadata generation.

## 8. Packing to SeisBench Format

*   **`08_pack_mseed_to_seisbench.py`**: Converts the curated MiniSEED + JSON files into SeisBench-compatible **HDF5 + CSV format**, applying a **70/15/15 train/dev/test split**. Output is written to `final_curated_seisbench_data/`.
*   **`08_pack_mseed_to_seisbench_explained.ipynb`**: Companion notebook explaining the SeisBench packing process.

## 9. Visualizing the Curated Dataset

*   **`09_visualize_curated_final_dataset.py`**: Loads `final_curated_seisbench_data`, prints dataset statistics (station counts, P/S pick distributions), and generates 3-component (Z, N, E) waveform plots with annotated P and S phase arrivals for multiple random traces.
*   **`09_visualize_curated_final_dataset.ipynb`**: Companion notebook explaining loading, statistics, and visualization.

## 10. Generating a Training Configuration

*   **`10_generate_training_config.py`**: Writes `icequake_train_config.json` — a portable config file centralising all training hyperparameters: dataset name, sampling rate, window length, batch size, learning rate, early stopping patience, loss weights, and device settings.
*   **`10_generate_training_config.ipynb`**: Companion notebook explaining each configuration key.

## 11. Demo Model Training

*   **`11_demo_training_ML.py`**: End-to-end training script that:
    1. Loads `icequake_train_config.json` and `final_curated_seisbench_data`.
    2. Applies SeisBench augmentations (`WindowAroundSample`, `RandomWindow`, `Normalize`, `ProbabilisticLabeller`).
    3. Initialises **PhaseNet** and trains with the **Adam** optimiser + **EarlyStopping** (with `tqdm` progress bars per epoch).
    4. Saves best/final model checkpoints to `checkpoints/`.
    5. Generates an annotated **loss history plot** (with best-epoch marker and min val loss annotation) to `output/`.
    6. Produces **5-panel prediction plots** (waveform + ground truth + model output) for 4 random test samples to `output/`.
    7. Evaluates **pick-time residuals** across the full test set and saves shaded P and S residual histograms to `output/`.

---

## Key Outputs

| File / Directory | Description |
|---|---|
| `filtered_picks_organized.csv` | Cleaned, best-pick earthquake catalog |
| `selected_quick_migrate_mseed/` | Matched raw waveforms for catalogued events |
| `trimmed_and_consistent_mseed/` | Curated 10-second ZNE windows + JSON metadata |
| `final_curated_seisbench_data/` | SeisBench-ready HDF5 + CSV dataset |
| `icequake_train_config.json` | Portable training hyperparameter config |
| `checkpoints/best_model.pth` | Best PhaseNet checkpoint (by validation loss) |
| `checkpoints/loss_history.json` | Epoch-by-epoch train/val loss history |
| `output/` | Loss curve, prediction plots, residual histograms |

## Dependencies

- [`obspy`](https://docs.obspy.org/) — Seismology data I/O and processing
- [`seisbench`](https://seisbench.readthedocs.io/) — ML dataset formatting and model zoo (PhaseNet)
- [`torch`](https://pytorch.org/) — Neural network training (PyTorch)
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
- `tqdm` — Progress bars during training and evaluation
