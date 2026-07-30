Quantum Graph Neural Network for Higgs Tracking
A research notebook exploring quantum-inspired Graph Neural Networks (GNNs) for reconstructing particle tracks and identifying Higgs boson decay signatures in high-energy physics detector data, benchmarked against a classical GNN baseline.
The project models detector hits as graphs — nodes are individual hits, edges connect spatially nearby hits — and trains both a quantum-inspired and a classical model side by side to compare tracking performance.
---
What's in this repo
This is a single, self-contained Google Colab notebook (`HiggsTracking (4).ipynb`), plus zipped exports of the data, trained models, checkpoints, and results produced by running it.
```
.
├── HiggsTracking (4).ipynb          # Main notebook — the entire pipeline
├── data-*.zip                       # Downloaded / cached CERN Open Data
├── models-*.zip                     # Saved model weights
├── checkpoints-*.zip                # Training checkpoints
├── results-*.zip                    # Metrics, plots, evaluation outputs
├── LICENSE
└── README.md
```
> The zip files are direct exports from the Google Drive folder the notebook uses for storage (`/content/drive/MyDrive/Higgs_Track_Filter`), so their timestamps reflect when the notebook was last run rather than a formal release.
---
How the notebook is organized
The notebook runs top-to-bottom as a single pipeline, organized into these sections:
Setup — mounts Google Drive, installs `torch-geometric` and `uproot`, sets a fixed random seed (`42`) for reproducibility.
Google Drive manager — a small utility class that creates/manages the project's working directories, and falls back to local storage automatically if not running in Colab.
Data generation — a `CERNOpenDataDownloader` that pulls public datasets from the CERN Open Data Portal, including Higgs→4-lepton, Higgs→tau-tau, and CMS track collections.
Hit-to-graph conversion — a `GraphProcessor` that turns raw detector hits into graphs: node features are `x, y, z, pt, layer, significance`, edges are built between the `k` nearest neighbors within a configurable distance threshold, and features are standardized before training.
Quantum-inspired model — a `QuantumInspiredLayer` that mimics qubit-style rotations and entanglement using classical tensor operations (parameterized `sin`/`tanh` transforms rather than an actual quantum circuit or simulator like Qiskit/PennyLane), embedded inside a graph neural network.
Training — a `RobustTrainer` class handling the training loop, validation, early stopping, and metric tracking (loss, accuracy, F1, precision, recall) for both models.
Visualization — a `HiggsVisualizer` that renders 3D particle tracks (Matplotlib/Plotly), colored by track significance.
Main pipeline — `main()` ties everything together: downloads data, builds graphs, trains a quantum-inspired model and a classical baseline under the same conditions, and returns both models with their metrics for comparison. Runtime settings (batch size, epochs, hidden dimension, event count) automatically scale down on CPU and up on GPU.
Saving outputs — trained weights are saved to Google Drive under `QuantumGNN_HiggsTracking/models`.
---
Tech stack
Python 3, run via Google Colab (GPU: T4)
PyTorch + PyTorch Geometric — GNN implementation
uproot — reading physics data formats
NumPy / pandas / scikit-learn — preprocessing and metrics
Matplotlib / Plotly — 3D track visualization
---
Running it
The notebook was built to run in Google Colab with minimal setup:
Open `HiggsTracking (4).ipynb` in Google Colab.
Run the cells in order. The first code cell will mount your Google Drive — accept the authorization prompt.
The dependency cell installs `torch-geometric` and `uproot`; the rest of the notebook runs after that.
Data is downloaded automatically from the CERN Open Data Portal on first run and cached locally / to Drive for subsequent runs.
The notebook detects whether a GPU is available and adjusts batch size, epoch count, and hidden dimensions accordingly, so it's usable on CPU-only runtimes as well (with smaller event counts).
To run it locally instead of Colab, the `GoogleDriveManager` class falls back to a local `./Higgs_Track_Filter` directory automatically — you'll just need PyTorch, PyTorch Geometric, uproot, and the other libraries above installed in your environment.
---
Results
Training produces, per run:
A trained quantum-inspired GNN and a trained classical GNN, saved as separate checkpoints
Validation metrics (loss, accuracy, precision, recall, F1) tracked per epoch for both
3D visualizations of reconstructed particle tracks colored by significance
See the `results-*.zip` archive for the metrics and plots from the notebook's most recent run.
---
Notes & limitations
The "quantum" component is quantum-inspired, not executed on real or simulated quantum hardware — it approximates qubit rotation/entanglement behavior using standard differentiable tensor operations, which keeps the whole pipeline runnable on CPU/GPU without a quantum computing framework.
This is a research/learning project rather than a packaged library — there's no `train.py` entry point or `requirements.txt`; everything lives in the notebook.
---
Author
Dhruv Chaudhari — VIT Bhopal University
Forked from kaustubh2204/Quantum-Graph-Neural-Network-for-Higgs-Tracking.
License
See LICENSE for details.
