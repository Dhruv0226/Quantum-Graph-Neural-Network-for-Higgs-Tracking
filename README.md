<div align="center">
⚛️ Quantum Graph Neural Network for Higgs Tracking
Quantum-inspired vs. classical GNNs for particle track reconstruction on real CERN Open Data
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-EE4C2C?logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyG-Graph%20Learning-3C2179)
![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)
</div>
---
A research notebook exploring quantum-inspired Graph Neural Networks (GNNs) for reconstructing particle tracks and identifying Higgs boson decay signatures in high-energy physics detector data, benchmarked head-to-head against a classical GNN baseline.
The project models detector hits as graphs — nodes are individual hits, edges connect spatially nearby hits — and trains both a quantum-inspired and a classical model under identical conditions to compare tracking performance.
---
📦 What's in this repo
This is a single, self-contained Google Colab notebook, plus zipped exports of the data, trained models, checkpoints, and results produced by running it.
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
> 💡 The zip files are direct exports from the Google Drive folder the notebook uses for storage (`/content/drive/MyDrive/Higgs_Track_Filter`), so their timestamps reflect when the notebook was last run rather than a formal release.
---
🧭 Pipeline overview
The notebook runs top-to-bottom as a single pipeline:
Stage	What happens
1. Setup	Mounts Google Drive, installs `torch-geometric` & `uproot`, fixes random seed (`42`)
2. Drive manager	Creates/manages project directories; falls back to local storage outside Colab
3. Data generation	`CERNOpenDataDownloader` pulls public datasets — Higgs→4ℓ, Higgs→ττ, CMS tracks — from the CERN Open Data Portal
4. Hit → graph	`GraphProcessor` builds graphs from hits: node features `x, y, z, pt, layer, significance`; edges via k-nearest-neighbors within a distance threshold; features standardized
5. Quantum-inspired model	`QuantumInspiredLayer` mimics qubit rotation/entanglement using differentiable `sin`/`tanh` tensor ops, embedded in a GNN
6. Training	`RobustTrainer` runs the training loop with early stopping and tracks loss, accuracy, F1, precision, recall
7. Visualization	`HiggsVisualizer` renders 3D particle tracks (Matplotlib/Plotly), colored by significance
8. Main pipeline	`main()` downloads data, builds graphs, and trains both models side by side — settings auto-scale for CPU vs. GPU
9. Saving	Trained weights saved to Google Drive under `QuantumGNN_HiggsTracking/models`
---
🛠️ Tech stack
Category	Tools
Language / Runtime	Python 3, Google Colab (GPU: T4)
Deep learning	PyTorch, PyTorch Geometric
Physics data	uproot
Data & metrics	NumPy, pandas, scikit-learn
Visualization	Matplotlib, Plotly
---
🚀 Running it
Open `HiggsTracking (4).ipynb` in Google Colab.
Run the cells in order — the first will mount your Google Drive (accept the auth prompt).
The dependency cell installs `torch-geometric` and `uproot`.
Data downloads automatically from the CERN Open Data Portal on first run and is cached for subsequent runs.
GPU vs. CPU is auto-detected, adjusting batch size, epochs, and hidden dimensions accordingly — so it runs on CPU too, just with fewer events.
Running locally instead of Colab? The `GoogleDriveManager` class falls back to a local `./Higgs_Track_Filter` directory automatically. You'll just need PyTorch, PyTorch Geometric, uproot, and the other libraries above installed.
---
📊 Results
Each run produces:
A trained quantum-inspired GNN and a trained classical GNN, saved as separate checkpoints
Per-epoch validation metrics (loss, accuracy, precision, recall, F1) for both models
3D visualizations of reconstructed particle tracks colored by significance
See `results-*.zip` for metrics and plots from the most recent run.
---
⚠️ Notes & limitations
The "quantum" component is quantum-inspired, not run on real or simulated quantum hardware — it approximates qubit rotation/entanglement behavior with standard differentiable tensor operations, keeping the whole pipeline runnable on ordinary CPU/GPU without a quantum computing framework.
This is a research/learning project, not a packaged library — there's no `train.py` entry point or `requirements.txt`; everything lives in the notebook.
---
👥 Authors
This project was built collaboratively as part of the original kaustubh2204/Quantum-Graph-Neural-Network-for-Higgs-Tracking repository.
Dhruv Chaudhari — Collaborator, VIT Bhopal University
This repo is a fork used for continued development; see the original repository above for the full contributor history.
📄 License
See LICENSE for details.
