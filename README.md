# Data Poisoning Toolkit

This toolkit is a comprehensive framework designed to test data poisoning attacks against machine learning models and various defense mechanisms against these attacks.

## Features

- **Multiple Attack Types**:
  - Random Poisoning Attack
  - Greedy Residue Attack
  - D-SPO (Differentiable Shielded Poisoning Oracle) Attack

- **Multiple Defense Mechanisms**:
  - LOF (Local Outlier Factor)
  - PCA (Principal Component Analysis)
  - KDE (Kernel Density Estimation)
  - Isolation Forest
  - Median Filter
  - Quantile Filter
  - DBSCAN

- **Dataset Support**:
  - Water Quality Dataset
  - Smart Grid Stability Dataset
  - Traffic Index Dataset
  - Solar Power Generation Dataset
  - Wind Power Dataset
  - Brisbane Water Quality Monitoring Dataset

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src
python -m datapoisontoolkit.demo_runner
```

## Configuration

All experiments can be configured through the `ExperimentConfig` class:
- Datasets
- Attack types and parameters
- Defense mechanisms
- Experiment duration and budget

## Author

Anil Yagiz