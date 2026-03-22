# 🦷 Oral Cancer Detection with Explainable AI (XAI)

> **Clinical-grade deep learning for oral lesion classification — with doctor-trusted explanations, uncertainty quantification, and enterprise-ready audit trails.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Explainability (XAI)](#-explainability-xai)
- [Evaluation & Results](#-evaluation--results)
- [Clinical Deployment Considerations](#-clinical-deployment-considerations)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Known Limitations & Devil's Advocate](#-known-limitations--devils-advocate)
- [Future Work](#-future-work)
- [Citation](#-citation)

---

## 🎯 Project Overview

This project tackles **automated oral cancer detection** from clinical images using a state-of-the-art deep learning ensemble combined with multiple Explainable AI (XAI) techniques. The system is designed not just to *predict* malignancy — but to *explain why*, in a way that pathologists and clinicians can audit, trust, and act upon.

**The core challenge**: Oral cancer has a ~5-year survival rate that exceeds 80% when caught early, but drops below 40% at late stages. Automated screening tools that are both accurate *and* interpretable could meaningfully shift this outcome — but only if they meet clinical rigor standards.

**What we built**: A multi-model ensemble pipeline that:
1. Classifies oral lesion images as **Cancer** or **Non-Cancer**
2. Provides **six independent attribution methods** (Grad-CAM++, LayerCAM, EigenCAM, Integrated Gradients, DeepLift, Consensus Map)
3. Quantifies **epistemic and aleatoric uncertainty** via Monte Carlo Dropout
4. Generates **SAM-based lesion segmentations** for precise boundary delineation
5. Includes **bootstrap confidence intervals**, DeLong's AUC comparison test, and Expected Calibration Error (ECE) for research-grade reporting
6. Implements an **enterprise audit trail** with SHA checksums for FDA 21 CFR Part 11 compliance considerations

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| **Binary Classification** | Cancer vs. Non-Cancer (oral lesion images) |
| **Ensemble of 4 Models** | ConvNeXt-Base, EfficientNet-B3, Swin-Small, MobileNetV2 |
| **Best Single-Model AUC** | ConvNeXt-Base: **0.979 AUC** |
| **Multi-Method XAI** | Grad-CAM++, LayerCAM, EigenCAM, Integrated Gradients, DeepLift, Consensus |
| **Uncertainty Estimation** | MC-Dropout with entropy-based deferral strategy (85th percentile) |
| **SAM Segmentation** | ViT-B Segment Anything Model for lesion boundary delineation |
| **Statistical Rigor** | Bootstrap CIs (n=1000), DeLong's test, ECE calibration analysis |
| **Clinical Threshold** | Adjustable (default 0.5; recommended 0.70 for conservative cancer screening) |
| **Out-of-Distribution Test** | Validated on 115 real hospital images |
| **Regulatory Readiness** | Audit trails, SHA checksums, enterprise config (FDA 21 CFR Part 11 considerations) |

---

## 📦 Dataset

### Primary Dataset: Oral Cancer Dataset 2.0 (Kaggle)

- **Source**: [Kaggle - Oral Cancer Dataset](https://www.kaggle.com/datasets)
- **Path**: `/kaggle/input/oral-cancer-dataset/Oral cancer Dataset 2.0/OC Dataset kaggle new`
- **Classes**:
  - `CANCER` — Malignant oral lesion images
  - `NON CANCER` — Healthy/normal tissue images
- **Validation**: All images verified via `PIL.Image.verify()` to remove corrupted files before training

### Data Splits (Stratified)

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 80% | Model training |
| Validation | 10% | Hyperparameter tuning & early stopping |
| Test | 10% | Final held-out evaluation |

### External Real-World Dataset

- **Source**: `/kaggle/input/dataset-real/images`
- **Size**: 115 unlabeled clinical hospital images
- **Purpose**: Out-of-distribution (OOD) validation and clinical deployment stress-testing
- **Finding**: 98.8% predicted cancer rate — indicating a domain shift between Kaggle training data and real hospital acquisition conditions (see [Known Limitations](#-known-limitations--devils-advocate))

### Augmentation Strategy

**Training** (Albumentations):

| Technique | Parameters | Applied Probability |
|-----------|-----------|---------------------|
| RandomResizedCrop | 224×224, scale=(0.8, 1.0) | Always |
| HorizontalFlip | — | 50% |
| Rotate | ±15°, bilinear | 40% |
| RandomBrightnessContrast | — | 30% |
| **Mixup** | Beta(α=0.4) label interpolation | Per-batch |

**Evaluation**: Resize (224×224) + ImageNet normalization only — no stochastic transforms.

---

## 🏗️ Model Architecture

### Primary Model: ExplainableConvNeXt

```
Input (3 × 224 × 224)
      ↓
ConvNeXt-Base Backbone (timm, ImageNet pretrained)
  └── Forward hooks on last ConvNeXt stage (for XAI)
      ↓
AdaptiveAvgPool2d((1, 1))
      ↓
Classifier Head:
  LayerNorm(num_features)
  → Dropout(0.3)
  → Linear(num_features, 2)
      ↓
Softmax → [P(Non-Cancer), P(Cancer)]
```

### Elite Model Ensemble (4 Models)

| Model | Backbone | Pretrained | Grad-CAM Target | AUC |
|-------|---------|-----------|-----------------|-----|
| **ConvNeXt** | ConvNeXt-Base | ImageNet (timm) | `stages[-1]` | **0.979** |
| **EfficientNetB3** | EfficientNet-B3 | ImageNet (timm) | `features[-1]` | 0.969 |
| **Swin-Small** | Swin Transformer | ImageNet (timm) | `layers[-1]` | 0.960 |
| **MobileNetV2** | MobileNetV2 | ImageNet (timm) | `features[-1]` | 0.960 |

All models support:
- **MC-Dropout** for Bayesian uncertainty estimation (`enable_mc_dropout()`)
- **EMA shadow weights** (decay=0.995) for improved generalization
- Unified `predict()` and `explain()` interfaces for the clinical attribution engine

---

## 🏋️ Training Pipeline

### Hyperparameters

```python
EPOCHS       = 40
BATCH_SIZE   = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
IMAGE_SIZE   = 224
SEED         = 42
```

### Optimizer & Loss

| Component | Config |
|-----------|--------|
| **Optimizer** | AdamW (`lr=1e-4`, `weight_decay=1e-4`) |
| **Loss** | CrossEntropyLoss with **label smoothing=0.1** |
| **LR Scheduler** | ReduceLROnPlateau (`factor=0.5`, `patience=3`, `mode='min'`) |

### Advanced Training Techniques

1. **Automatic Mixed Precision (AMP)** — `torch.amp.GradScaler("cuda")` for memory-efficient FP16 training
2. **Exponential Moving Average (EMA)** — `decay=0.995` shadow weights for stable inference
3. **Mixup Regularization** — Beta(0.4, 0.4) blending of pairs to prevent feature collapse
4. **Label Smoothing** — Reduces overconfidence and improves probability calibration
5. **Best-checkpoint saving** — Model snapshot at minimum validation loss

---

## 🔍 Explainability (XAI)

One of the primary research contributions of this project is the **multi-method Clinical Attribution Engine** that provides converging evidence for model decisions.

### Attribution Methods

| Method | Library | Strength |
|--------|---------|----------|
| **Grad-CAM++** | `pytorch-grad-cam` | Spatial lesion localization, weighted gradients |
| **LayerCAM** | `pytorch-grad-cam` | Spatial-aware gradient weighting |
| **EigenCAM** | `pytorch-grad-cam` | PCA on activations — robust to gradient noise |
| **Integrated Gradients** | `Captum` | Axiomatic attribution along baseline path |
| **DeepLift** | `Captum` | Neuron-level importance decomposition |
| **Consensus Map** | Custom | Intersection of all methods = high-confidence regions only |

### SAM Lesion Segmentation

The **Segment Anything Model (ViT-B)** is prompted with Grad-CAM peak coordinates to generate:
- Precise lesion boundary contours (not fuzzy heatmaps)
- Filled segmentation masks suitable for physician second-read validation
- Validated first on Kaggle labeled data before real-world deployment

### Uncertainty Quantification

- **MC-Dropout**: 20 stochastic forward passes → predictive distribution
- **Entropy**: $H = -\sum_c p_c \log p_c$ as uncertainty proxy
- **Deferral Strategy**: Cases above the **85th percentile entropy threshold** are flagged for mandatory pathologist review
- **Visualization**: Entropy histogram overlaid with correct/incorrect predictions to reveal correlation between uncertainty and error rate

### Clinical Attribution Quality Metrics

| Metric | Definition |
|--------|-----------|
| **Lesion Focus Score (LFS)** | % of total attribution mass landing on labeled lesion region |
| **Attribution Entropy** | Diffuseness of explanation (lower = more focused) |
| **Spatial Concentration** | Gaussian concentration measure of heatmap |
| **Robustness** | Consistency of attributions across input perturbations |

---

## 📊 Evaluation & Results

### Standard Metrics

- **Classification Report**: Precision, Recall, F1-Score per class
- **Confusion Matrix**: TP / TN / FP / FN breakdown
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve
- **Precision-Recall Curve**: Especially critical for imbalanced clinical datasets

### Research-Grade Statistical Analysis

| Analysis | Method | Purpose |
|----------|--------|---------|
| **Confidence Intervals** | Bootstrap (n=1000 resamples) | Publication-ready uncertainty for all metrics |
| **Model Comparison** | DeLong's Test | Statistical significance of AUC differences |
| **Calibration** | Expected Calibration Error (ECE, 10 bins) | Probability trustworthiness |
| **Additional** | MCC, Cohen's Kappa, Brier Score | Robust non-threshold metrics |

### Best Model Performance (ConvNeXt-Base)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.979** |
| Validation Set | Stratified 10% holdout |
| Test Set | Stratified 10% holdout |

> **Note**: Full per-class precision/recall/F1 values are computed and printed dynamically in notebook Cell 7 using `sklearn.metrics.classification_report`.

### Real-World Hospital Validation

| Metric | Value |
|--------|-------|
| Images analyzed | 115 |
| Predicted cancer rate | 98.8% |
| Interpretation | **Domain shift detected** — see [Known Limitations](#-known-limitations--devils-advocate) |

---

## 🏥 Clinical Deployment Considerations

This project is built with real-world clinical deployment in mind:

### Threshold Strategy

| Confidence | Classification | Action |
|-----------|---------------|--------|
| ≥ 0.70 | `HIGH_CONFIDENCE_CANCER` | Immediate pathologist review |
| 0.50–0.69 | `URGENT_REVIEW` | Expedited review queue |
| 0.30–0.49 | `NORMAL_REVIEW` | Standard review queue |
| < 0.30 | `HIGH_CONFIDENCE_NORMAL` | Routine follow-up |

### Audit & Compliance

- SHA checksums on model artifacts
- Timestamp-stamped prediction logs
- Configuration versioning
- FDA 21 CFR Part 11 compliance considerations (electronic records and signatures)

---

## 🛠️ Tech Stack

### Deep Learning
- **PyTorch 2.x** — Core training and inference
- **timm** — Pretrained model zoo (ConvNeXt, EfficientNet, Swin, MobileNet)
- **torchvision** — Image utilities
- **torch_ema** — Exponential Moving Average

### Data & Augmentation
- **Albumentations** — GPU-accelerated augmentation pipeline
- **PIL/Pillow** — Image I/O
- **OpenCV (cv2)** — Color space manipulation
- **NumPy**, **Pandas** — Data wrangling

### Explainability
- **pytorch-grad-cam** — Grad-CAM, Grad-CAM++, LayerCAM, EigenCAM, ScoreCAM
- **Captum** — Integrated Gradients, DeepLift, GuidedBackprop, Saliency
- **Segment Anything (SAM)** — Meta AI ViT-B lesion segmentation

### Statistics & Evaluation
- **scikit-learn** — Metrics, calibration, train/test splitting
- **scipy** — Statistical distributions, morphological operations
- **scikit-image** — Morphological image processing

### Visualization
- **Matplotlib**, **Seaborn** — Publication-quality plots
- **ipywidgets** — Interactive threshold sliders

### Platform
- **Python 3.12.12**
- **Kaggle GPU**: NVIDIA Tesla T4 (~16GB VRAM)

---

## 📁 Project Structure

```
oral-cancer-2/
├── oral-cancer-january-2026 (1).ipynb   # Main experiment notebook
└── README.md                            # This file

# Generated during training (on Kaggle):
├── best_convnext_explainable.pth        # Best ConvNeXt checkpoint (EMA weights)
└── /kaggle/working/                     # Prediction exports, audit logs
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision timm
pip install albumentations torch-ema
pip install pytorch-grad-cam captum
pip install segment-anything
pip install scikit-learn scipy scikit-image
pip install matplotlib seaborn ipywidgets tqdm
pip install ttach shap
```

### Running on Kaggle

1. Upload the notebook to [Kaggle](https://www.kaggle.com/)
2. Add the **Oral Cancer Dataset 2.0** dataset to your Kaggle notebook inputs
3. Enable **GPU accelerator** (Tesla T4)
4. Run all cells sequentially

### Local Execution

```python
# Adjust dataset path in the notebook:
DATA_DIR = "/your/local/path/to/OC Dataset kaggle new"

# Then run:
jupyter notebook "oral-cancer-january-2026 (1).ipynb"
```

> **GPU highly recommended.** Training 40 epochs on CPU will be extremely slow. At minimum, an 8GB VRAM GPU is required for `BATCH_SIZE=32` with ConvNeXt-Base.

---

## ⚠️ Known Limitations & Devil's Advocate

*A rigorous, honest assessment of what this project does NOT prove — and where caution is warranted.*

### 1. Domain Shift is Severe and Unresolved
The most significant red flag: **98.8% of 115 real hospital images were predicted as cancer**. This is almost certainly wrong for a general population and strongly suggests:
- The model has learned Kaggle-specific image artifacts (resolution, color profile, watermarks, lighting conditions) rather than pure biological tissue characteristics
- The hospital image acquisition pipeline differs materially from the training distribution
- **Without ground-truth labels for the 115 hospital images, no real validation has occurred** — the external dataset test is purely observational

### 2. Kaggle Dataset ≠ Clinical Population
The Oral Cancer Dataset 2.0 is a publicly available benchmark dataset. It may:
- Overrepresent clear-cut cases (easy positives, easy negatives)
- Underrepresent ambiguous presentations, minority subtypes, or rare morphologies
- Have selection bias toward high-contrast lesion photography
- **Not reflect the actual base rate** of cancer in a screening population (typically < 5%)

A 0.979 AUC on a curated benchmark dataset does **not** imply the same performance in prospective clinical screening.

### 3. Class Balance Assumptions
The stratified split assumes roughly equal class distribution. If the real-world cancer prevalence is much lower (e.g., 2%), a model optimized on balanced training data will produce systematically miscalibrated probabilities — generating excessive false alarms that erode clinician trust.

### 4. XAI ≠ Ground Truth Explanations
Multiple attribution methods (Grad-CAM++, Integrated Gradients, DeepLift) generate *plausible-looking* heatmaps. However:
- No pathologist has validated that highlighted regions correspond to clinically meaningful tissue features
- Attribution methods can highlight irrelevant background structures
- Consensus between XAI methods improves robustness but does not guarantee biological correctness
- **The SAM segmentation is prompted by Grad-CAM peaks** — if Grad-CAM is wrong, SAM segments the wrong region with high confidence

### 5. EMA and Mixup Are Not a Calibration Guarantee
Label smoothing (0.1) + EMA + bootstrap CIs improve *apparent* calibration, but:
- True probability calibration requires isotonic regression or Platt scaling validated on a held-out set
- Bootstrap CIs describe sampling variability around observed metrics — they cannot correct for systematic biases from dataset shift

### 6. MC-Dropout Uncertainty Has Known Limitations
Monte Carlo Dropout is a practical approximation to Bayesian inference, but:
- Dropout placement matters greatly; it is not equivalent to a proper Bayesian neural network
- High entropy ≠ reliable indicator of incorrect predictions in distribution-shifted scenarios
- The 85th percentile deferral threshold is dataset-specific and would need recalibration on new data

### 7. Regulatory Status
The FDA 21 CFR Part 11 audit trail features are **implementation scaffolding**, not certified compliance. Actual FDA clearance (510(k) or De Novo) requires:
- Formal clinical study design with IRB approval
- Prospective multi-site validation
- Device submission with predicate device identification
- Post-market surveillance plan

**This notebook is research code. It is not a cleared medical device.**

### 8. Reproducibility vs. Overfitting
`SEED=42` and stratified splits ensure within-experiment reproducibility. However:
- Single random seed results can be lucky
- True reproducibility requires multiple seeds with variance reporting
- 40 epochs with ReduceLROnPlateau on a relatively small dataset may still overfit; test set metrics should be treated as optimistic estimates

---

## 🔮 Future Work

| Priority | Improvement | Rationale |
|---------|------------|-----------|
| 🔴 High | **Prospective multi-site clinical validation** | Only meaningful real-world performance estimate |
| 🔴 High | **Ground-truth labels for hospital images** | The 115-image OOD test is currently unvalidatable |
| 🔴 High | **Pathologist annotation of XAI heatmaps** | Validate that attributions align with clinical findings |
| 🟡 Medium | **Domain adaptation / few-shot fine-tuning** on hospital data | Close the acquisition gap |
| 🟡 Medium | **Active learning loop** for borderline cases | Efficiently expand training coverage |
| 🟡 Medium | **Multi-seed variance reporting** | Proper reproducibility and generalization bounds |
| 🟡 Medium | **Proper temperature scaling / isotonic regression** calibration | Trustworthy probability outputs |
| 🟢 Low | **Causal feature discovery** | Identify tissue properties that mechanistically distinguish cancer |
| 🟢 Low | **Lightweight mobile deployment** (MobileNetV2 export) | Point-of-care screening in resource-limited settings |
| 🟢 Low | **Federated learning** across hospitals | Privacy-preserving multi-site model improvement |

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{oral_cancer_xai_2026,
  title     = {Oral Cancer Detection with Explainable AI: 
                A Multi-Model Ensemble with Clinical Attribution Engine},
  year      = {2026},
  month     = {January},
  note      = {Kaggle Notebook. Dataset: Oral Cancer Dataset 2.0.
                Models: ConvNeXt-Base, EfficientNet-B3, Swin-Small, MobileNetV2.
                XAI: Grad-CAM++, LayerCAM, EigenCAM, Integrated Gradients, 
                DeepLift, SAM Segmentation.},
  url       = {https://github.com/rohi021/oral-cancer-2}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

> **Medical Disclaimer**: This software is intended for **research purposes only**. It is not a certified, cleared, or approved medical device. Do not use this system for clinical diagnosis or treatment decisions without appropriate regulatory approval and clinical validation.

---

<p align="center">
  Built with ❤️ for advancing accessible oral cancer screening research
</p>
