# GPO: Gaussian Primitive Optimized Deformable Retinal Image Registration

**Early accepted by MICCAI 2025**

Deformable retinal image registration using Gaussian Primitive Optimization (GPO) framework with adaptive local deformations and descriptor-based control nodes.

## Overview

GPOreg introduces a novel iterative framework that addresses the notorious difficulty of deformable retinal image registration. The method learns deformable displacement fields from minimal descriptor-based control nodes (DCN), strategically placed at salient anatomical structures like major vessels.

### Key Innovation

- **Gaussian Primitive Optimization (GPO)**: Each control node is modeled as a Gaussian primitive with trainable position, displacement, and radius
- **Adaptive Spatial Influence**: Nodes adapt their influence to local deformation scales
- **KNN Gaussian Interpolation**: Blends local transformations into globally coherent displacement fields
- **Strategic Node Placement**: Anchors nodes in high-gradient regions to ensure robust gradient flow

## Performance

On the FIRE dataset, GPOreg achieves:
- **Target Registration Error**: Reduced from 5.9px to 2.4px
- **AUC at 25px**: Increased from 0.770 to 0.938

## Usage

Run the registration with different configurations:

```bash
# Without geometric initialization, field split A
python test_firereg.py --n_iters 200 --enable_geo_init 0 --field_split A

# With geometric initialization, field split P
python test_firereg.py --n_iters 100 --enable_geo_init 1 --field_split P

# With geometric initialization, field split S
python test_firereg.py --n_iters 100 --enable_geo_init 1 --field_split S
```

## Parameters

- `--n_iters`: Number of optimization iterations
- `--enable_geo_init`: Enable/disable geometric initialization (0/1)
- `--field_split`: Field splitting strategy (A/P/S)

## Requirements

- Python 3.x
- Required dependencies (see requirements.txt)

## Citation

If you use this code in your research, please cite our MICCAI 2025 paper:

```bibtex
@inproceedings{gporeg2025,
    title={Gaussian Primitive Optimized Deformable Retinal Image Registration},
    author={[Author Names]},
    booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
    year={2025}
}
```

# 🧿 GPO: Gaussian Primitive Optimized Deformable Retinal Image Registration

📄 **Paper (MICCAI 2025, LNCS 15963)**: *Gaussian Primitive Optimized Deformable Retinal Image Registration*  
🔗 **arXiv**: https://arxiv.org/abs/2508.16852  
🔗 **Springer (MICCAI proceedings chapter)**: https://link.springer.com/chapter/10.1007/978-3-032-04965-0_21  

This repository contains the reference implementation for the paper above, providing a **pairwise** deformable retinal registration method based on **Gaussian primitives** and **KNN-weighted blending** optimized via image similarity losses.

---

## ✨ What this code does (high level)

Given a retinal image pair:

- **Fixed** image: `*_1.jpg`
- **Moving** image: `*_2.jpg`

the method optimizes a deformation model parameterized by **Gaussian control nodes**. Each node contributes a local motion; pixel-wise displacement is obtained by **KNN Gaussian weighting**. The model parameters are optimized directly (test-time optimization) to align the two images, and performance is evaluated using landmark error (**TRE**) and threshold-based success metrics.

---

## 🧠 Modes supported (paper ↔ code)

The paper describes two practical setups:

- **🧩 GPO-DCN (Descriptor/Correspondence Control Nodes)**  
  Uses precomputed dense correspondences (“geo points”) to initialize node locations and translations.  
  ✅ In code: `--enable_geo_init 1`

- **🟦 GPO-GCN (Grid Control Nodes)**  
  Uses a regular grid of control nodes when correspondences are unavailable.  
  ✅ In code: `--enable_geo_init 0` (uses `--node_shape`)

---

## 📦 Installation

### 1) Create environment and install dependencies
```bash
pip install -r requirements.txt

