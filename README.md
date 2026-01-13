# 🧿 GPOreg: Gaussian Primitive Optimized Deformable Retinal Image Registration

✅ **Early accepted by MICCAI 2025**  
📌 **Deformable retinal image registration using Gaussian Primitive Optimization (GPO)** with adaptive local deformations and descriptor-based control nodes.

🔗 **arXiv**: https://arxiv.org/abs/2508.16852  
🔗 **Springer (MICCAI proceedings chapter)**: https://link.springer.com/chapter/10.1007/978-3-032-04965-0_21  

---

## ✨ Overview

**GPOreg** introduces an iterative optimization framework to address the notorious difficulty of **deformable retinal image registration**—especially when gradient signals are sparse in vessel-poor regions. The method estimates a dense displacement field from a **minimal set of control nodes**, where nodes can be:

- **DCN (Descriptor-based Control Nodes)**: strategically placed at salient anatomical structures (e.g., major vessels), or
- **GCN (Grid Control Nodes)**: regular grid nodes used when descriptor nodes are not available.

### Key Innovation

- **Gaussian Primitive Optimization (GPO)**: Each control node is modeled as a Gaussian primitive with trainable position, displacement, and radius
- **Adaptive Spatial Influence**: Nodes adapt their influence to local deformation scales
- **KNN Gaussian Interpolation**: Blends local transformations into globally coherent displacement fields
- **Strategic Node Placement**: Anchors nodes in high-gradient regions to ensure robust gradient flow

## Performance

On the FIRE dataset, GPOreg achieves:
- **Target Registration Error**: Reduced from 5.9px to 2.4px
- **AUC at 25px**: Increased from 0.770 to 0.938

## 📦 Installation

### 1) Install Python dependencies
```bash
pip install -r requirements.txt
```

