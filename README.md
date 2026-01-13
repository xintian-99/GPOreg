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

## 🧱 Install PyTorch3D (required)

This project uses `pytorch3d.ops.knn_points` for fast KNN on GPU. PyTorch3D must be installed with **CUDA support**, otherwise you may see:

> `RuntimeError: Not compiled with GPU support`

### Option A (recommended): prebuilt wheels (match Python / PyTorch / CUDA)
Install PyTorch3D from the official wheel index:

- Command: `pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/<WHEEL_INDEX>/download.html`

### Option B (fallback): build from source
- Commands:
  - `pip install -U pip setuptools wheel`
  - `pip install -v "git+https://github.com/facebookresearch/pytorch3d.git@stable"`

**HPC tip:** ensure CUDA toolkit is available (e.g., `module load CUDA/...`) before installing from source.

---

## 🗂️ Dataset format (FIRE)

This repo includes a FIRE dataset loader: `loaders/firereg_loader.py`.

Your `--datasets_path` must contain:

- `<data_root>/ipa_org_img/`  
  - `P01_1.jpg`, `P01_2.jpg`, ...
- `<data_root>/ipa_txt/`  
  - `control_points_P01_1_2.txt`, ...
- `<data_root>/geo_kps_1000/`  
  - `P01_1_2.csv`, ...

### ❓ What is `ipa_*`?

These are folder names used by this repo’s FIRE packaging:

- `ipa_org_img/` → input retinal images  
- `ipa_txt/` → FIRE landmark/control point text files  
- `geo_kps_<N>/` → precomputed descriptor correspondences used for geo initialization (DCN mode)

They are not a universal FIRE naming requirement—just the structure expected by `firereg_loader.py`. If your FIRE files are stored differently, edit:

- `self.img_folder = 'ipa_org_img'`
- `self.kps_folder = 'ipa_txt'`
- `self.geo_folder = f'geo_kps_{self.geo_num}'`

---

## 🚀 Quick Start (exact commands)

Minimal sanity run on split A (GCN / grid nodes):  
- `python gpo.py --n_iters 2 --enable_geo_init 0 --field_split A`

Run only one sample (P14) and save warped images:  
- `python gpo.py --field_split P --only_sub P14 --n_iters 2 --save_images 1 --save_losses 0`

---

## 🧪 Modes and mapping to CLI flags

**🧩 DCN mode (descriptor control nodes)**  
Uses `geo_kps_<geo_num>/<sub_id>_1_2.csv` as initialization:  
- `python gpo.py --enable_geo_init 1 --geo_num 1000`

**🟦 GCN mode (grid control nodes)**  
Uses a regular grid defined by `--node_shape`:  
- `python gpo.py --enable_geo_init 0 --node_shape 30`

---

## ⚙️ Key arguments

| Flag | Example | Meaning | Notes |
|---|---:|---|---|
| `--field_split` | `P` | FIRE split subset (`A/P/S`) | Must match the ID prefix |
| `--only_sub` | `P14` | Run exactly one image pair | Optional |
| `--enable_geo_init` | `1` | DCN-like init vs grid nodes | `1`: geo init, `0`: grid |
| `--geo_num` / `--node_shape` | `1000` / `30` | Geo correspondences or grid resolution | `geo_kps_1000/` expected when geo init |

---

## 📤 Outputs

**📄 CSV results**  
Saved to: `results/<dataset>/<auto_generated_name>.csv`  
Each row contains: `init_tre`, `eval_tre`, runtime per pair, AUC-style success scores at thresholds `[5, 10, 15, 25, 50]`, and up to 10 per-point errors.

**🖼️ Warped images (optional)**  
If `--save_images 1`, warped outputs are saved via `utils/setters.py`.

**📉 Loss logs (optional)**  
If `--save_losses 1`, per-iteration losses are saved via `utils/setters.py`.

---

## 📚 Citation

@InProceedings{TiaXin_Gaussian_MICCAI2025,
        author = { Tian, Xin AND Wang, Jiazheng AND Zhang, Yuxi AND Chen, Xiang AND Hu, Renjiu AND Li, Gaolei AND Liu, Min AND Zhang, Hang},
        title = { { Gaussian Primitive Optimized Deformable Retinal Image Registration } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15963},
        month = {September},
        page = {218 -- 228}
}



