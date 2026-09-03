# 🧿 GPOreg: Gaussian Primitive Optimized Deformable Retinal Image Registration

**MICCAI 2025 (Early accepted)** 

**Authors**: Xin Tian, Jiazheng Wang, Yuxi Zhang, Xiang Chen, Renjiu Hu, Gaolei Li, Min Liu, Hang Zhang

[![arXiv](https://img.shields.io/badge/arXiv-2508.16852-b31b1b.svg)](https://arxiv.org/abs/2508.16852) [![Springer](https://img.shields.io/badge/Springer-MICCAI%202025-37677e.svg)](https://link.springer.com/chapter/10.1007/978-3-032-04965-0_21)

---

## Overview

GPOreg registers a pair of fundus images by test-time optimization of a sparse set of **Gaussian primitives** (control nodes), each with a trainable position, displacement and radius. The dense displacement field is obtained by K-nearest-neighbour Gaussian blending of the nodes, so the method stays well-conditioned even in vessel-poor regions where image gradients are sparse. Nodes are either **DCN** (descriptor control nodes, placed on matched vessel keypoints) or **GCN** (a regular grid). 

## Installation

Tested with Python 3.9, PyTorch 2.0.1 + CUDA 11.7, PyTorch3D 0.7.4.

```bash
pip install -r requirements.txt
```
PyTorch3D must be built with CUDA support (needed for pytorch3d.ops.knn_points):
```bash
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
```
or build from source: 
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

If you see `RuntimeError: Not compiled with GPU support`, PyTorch3D was installed without CUDA.

## Data

`data/fire/` ships the **28-pair FIRE test subset** used for the numbers above
(A12–A14, P40–P49, S57–S71), in the layout the loader expects:

```
data/fire/img/{ID}_1.jpg, {ID}_2.jpg            # image pair (fixed, moving)
data/fire/gt_kps/control_points_{ID}_1_2.txt    # 10 ground-truth landmarks: x1 y1 x2 y2
data/fire/geo_kps_1000/{ID}_1_2.csv             # 1000 matched vessel keypoints (DCN init)
```

For the full dataset (133 pairs) download [FIRE](https://projects.ics.forth.gr/cvrl/fire/) and
place all pairs in the same layout; `--field_split A|P|S` then selects the official categories.

## 🚀 Quick Start +  Modes

> The bundled `data/fire/` contains only the 28-pair `test` split (A12–A14, P40–P49, S57–S71), which is what `python gpo.py` runs by default.
> Commands that use `--field_split A|P|S` or other pair IDs require the full FIRE dataset placed in the same layout.

### DCN mode (descriptor control nodes)
DCN initializes control nodes from precomputed descriptor correspondences (e.g., `geo_kps_<geo_num>/<sub_id>_1_2.csv`).

- Run DCN (example with `geo_num=1000`):  
  `python gpo.py --enable_geo_init 1 --geo_num 1000`

- DCN single-pair run (P46, 2 iterations, save warped images):  
  `python gpo.py --only_sub P46 --n_iters 2 --save_images 1`

### GCN mode (grid control nodes)
GCN uses a regular control-node grid (no descriptor initialization); grid resolution is controlled by `--node_shape`.

- Run GCN (example with `node_shape=30`):  
  `python gpo.py --enable_geo_init 0 --node_shape 30`

- Minimal GCN sanity run (split A, 2 iterations) — *requires the full dataset*:  
  `python gpo.py --n_iters 2 --enable_geo_init 0 --field_split A`
 
### Key arguments

| Flag | Example | Meaning | Notes |
|---|---:|---|---|
| `--field_split` | `P` | FIRE split subset (`A/P/S`) | `test` = bundled subset (default); `A/P/S` need the full dataset |
| `--only_sub` | `P14` | Run exactly one image pair | Pair must exist in the data folder (bundled: e.g. P46) |
| `--enable_geo_init` | `1` | DCN-like init vs grid nodes | `1`: geo init, `0`: grid |
| `--geo_num` / `--node_shape` | `1000` / `30` | Geo correspondences or grid resolution | `geo_kps_1000/` expected when geo init | 
---

## Outputs

- `results/firereg/<run_name>.csv` — one row per pair (`init_tre`, `eval_tre`, time, AUC@5/10/15/25/50,
  per-landmark errors) plus an `Avg` row. `<run_name>` encodes the hyperparameters.
- `results/warped_img/<run_name>/{ID}_1_warped.png` (viridis) and `{ID}_1_warped_rgb.png` (true colour) with `--save_images 1`.
- `logs/losses/<run_name>.csv` — per-iteration losses (appended across runs).

### Performance

On the FIRE dataset, GPOreg achieves:
| Setting | TRE (px) | AUC@25 |
|---|---|---|
| Paper — full FIRE (133 pairs) | ~2.4 | 0.770 → 0.938 |
| This repo — bundled 28-pair test split, `python gpo.py` | 2.4–2.7 | 0.91–0.93 |

---

## Citation

```bibtex
@InProceedings{TiaXin_Gaussian_MICCAI2025,
  author    = {Tian, Xin and Wang, Jiazheng and Zhang, Yuxi and Chen, Xiang and Hu, Renjiu and Li, Gaolei and Liu, Min and Zhang, Hang},
  title     = {Gaussian Primitive Optimized Deformable Retinal Image Registration},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  series    = {LNCS},
  volume    = {15963},
  pages     = {218--228}
}
```

If you use the bundled FIRE subset, please also cite the dataset:

```bibtex
@article{HernandezMatas_FIRE_2017,
  author  = {Hernandez-Matas, Carlos and Zabulis, Xenophon and Triantafyllou, Areti and Anyfanti, Panagiota and Douma, Stella and Argyros, Antonis A.},
  title   = {FIRE: Fundus Image Registration Dataset},
  journal = {Journal for Modeling in Ophthalmology},
  year    = {2017},
  volume  = {1},
  number  = {4},
  pages   = {16--28},
  doi     = {10.35119/maio.v1i4.42}
}
```

## License

Code is released under the [MIT License](LICENSE). The `data/fire/` subset remains the property of the FIRE dataset authors — see the data notice in [LICENSE](LICENSE) and the [official FIRE page](https://projects.ics.forth.gr/cvrl/fire/).
