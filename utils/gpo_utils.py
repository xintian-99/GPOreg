import torch

def make_coords(shape, mask=None):

    n_dim = len(shape)
    vectors = [torch.arange(0, s) for s in shape]
    coords = torch.meshgrid(vectors, indexing="ij")
    coords = torch.stack(coords, dim=-1)
    coords = coords.view(-1, n_dim)

    if mask is not None:
        coords = coords[mask.flatten()]

    return coords

def sparse_sampling_prep(img_size, _flow, _coords):

    _flow = _flow.unsqueeze(0).unsqueeze(0)
    _coords = _coords.unsqueeze(0).unsqueeze(0)
    new_coords = _coords + _flow
    for idx_, sha in enumerate(img_size):
        new_coords[...,idx_] = 2 * (new_coords[...,idx_]/(sha-1) - 0.5)
        _coords[...,idx_] = 2 * (_coords[...,idx_]/(sha-1) - 0.5)
    new_coords_out = new_coords.contiguous()[..., [1,0]]
    _coords_out = _coords.contiguous()[..., [1,0]]

    return new_coords_out, _coords_out