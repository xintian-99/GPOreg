import torch
import math
import numpy as np
import pytorch3d.ops as ops

from torch import nn
from tqdm import tqdm
from torch.nn import functional as F

class gaussianSplatComplex(nn.Module):
    def __init__(self,
        K = '20',
        node_shape = 30,
        img_shape = '[1024,1024]',
        x_geos=None, 
        y_geos=None
    ):
        super(gaussianSplatComplex, self).__init__()

        self.K = int(K)
        # self.node_shape = eval(node_shape)
        if isinstance(node_shape, int):
            self.node_shape = [node_shape, node_shape]  # Convert single int to [int, int]
        else:
            self.node_shape = node_shape  # If already a list, keep it as-is

        self.img_shape = eval(img_shape)
        self.n_dim = len(self.node_shape)
        self.img_shape = eval(img_shape)
        self.n_dim = len(self.node_shape)

        # print("KNN K: %d, Node Shape: %s, Image Shape: %s, Max Densify Ratio: %.2f, Max Scaling: %.2f" % (self.K, self.node_shape, self.img_shape, self.max_densify_ratio, self.max_scaling))

        node_coords, node_radius_cap = self.make_coors(self.node_shape, self.img_shape)
        self.node_radius_cap = 4*node_radius_cap
        if x_geos is not None and y_geos is not None:
            y_geos = y_geos.squeeze()
            x_geos = x_geos.squeeze()
            self.node_init_num = x_geos.shape[0]
            self.node_position = nn.Parameter(y_geos)
            self.translation = nn.Parameter(x_geos - y_geos)
        else:
            self.node_init_num = np.prod(self.node_shape)
            self.node_position = nn.Parameter(node_coords)
            self.translation = nn.Parameter(torch.zeros(self.node_init_num, self.n_dim))
        self.meta_node_radius = nn.Parameter(torch.zeros(self.node_init_num))

    def get_node_radius(self):
        node_radius = F.sigmoid(self.meta_node_radius) * self.node_radius_cap+0.1
        return node_radius

    def make_coors(self, shape, img_shape):
        """Make a coordinate tensor."""

        n_dim = len(shape)

        vectors = [torch.linspace(0, 1, size + 1)[:-1] + 1 / size for size in shape]
        coords = torch.meshgrid(vectors, indexing="ij")
        coords = torch.stack(coords, dim=-1)
        for i in range(len(shape)):
            coords[..., i] = coords[..., i] * img_shape[0]
        coords = coords.view(-1, n_dim)

        radius = 0
        for i in range(n_dim):
            radius += (img_shape[i] / shape[i]) ** 2
        radius = math.sqrt(radius)

        return coords, radius

    def cal_nn_weight(self, x, nodes=None, K=None):
        '''Compute K-NN weights for each coords
        x: [M, 3]
        '''
        K = self.K if K is None else K
        node_radius = self.get_node_radius()

        _, nn_idxs, _ = ops.knn_points(x[None], nodes[None], None, None, K=K) 
        nn_idxs = nn_idxs[0]  # both [M, K]

        relative_p = x[:, None] - nodes[nn_idxs]  # [M, K, 3]
        nn_dist =  torch.sum(relative_p ** 2, dim=-1)
        nn_radius = node_radius[nn_idxs]  # [M, K]
        nn_weight = torch.exp(- nn_dist / (2 * nn_radius ** 2))  # [M, K] 
        # nn_weight = nn_weight / (torch.abs(nn_radius) + 1e-7)
        nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # [M, K]
        return nn_weight, nn_idxs

    def forward(self, x, return_reshaped=False):
        nn_weight, nn_index = self.cal_nn_weight(x, self.node_position, K=self.K)
        acc_flow = nn_weight[..., None] * self.translation[nn_index]
        acc_flow = acc_flow.sum(dim=1)

        if return_reshaped:
            acc_flow = acc_flow.permute(1,0).view(1,self.n_dim,*self.img_shape)

        return acc_flow