from a3mdnet.utils import DxGrid
import torch


if __name__ == "__main__":

    dxg = DxGrid(device=torch.device('cuda:0'), dtype=torch.float, resolution=0.25, spacing=2.0)
    g, dv, cell_info = dxg.generate_grid(torch.randn(1, 10, 3).to(torch.device('cuda:0')))
    u = (-dv.norm(dim=3)).exp().sum(2).to(torch.device('cuda:0'))
    dx = dxg.dx(u, **cell_info)
    dx.write('test.dx')
    print("--done")
