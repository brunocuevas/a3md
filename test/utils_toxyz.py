import torch
from a3mdnet.utils import to_xyz_file

if __name__ == "__main__":

    r = torch.randn(10, 3).unsqueeze(0)
    n = torch.tensor([1, 1, 1, 1, 1, 1, 6, 8, 0, 0]).unsqueeze(0)

    for xyz in to_xyz_file(n, r):
        with open('test.xyz', 'w') as f:
            f.write(xyz)
