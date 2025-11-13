import torch

def soft_hist2d(xy, bins=50,lo=-1.5, hi=1.5, bandwidth=0.06):
    """
    xy: [B,2] tensor of positions (agents as particles)
    returns prob grid [bins,bins] that sums to 1
    """

    device = xy.device
    edges = torch.linspace(lo, hi, bins, device=device)
    Xc, Yc = torch.meshgrid(edges, edges, indexing='ij')

    #gaussian kernels
    diffx = xy[:,0].unsqueeze(1).unsqueeze(2) - Xc
    diffy = xy[:,1].unsqueeze(1).unsqueeze(2) - Yc
    w = torch.exp(-(diffx**2 + diffy**2) / (2*bandwidth**2))

    p = w.sum(dim=0)
    p = p / (p.sum() + 1e-8)

    return p


def entropy_from_grid(p):
    return -(p* (p.clamp_min(1e-8)).log()).sum()