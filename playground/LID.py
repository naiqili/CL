import numpy as np
import matplotlib.pyplot as plt
import torch

device = "gpu" if torch.cuda.is_available() else "cpu"


def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX, -1).to(device)
    X2 = (X*X).sum(1).resize_(nX, 1)
    Y = Y.view(nY, -1).to(device)
    Y2 = (Y*Y).sum(1).resize_(nY, 1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def LID_elementwise(x, X, k):
    # only for X: [B, d]
    r = np.power(np.sum(np.power(X - x, 2), axis=-1), 0.5)
    r = np.sort(r, axis=0)
    r_max = r[k-1]
    return -1 / (np.mean(np.log(r[0:k] / r_max)))


def LID(X, Y, k):
    # X, Y: [B, h, w], [B, l], ...
    sum_axis = tuple([i for i in range(2, len(X.shape) + 1)])
    XX = X.reshape(X.shape[0], 1, *X.shape[1:])
    YY = Y.reshape(1, Y.shape[0], *Y.shape[1:])
    dist_mat = np.power(np.sum(np.power(XX - YY, 2), axis=sum_axis), 0.5)
    dist_mat = np.where(dist_mat < 1e-10, 1e10, dist_mat)

    sorted_mat = np.sort(dist_mat, axis=1)
    r_max = sorted_mat[:, k-1].reshape(-1, 1)
    # est = - 1 / np.mean(np.log(sorted_mat[:, 0:k] / (r_max + 1e-10)), axis=1)
    mask = (dist_mat <= r_max).astype(np.float)

    est = -1 / (1 / k * np.sum(np.log(dist_mat) * mask, axis=1, keepdims=True) - np.log(r_max))
    return est.reshape(-1)


def gaussian_dist(x, mean, std):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-np.power((x - mean), 2) / (2 * std ** 2))


def lid(Mxy, k):
    eps_mat = torch.where(Mxy > 1e-20, torch.zeros((1, 1)), torch.ones((1, 1)) * 1e-20).detach()
    Mxy = Mxy + eps_mat
    value, idx = Mxy.topk(k=k, largest=False)
    mask = torch.zeros(Mxy.size()).type(Mxy.type())
    mask.scatter_(1, idx, 1.0)
    r_max = value[:, -1].detach()

    # est = -1 / (1. / k * torch.sum(torch.log(Mxy + eps_mat) * mask, dim=-1) - torch.log(r_max))
    est = -1 / (torch.mean(torch.log(value), dim=-1) - torch.log(r_max))
    return est