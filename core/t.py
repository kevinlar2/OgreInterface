import numpy as np
import time

mu1 = np.array([1,2,3])
mu2 = np.array([0,0,0])

s1 = 1.5 * np.eye(3)
s2 = 1.2 * np.eye(3)

s1_norm = np.linalg.norm(s1)
s2_norm = np.linalg.norm(s2)

s1_inv = np.linalg.inv(s1)
s2_inv = np.linalg.inv(s2)

c = np.vstack([mu1, mu2])
r = np.array([1.5, 1.2])

def kl_overlap(coords, r):
    x1 = coords[:,0].reshape(-1,1) @ np.ones((1, len(coords)))
    x2 = x1.T

    y1 = coords[:,1].reshape(-1,1) @ np.ones((1, len(coords)))
    y2 = y1.T

    z1 = coords[:,2].reshape(-1,1) @ np.ones((1, len(coords)))
    z2 = z1.T

    r1 = r.reshape(-1,1) @ np.ones(r.shape[0]).reshape(1,-1)
    r2 = r1.T

    xs = x1 - x2
    ys = y1 - y2
    zs = z1 - z2

    t1 = np.log(r2/r1)
    t2 = 3
    t3 = (xs**2 + ys**2 + zs**2) / r2
    t4 = 3 * (r1 / r2) 

    kl = (t1 - t2 + t3 + t4) / 2

    return kl

def kl_divergence(mu1, mu2, s1, s1_norm, s2_norm, s2_inv):
    kl = 0.5 * (
        np.log(s2_norm/s1_norm) - 3 + \
        ((mu1 - mu2).T @ s2_inv @ (mu1 - mu2)) + \
        np.trace(s2_inv @ s1)
    )

    return kl

print(kl_divergence(mu1, mu2, s1, s1_norm, s2_norm, s2_inv))
print(kl_divergence(mu2, mu1, s2, s2_norm, s1_norm, s1_inv))

print(kl_overlap(c, r))
