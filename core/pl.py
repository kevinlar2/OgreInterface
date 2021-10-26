import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(4,4), dpi=400)
plt.axis('off')

matrix = np.array([
    [1,-0.5],
    [1,0.5],
])

bx = np.array([0,1,1,0,0])
by = np.array([0,0,1,1,0])

x = np.array([0, 0.25])
y = np.array([0, 0.25])

xs = np.c_[[x-1,x-1,x-1,x,x,x,x+1,x+1,x+1]].ravel()
ys = np.c_[[y-1,y,y+1,y-1,y,y+1,y-1,y,y+1]].ravel()

nb = np.c_[bx, by].dot(matrix)
nc = np.c_[xs, ys].dot(matrix)

bx, by = nb[:,0], nb[:,1]
cx, cy = nc[:,0], nc[:,1]

ax.set_xlim(-2,2)
ax.set_ylim(-2,2)

ax.plot(
    bx,
    by,
    color='black',
)

ax.scatter(
    cx,
    cy,
)

fig.tight_layout()
fig.savefig('test2.png')


