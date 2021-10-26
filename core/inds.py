import numpy as np

inds = np.array([
    [1,0,0],
    [1,-1,0],
    [1,1,-1],
])


xlabels = []
for i in inds:
    neg_inds = i < 0
    i[neg_inds] *= -1
    str_array = i.astype(str)
    str_array = np.array([r'\bar{' + f'{j}' + '}' if n else j for j, n in zip(i, neg_inds)])
    str_inds = f'{str_array}'.replace('[', '(').replace(']', ')').replace("'", "")
    xlabels.append(str_inds)

#  xlabels = [f'{i}'.replace('[', '(').replace(']', ')') for i in self.substrate_inds]
