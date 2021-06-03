import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

times_AlInAs = [
    4.309343118667602,
    2.462775344848633,
    1.8847497367858888,
    1.6177744388580322,
    1.485772647857666,
    1.4066347312927245,
    #  1.399104766845703,
    #  1.3598015308380127,
    #  1.3584164810180663,
    #  1.345594472885132,
    #  1.3421750354766846,
]
times_FeInSb = [
    1.82687424659729,
    1.0759828472137452,
    0.8219082260131836,
    0.7626700878143311,
    0.7353337669372558,
    0.6976130390167237,
    #  0.7116126251220704,
    #  0.7039773654937744,
    #  0.7491522789001465,
    #  0.7345027351379394,
    #  0.7428419399261474,
]

fig, ax = plt.subplots(figsize=(4,3), dpi=400)
ax.set_xlabel(r"Number of Cores")
ax.set_ylabel(r"Time (s)")

ax.plot(
    np.log2(range(1, len(times_AlInAs) + 1)),
    np.log2(times_AlInAs),
    color='red',
    marker='o',
    label='Al(100)/InAs(100)'
)

ax.plot(
    np.log2(range(1, len(times_FeInSb) + 1)),
    np.log2(times_FeInSb),
    color='blue',
    marker='o',
    label='Fe(100)/InSb(100)'
)

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))

ax.set_xticks([0, 1, 2])
ax.set_xticklabels([1, 2, 4])

ax.set_yticks([0, 1, 2])
ax.set_yticklabels([1, 2, 4])

ax.legend()
fig.tight_layout(pad=0.4)
fig.savefig('scale_plot.png')
