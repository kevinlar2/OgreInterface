from pymatgen.core.structure import Structure
import numpy as np

s1 = Structure.from_file('./Al001-InAs001/POSCAR_0')
s2 = Structure.from_file('./Al-InAs111A/larger/POSCAR_0')

mat1 = s1.lattice.matrix
mat2 = s2.lattice.matrix

area1 = np.linalg.norm(np.cross(mat1[0], mat1[1]))
area2 = np.linalg.norm(np.cross(mat2[0], mat2[1]))

print('Area 1 =', area1)
print('Area 2 =', area2)
