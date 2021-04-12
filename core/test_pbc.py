import numpy as np
from vaspvis.utils import generate_slab
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.inputs import Poscar

slab = generate_slab(
    bulk='./poscars/POSCAR_InAs_conv',
    miller_index=[2,1,1],
    layers=2,
    passivate=False,
    vacuum=10,
    write_file=True,
)

frac_coords = np.round(slab.frac_coords, 6)
species = np.array(slab.species, dtype=str)

for _ in range(2):
    x0_ind = np.where(np.isclose(frac_coords[:,0], 0))
    x1_ind = np.where(np.isclose(frac_coords[:,0], 1))
    y0_ind = np.where(np.isclose(frac_coords[:,1], 0))
    y1_ind = np.where(np.isclose(frac_coords[:,1], 1))

    x0_coords = frac_coords[x0_ind]
    x1_coords = frac_coords[x1_ind]
    y0_coords = frac_coords[y0_ind]
    y1_coords = frac_coords[y1_ind]

    x0_species = species[x0_ind[0]]
    x1_species = species[x1_ind[0]]
    y0_species = species[y0_ind[0]]
    y1_species = species[y1_ind[0]]


    if len(x0_coords) != 0: x0_coords[:,0] = 1
    if len(x1_coords) != 0: x1_coords[:,0] = 0
    if len(y0_coords) != 0: y0_coords[:,1] = 1
    if len(y1_coords) != 0: y1_coords[:,1] = 0

    new_coords = np.vstack(
        [i for i in [x0_coords, x1_coords, y0_coords, y1_coords] if len(i) != 0]
    )

    new_species = np.concatenate(
        [i for i in [x0_species, x1_species, y0_species, y1_species] if len(i) != 0]
    )

    frac_coords = np.vstack([frac_coords, new_coords])
    species = np.concatenate([species, new_species])

pbc_coords, inds = np.unique(frac_coords, axis=0, return_index=True)
pbc_species = species[inds]

s = Structure(
    lattice=slab.lattice,
    coords=pbc_coords,
    species=pbc_species,
    coords_are_cartesian=False,
    to_unit_cell=False,
)
Poscar(s.get_sorted_structure()).write_file('POSCAR_PBC')
