from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.inputs import Poscar
import itertools
import spglib
import numpy as np


structure = Structure.from_file('./test/POSCAR_1')
positions = structure.frac_coords
sp = structure.site_properties
species = [site.species for site in structure]
site_data = species
unique_species = []
numbers = []

for species, g in itertools.groupby(site_data):
    if species in unique_species:
        ind = unique_species.index(species)
        numbers.extend([ind + 1] * len(tuple(g)))
    else:
        unique_species.append(species)
        numbers.extend([len(unique_species)] * len(tuple(g)))

unique_species_dict = {i+1: unique_species[i] for i in range(len(unique_species))}

cell = (structure.lattice.matrix, positions, numbers)

lattice, scale_pos, atom_num = spglib.standardize_cell(
    cell, to_primitive=False, no_idealize=False, symprec=1e-5
)

print(np.round(lattice, 5))

spg_struct = (lattice, scale_pos, atom_num)

#  lattice, scale_pos, atom_num = spglib.niggli_reduce(
    #  spg_struct
#  )

lattice = spglib.delaunay_reduce(
    lattice
)

print(np.round(lattice, 5))

new_struc = Structure(
    lattice=Lattice(lattice),
    species=[unique_species_dict[i] for i in atom_num],
    coords=scale_pos,
    to_unit_cell=True,
)

Poscar(new_struc).write_file('POSCAR_spglib')

