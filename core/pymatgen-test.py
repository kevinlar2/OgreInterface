from pymatgen.analysis.substrate_analyzer import ZSLGenerator, SubstrateAnalyzer
from pymatgen.core.structure import Structure
import numpy as np
import time 

sub_structure = Structure.from_file('./POSCAR_InAs_conv')
sub_miller_index = [1,0,0]

film_structure = Structure.from_file('./POSCAR_Al_conv')
film_miller_index = [1,0,0]

ZSL = ZSLGenerator(
    max_area_ratio_tol=0.05,
    max_angle_tol=0.05,
    max_length_tol=0.05,
    max_area=500,
)

SA = SubstrateAnalyzer(zslgen=ZSL)

M = SA.calculate(
    film=film_structure,
    substrate=sub_structure,
    film_millers=[film_miller_index],
    substrate_millers=[sub_miller_index],
)

M_list = list(M)
print(M_list[0].keys())

matrices = np.array([np.c_[i['film_transformation'], i['substrate_transformation']] for i in M])
#  film_matrices = matrices[:,:,[0,1]]
#  sub_matrices = matrices[:,:,[2,3]]
#  print(film_matrices)
#  print(sub_matrices)

