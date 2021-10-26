from generate import InterfaceGenerator, SurfaceGenerator
from vaspvis.utils import passivator
from vaspvis.utils import get_periodic_vacuum
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sub_layer = 5
film_layer = 5

subs = SurfaceGenerator.from_file(
    './poscars/POSCAR_InSb_conv',
    miller_index=[1,1,0],
    layers=sub_layer,
    vacuum=5,
)

films = SurfaceGenerator.from_file(
    './poscars/POSCAR_aSn_conv',
    miller_index=[1,1,0],
    layers=film_layer,
    vacuum=20,
)

#  subs.slabs[3].remove_layers(num_layers=5)
#  films.slabs[0].remove_layers(num_layers=1, top=True)

inter = InterfaceGenerator(
    substrate=subs.slabs[0],
    film=films.slabs[0],
    length_tol=0.01,
    angle_tol=0.01,
    area_tol=0.01,
    max_area=200,
    interfacial_distance=2.2,
    #  sub_strain_frac=0,
    vacuum=20,
)

Poscar(films.slabs[0].slab_pmg).write_file('./test/aSn_InSb/POSCAR_aSn')

interfaces = inter.generate_interfaces()

for i, interface in enumerate(interfaces):
    #  pas = passivator(
        #  struc=interface.interface,
        #  bot=True,
        #  top=False,
        #  symmetrize=False,
        #  passivated_struc='./test/Al-InAs111A/CONTCAR'
    #  )
    #  coords = pas.frac_coords
    #  shift_val = (coords[:,-1].max() - coords[:,-1].min()) / 2
    #  pas.translate_sites(
        #  indices=range(len(pas)),
        #  vector=[0,0,shift_val],
        #  frac_coords=True,
        #  to_unit_cell=True,
    #  )

    #  reduced_interface_struc = interface_struc.get_reduced_structure()
    #  sg = SpacegroupAnalyzer(reduced_interface_struc)
    #  refined_interface_struc = sg.get_conventional_standard_structure()
    #  primitive_interface_struc = refined_interface_struc.get_reduced_structure()
    #  primitive_interface_struc = primitive_interface_struc.get_primitive_structure()
    aSn_film = get_periodic_vacuum(
        slab=interface.strained_film.get_primitive_structure(),
        bulk='./poscars/POSCAR_aSn_conv',
        vacuum=40,
        miller_index=[1,1,0],
    )
    Poscar(aSn_film).write_file(f'./test/aSn_InSb/POSCAR_{i}')
