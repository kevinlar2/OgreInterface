from generate import InterfaceGenerator, SurfaceGenerator
from vaspvis.utils import passivator
from pymatgen.io.vasp.inputs import Poscar

sub_layer = 5
film_layer = 5

subs = SurfaceGenerator.from_file(
    './poscars/POSCAR_InSb_conv',
    miller_index=[1,0,0],
    layers=sub_layer,
    vacuum=5,
)

films = SurfaceGenerator.from_file(
    './poscars/POSCAR_EuS_conv',
    miller_index=[1,0,0],
    layers=film_layer,
    vacuum=5,
)

# You can remove selected layers from top or bottom of each slab if you want
#  subs.slabs[3].remove_layers(num_layers=5, top=False)
#  films.slabs[0].remove_layers(num_layers=1, top=True)

inter = InterfaceGenerator(
    substrate=subs.slabs[1],
    film=films.slabs[0],
    length_tol=0.06,
    angle_tol=0.06,
    area_tol=0.06,
    max_area=400,
    interfacial_distance=2.2,
    vacuum=50,
)

interfaces = inter.generate_interfaces()

for i, interface in enumerate(interfaces):
    # Passivation option below
    pas = passivator(
        struc=interface.interface,
        bot=True,
        top=True,
        symmetrize=False,
        #  passivated_struc='./test/Al-InAs111A/CONTCAR'
    )
    Poscar(pas).write_file(f'./test/EuS_InSb/POSCAR_{i}')

    # Not passivated option below
    #  Poscar(interface.interface).write_file(f'./test/aSn_InSb/POSCAR_{i}')
