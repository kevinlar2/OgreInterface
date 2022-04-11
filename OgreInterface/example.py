from generate import InterfaceGenerator, SurfaceGenerator
from miller import MillerSearch
from vaspvis.utils import passivator
from pymatgen.io.vasp.inputs import Poscar


# A Miller index scan can be performed as follows
ms = MillerSearch(
    substrate='./poscars/POSCAR_InAs_conv',
    film='./poscars/POSCAR_Al_conv',
    max_film_index=2,
    max_substrate_index=2,
    length_tol=0.01,
    angle_tol=0.01,
    area_tol=0.01,
    max_area=500,
)
ms.run_scan()
ms.plot_misfits(figsize=(6.5,5), fontsize=17, labelrotation=0)

# Generate a list of InAs(111) Surface classes with different terminations
subs = SurfaceGenerator.from_file(
    './poscars/POSCAR_InAs_conv',
    miller_index=[1,1,1],
    layers=5,
    vacuum=10,
)

# Generate a list of Al(111) Surface classes with different terminations
films = SurfaceGenerator.from_file(
    './poscars/POSCAR_Al_conv',
    miller_index=[1,1,1],
    layers=5,
    vacuum=10,
)

# Select which Surface you want from the list of Surface's
sub = subs.slabs[2]
film = films.slabs[0]

# Optional way to remove layers from the substrate or film (Usefull for complete control of termination)
#  sub.remove_layers(num_layers=3, top=False)
#  film.remove_layers(num_layers=1, top=True)

# Set up the interface generator
interface_generator = InterfaceGenerator(
    substrate=sub,
    film=film,
    length_tol=0.01,
    angle_tol=0.01,
    area_tol=0.01,
    max_area=400,
    interfacial_distance=2.2,
    vacuum=40,
)

# Generate the interfaces
interfaces = interface_generator.generate_interfaces()

# Loop through interfaces to do whatever you want with them
for i, interface in enumerate(interfaces):

    # You can run surface matching
    interface.run_surface_matching(
        scan_size=4,
        output=f'PES_{i}.png',
        grid_density_x=250,
        grid_density_y=250,
        fontsize=20,
    )

    # You can passivate the interface with hydrogens
    passivated_interface = passivator(
        struc=interface.interface,
        bot=True,
        top=False,
        symmetrize=False,
        # passivated_struc='CONTCAR' # Uncomment this line if there is a CONTCAR available with relaxed hydrogens
    )
    
    # Write the unpassivated structure
    Poscar(interface.interface).write_file(f'POSCAR_{i}')

    # Write the passivated structure
    Poscar(passivated_interface).write_file(f'POSCAR_pas_{i}')

