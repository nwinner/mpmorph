from fireworks import Workflow, Firework
from atomate.vasp.fireworks.core import MDFW, OptimizeFW, StaticFW
from mpmorph.workflow.mdtasks import SpawnMDFWTask, CopyCalsHome
from mpmorph.runners.amorphous_maker import AmorphousMaker
from mpmorph.analysis.structural_analysis import get_sample_structures
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Molecule

from atomate.lammps.workflows.core import PackmolFW, LammpsFW, get_packmol_wf
from atomate.lammps.firetasks.run_calc import RunPackmol
from fireworks.user_objects.firetasks.script_task import ScriptTask
from atomate.common.firetasks.glue_tasks import CopyFilesFromCalcLoc, PassCalcLocs
from pymatgen.core.structure import Structure

from mpmorph.workflow.mdtasks import AmorphousMakerTask, LammpsToVaspMD, MDAnalysisTask, PackToLammps

import os


def get_wf_density(structure, temperature, pressure_threshold=5.0, max_rescales=6, nsteps=2000, wall_time=19200,
                   vasp_input_set=None, vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<", name="density_finder",
                   optional_MDWF_params=None, override_default_vasp_params=None,
                   amorphous_maker_params=None, copy_calcs=False, calc_home="~/wflows"):
    """
    :param structure: (
    :param temperature:
    :param pressure_threshold:
    :param max_rescales:
    :param nsteps:
    :param wall_time:
    :param vasp_input_set:
    :param vasp_cmd:
    :param db_file:
    :param name:
    :param optional_MDWF_params:
    :param override_default_vasp_params:
    :param amorphous_maker_params:
    :param copy_calcs:
    :param calc_home:
    :return:
    """
    if copy_calcs:
        if not os.path.exists(calc_home):
            raise ValueError("calc_home must be an existing folder.")
        elif os.path.exists(calc_home+"/"+name):
            raise ValueError("WF name already exists, choose a different name.")
        else:
            calc_home = os.path.join(calc_home,name)
            os.mkdir(calc_home)

    # If structure is in fact just a composition, create a random packed Structure!
    if not isinstance(structure, Structure) and isinstance(structure, dict):
        if not amorphous_maker_params:
            raise ValueError("amorphous_maker_params must be defined!")
        glass = AmorphousMaker(structure, **amorphous_maker_params)
        structure = glass.random_packed_structure

    optional_MDWF_params = optional_MDWF_params or {}
    override_default_vasp_params = override_default_vasp_params or {}
    override_default_vasp_params['user_incar_settings'] = override_default_vasp_params.get('user_incar_settings') or {}
    override_default_vasp_params['user_incar_settings'].update({"ISIF": 1, "LWAVE": False})

    fw1 = MDFW(structure=structure, start_temp=temperature, end_temp=temperature, nsteps=nsteps,
               name=name+"run0", vasp_input_set=vasp_input_set, db_file=db_file,
               vasp_cmd=vasp_cmd, wall_time=wall_time, override_default_vasp_params=override_default_vasp_params,
               **optional_MDWF_params)
    t = [CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True, additional_files =["XDATCAR", "OSZICAR", "DOSCAR"])]

    if copy_calcs:
         t.append(CopyCalsHome(calc_home=calc_home, run_name="run0"))
    t.append(SpawnMDFWTask(pressure_threshold=pressure_threshold, max_rescales=max_rescales,
                       wall_time=wall_time, vasp_cmd=vasp_cmd, db_file=db_file,
                       copy_calcs=copy_calcs, calc_home=calc_home,
                       spawn_count=0))

    fw2 = Firework(t, parents=[fw1], name=name+"_initial_spawn")
    return Workflow([fw1, fw2], name=name+"_WF")


def get_wf_structure_sampler(xdatcar_file, n=10, steps_skip_first=1000, vasp_cmd=">>vasp_cmd<<",
                             db_file=">>db_file<<", name="structure_sampler", **kwargs):
    """
    :param xdatcar_file:
    :param n:
    :param steps_skip_first:
    :param vasp_cmd:
    :param db_file:
    :param name:
    :param kwargs:
    :return:
    """
    structures = get_sample_structures(xdatcar_path=xdatcar_file, n=n, steps_skip_first=steps_skip_first)
    wfs = []
    for s in structures:
        fw1=OptimizeFW(s, vasp_cmd=vasp_cmd, db_file=db_file, parents=[], **kwargs)
        fw2=StaticFW(s, vasp_cmd=vasp_cmd, db_file=db_file, parents=[fw1])
        wfs.append(Workflow([fw1,fw2], name=name+str(s.composition.reduced_formula)) )
    return wfs


def get_relax_static_wf(structures, vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<", name="regular_relax", **kwargs):
    """
    :param structures:
    :param vasp_cmd:
    :param db_file:
    :param name:
    :param kwargs:
    :return:
    """
    wfs = []
    for s in structures:
        fw1=OptimizeFW(s, vasp_cmd=vasp_cmd, db_file=db_file, parents=[], **kwargs)
        fw2=StaticFW(s, vasp_cmd=vasp_cmd, db_file=db_file, parents=[fw1])
        wfs.append(Workflow([fw1,fw2], name=name+str(s.composition.reduced_formula)) )
    return wfs


def get_wf_pack_lammps_vasp(pack_input_set = {}, pre_relax_input_set = {}, md_input_set = {},
                              name="MD WF", metadata=None, db_file=None):

    """
    A worflow for performing molecular dynamics with VASP. The real change versus the default one is the option
    to pre-relax the structure. Pre-relaxation performs a random packing of your molecules/atoms into a box,
    runs lammps on that structure, and then takes the resultant structure and gives it to VASP for ab-initio
    molecular dynamics. This technique can greatly accelerate the time needed to converge a VASP MD run by having
    the initial very poor structure handled by a classical force field, which will take you to a "reasonable" first
    guess of a stable structure. All arguments following pre_relax are related to the pre-relaxation stage and
    most of them need to be specified (see get_packmol_wf in atomate.lammps.workflows.core to see which are absolutely
    necessary and which are optional).

    If you perform pre-relaxation, set structure=None and instead give constituent molecules.

    Args:
        structure (Structure): input structure to be run through VASP if no pre-relaxation is needed
        start_temp (int): the starting temperature for the MD run in VASP
        start_temp (int): the ending
        nsteps (int): the number of time steps to perform the VASP MD run
        time_step (int): The time step for the VASP MD run (POTIM in the input file). Default=1
        vasp_cmd (str): command to run (e.g. "vasp_std"). Default=">>vasp_cmd<<"
        vasp_input_set (DictSet): vasp input set. Default=None
        user_kpoints_settings (dict): example: {"grid_density": 7000}. Default=None
        db_file (str): path to file containing the database credentials. Default=None
        metadata (dict): meta data

        pre_relax (bool): Whether or not to pre-relax the structure with packmol and lammps. Default=False
        lammps_input_file (str): path to lammps input or template file. Strongly recommend having a line
                                at the end of the file that simply says "write data final.data" but you can also
                                specify the variable "lammps_final_data" to be the name of the file you wish
                                to write in the lammps_user_settings
        lammps_user_settings (dict): a dictionary with the user settings for the lammps run. These are used on a
                                    template Lammps input file if variables are present in it.
        constituent_molecules ([Molecules]): list of pymatgen Molecule objects
        packing_config ([dict]): list of configuration dictionaries, one for each constituent molecule.
        forcefield (ForceField): pymatgen.io.lammps.forcefield.ForceField object
        final_box_size ([list]): list of list of low and high values for each dimension [[xlow, xhigh], ...]
        topologies ([Topology]): list of Topology objects. If not given, will be set from the
            topology of the constituent molecules.
        ff_site_property (str): the name of the site property used for forcefield mapping
        tolerance (float): packmol tolerance
        filetype (str): packmol i/o file type.
        control_params (dict): packmol control params
        lammps_cmd (string): lammps command to run (skip the input file).
        packmol_cmd (string): path to packmol bin
        dump_filenames ([str]): list of dump file names
        db_file (string): path to the db file.
        name (str): workflow name

    Returns:
        Workflow
    """


    fws     = []
    parents = []

    atom_style = pack_input_set.get('atom_style') or 'charge'

    # -------------------------------------------------------------------- #
    # ----------------------- PACKMOL SECTION ---------------------------- #
    # -------------------------------------------------------------------- #

    composition = pack_input_set.get('composition')
    box_size = pack_input_set.get('box_size')
    tolerance = pack_input_set.get('tol') or 2
    charges = pack_input_set.get('charges') or None

    if isinstance(box_size, float) or isinstance(box_size, int):
        box_size = [(0, box_size), (0, box_size), (0, box_size)]

    inside_box = []
    for i in box_size:
        inside_box.append(i[0])
    for i in box_size:
        inside_box.append(i[1]-tolerance)

    packing_config = []
    molecules = []
    for k, v in composition.items():
        molecules.append(k)
        packing_config.append({'number': v, 'inside box': inside_box})

    pack_fw   = PackmolFW(molecules, packing_config=packing_config, tolerance=2.0, filetype="xyz",
                          copy_to_current_on_exit=True,
                          output_file="packed_mol.xyz", parents=None, name="PackFW", packmol_cmd="packmol")

    t = pack_fw.tasks
    t.append(PackToLammps(atom_style=atom_style, box_size=box_size, charges=charges))

    # -------------------------------------------------------------------- #
    # ----------------------- LAMMPS SECTION ----------------------------- #
    # -------------------------------------------------------------------- #

    lammps_input_set      = pre_relax_input_set.get('lammps_input_set') or {}
    lammps_input_filename = pre_relax_input_set.get('lammps_input_file') or 'in.lammps'
    data_filename         = pre_relax_input_set.get('data_filename') or 'lammps.data'
    lammps_cmd            = pre_relax_input_set.get('lammps_cmd') or ">>lammps_cmd<<"
    lammps_db_file        = pre_relax_input_set.get('db_file') or None
    dump_filename         = pre_relax_input_set.get('dump_filename') or None

    pre_relax_fw = LammpsFW(lammps_input_set=lammps_input_set, input_filename=lammps_input_filename,
                         data_filename=data_filename, lammps_cmd=lammps_cmd,
                         parents=None, name="PreRelaxFW", db_file=lammps_db_file,
                         log_filename="log.lammps", dump_filename=dump_filename)

    t.extend(pre_relax_fw.tasks)

    # -------------------------------------------------------------------- #
    # ----------------------- VASP SECTION ------------------------------- #
    # -------------------------------------------------------------------- #

    md_input_set['atom_style'] = atom_style
    t.append(LammpsToVaspMD(**md_input_set))

    md_fw = Firework(tasks=t, name="MDFW", parents=None)

    fws.append(md_fw)

    # -------------------------------------------------------------------- #
    # ---------------------- ANALYSIS SECTION ---------------------------- #
    # -------------------------------------------------------------------- #

    t = MDAnalysisTask(time_step=md_input_set.get('time_step', 1))

    fws.append(Firework(t, name="AnalysisTask", parents=md_fw))

    wfname = name or "MD-WF"
    wf = Workflow(fws, name=wfname, metadata=metadata)

    return wf







