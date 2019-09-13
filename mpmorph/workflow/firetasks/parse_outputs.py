import os
import json
import gzip
import numpy as np
from monty.serialization import loadfn, dumpfn
from monty.shutil import gzip_dir, compress_file, decompress_file
from monty.io import zopen

from fireworks import FireTaskBase, Firework, FWAction, explicit_serialize

from pymatgen.io.vasp.outputs import Vasprun, Xdatcar
from pymatgen.analysis.structure_analyzer import VoronoiAnalyzer

from atomate.utils.utils import get_logger, env_chk, load_class
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs, get_calc_loc
from atomate.vasp.database import VaspCalcDb

from mpmorph.analysis.md_data import MD_Data
from mpmorph.analysis.structural_analysis import RadialDistributionFunction, \
    BondAngleDistribution, CageCorrelationFunction, VoronoiAnalysis
from mpmorph.analysis.transport import VDOS, Viscosity, Diffusion
from mpmorph.runners.rescale_volume import RescaleVolume

__authors__ = 'Nicholas Winner, Muratahan Aykol'

logger = get_logger(__name__)


@explicit_serialize
class MDAnalysisTask(FireTaskBase):
    required_params = []
    optional_params = ['time_step', 'get_rdf', 'get_diffusion', 'get_viscosity',
                       'get_vdos', 'get_run_data', 'checkpoint_dirs', 'analysis_spec']

    def run_task(self, fw_spec):

        get_rdf = self.get('get_rdf') or True
        get_diffusion = self.get('get_diffusion') or True
        get_viscosity = self.get('get_viscosity') or True
        get_vdos = self.get('get_vdos') or True
        get_run_data = self.get('get_run_data') or True
        time_step = self.get('time_step') or 2
        checkpoint_dirs = fw_spec.get('checkpoint_dirs', False)

        calc_dir = get_calc_loc(True, fw_spec["calc_locs"])["path"]
        calc_loc = os.path.join(calc_dir, 'XDATCAR.gz')

        ionic_step_skip = self.get('ionic_step_skip') or 1
        ionic_step_offset = self.get('ionic_step_offset') or 0

        analysis_spec = self.get('analysis_spec') or {}

        logger.info("Reading in ionic_steps...")

        decompress_file("ionic_steps.json.gz")
        ionic_steps = loadfn("ionic_steps.json")
        structures = [s.structure for s in ionic_steps]
        compress_file("ionic_steps.json")

        db_dict = {}
        db_dict.update({'density': float(structures[0].density)})
        db_dict.update(structures[0].composition.to_data_dict)

        if get_rdf:
            logger.info("LOGGER: Calculating radial distribution functions...")
            rdf = RadialDistributionFunction(structures=structures)
            rdf_dat = rdf.get_radial_distribution_functions(nproc=4)
            db_dict.update({'rdf': rdf.get_rdf_db_dict()})
            del rdf
            del rdf_dat

        if get_vdos:
            logger.info("LOGGER: Calculating vibrational density of states...")
            vdos = VDOS(structures)
            vdos_dat = vdos.calc_vdos_spectrum(time_step=time_step*ionic_step_skip)
            vdos_diff = vdos.calc_diffusion_coefficient(time_step=time_step*ionic_step_skip)
            db_dict.update({'vdos': vdos_dat})
            del vdos
            del vdos_dat

        if get_diffusion:
            logger.info("LOGGER: Calculating the diffusion coefficients...")
            diffusion = Diffusion(structures, t_step=time_step, l_lim=50, skip_first=250, block_l=1000, ci=0.95)
            D = {'msd':{}, 'vdos':{}}
            for s in structures[0].types_of_specie:
                D['msd'][s.symbol] = diffusion.getD(s.symbol)
            if vdos_diff:
                D['vdos'] = vdos_diff
            db_dict.update({'diffusion': D})
            del D

        if get_viscosity:
            logger.info("LOGGER: Calculating the viscosity...")
            viscosities = []
            if checkpoint_dirs:
                for dir in checkpoint_dirs:
                    visc = Viscosity(dir).calc_viscosity()
                    viscosities.append(visc['viscosity'])
            viscosity_dat = {'viscosity': np.mean(viscosities), 'StdDev': np.std(viscosities)}
            db_dict.update({'viscosity': viscosity_dat})
            del viscosity_dat

        if get_run_data:
            if checkpoint_dirs:
                logger.info("LOGGER: Assimilating run stats...")
                data = MD_Data()
                for directory in checkpoint_dirs:
                    data.parse_md_data(directory)
                md_stats = data.get_md_stats()
            else:
                logger.info("LOGGER: Getting run stats...")
                data = MD_Data()
                data.parse_md_data(calc_dir)
                md_stats = data.get_md_stats()
            db_dict.update({'md_data': md_stats})

        if analysis_spec:
            logger.info("LOGGER: Adding user-specified data...")
            db_dict.update(analysis_spec)

        logger.info("LOGGER: Pushing data to database collection...")
        db_file = env_chk(">>db_file<<", fw_spec)
        db = VaspCalcDb.from_db_file(db_file, admin=True)
        db.collection = db.db["md_data"]
        db.collection.insert_one(db_dict)

        return FWAction()


@explicit_serialize
class MDStats(FireTaskBase):

    required_params = []
    optional_params = ['checkpoint_dirs']

    def run_task(self, fw_spec):

        checkpoint_dirs = self.get("checkpoint_dirs", False)
        calc_dir = get_calc_loc(True, fw_spec["calc_locs"])["path"]

        if checkpoint_dirs:
            logger.info("LOGGER: Assimilating run stats...")
            data = MD_Data()
            for directory in checkpoint_dirs:
                data.parse_md_data(directory)
            md_stats = data.get_md_stats()
        else:
            logger.info("LOGGER: Getting run stats...")
            data = MD_Data()
            data.parse_md_data(calc_dir)
            md_stats = data.get_md_stats()

        os.mkdir(os.path.join(calc_dir, 'md_stats'))
        dumpfn(md_stats, os.path.join(calc_dir, 'md_stats', 'md_stats.json'))


@explicit_serialize
class VaspMdToDbTask(FireTaskBase):
    pass


@explicit_serialize
class VaspMdToDiffusion(FireTaskBase):

    required_params = ['time_step']
    optional_params = ['checkpoint_dirs', 'output_file']

    def run_task(self, fw_spec):

        time_step = fw_spec.get('time_step', 2)
        checkpoint_dirs = fw_spec.get('checkpoint_dirs', False)
        output_file = fw_spec.get('output_file', 'diffusion.json')

        calc_dir = get_calc_loc(True, fw_spec["calc_locs"])["path"]
        calc_loc = os.path.join(calc_dir, 'XDATCAR.gz')

        if checkpoint_dirs:
            logger.info("LOGGER: Assimilating checkpoint structures")
            structures = []
            for d in checkpoint_dirs:
                structures.extend(Vasprun(os.path.join(d, 'vasprun.xml.gz')).structures)
        else:
            structures = Xdatcar(calc_loc).structures

        db_dict = {}
        db_dict.update({'density': float(structures[0].density)})
        db_dict.update(structures[0].composition.to_data_dict)

        logger.info("LOGGER: Calculating the diffusion coefficients...")
        diffusion = Diffusion(structures, t_step=time_step, l_lim=50, skip_first=250, block_l=1000, ci=0.95)

        vdos = VDOS(structures)
        vdos_dat = vdos.calc_vdos_spectrum(time_step=time_step)
        vdos_diff = vdos.calc_diffusion_coefficient(time_step=time_step)

        D = {'msd': {}, 'vdos': {}}
        for s in structures[0].types_of_specie:
            D['msd'][s.symbol] = diffusion.getD(s.symbol)
            D['vdos'] = vdos_diff

        db_dict.update({'diffusion': D})

        with open(os.path.join(output_file)) as f:
            jsn = json.dumps(db_dict)
            f.write(jsn)

        return FWAction()


@explicit_serialize
class StructuralAnalysis(FireTaskBase):
    """
    This task uses lists of ionic_steps to perform several
    common structural analysis schemes and write the results.

    Analysis currently supported:
        (1) Radial distribution functions
                -including first shell coordination numbers
        (2) Cage correlation functions
                -requires RDFs be calculated
        (3) Potential of Mean Force
                -not structural, but obtained from g(r)
                -requires RDFs be calculated
        (4) Bond Angle distribution functions
        (5) Voronoi Analysis
                -be careful, voronoi analysis can take a long time and
                 be memory intensive for large trajectories
        (6) Polyhedra connectivity distribution functions
                -connection motifs of the voronoi tetrahedra (second-nn correlations)

    required_params:
        data_dir: (str) path to the directory containing the ionic_steps file(s). If a single MD
                    run was performed, this is likely {$CURRENT_DIR}/analysis. If checkpointing
                    or multiple runs were performed, then it is likely the current directory, which
                    is the directory of the analysis firework

    optional_params (all default to True):
        calc_rdf: (bool) calculate the radial distribution functions.
        calc_bad: (bool) calculate the bond angle distribution functions.
        calc_voronoi: (bool) calculate the voronoi polyhedra distribution.
        calc_cage: (bool) calculate the cage correlation functions.
        calc_pmf: (bool) calculate the potential of mean force from g(r).
        calc_connectivity: (bool) calculate g(r) decomposed by the second NN connectivity motifs.

    """
    required_params = ['data_dir']
    optional_params = ['calc_rdf', 'calc_bad', 'calc_voronoi', 'calc_cage', 'calc_pmf', 'calc_connectivity']

    def run_task(self, fw_spec):

        # Get
        data_dir = self.get('data_dir')
        calc_rdf = self.get('calc_rdf', True)
        calc_bad = self.get('calc_bad', True)
        calc_voronoi = self.get('calc_voronoi', False)
        calc_cage = self.get('calc_cage', True)
        calc_pmf = self.get('calc_pmf', False)
        calc_connectivity = self.get('calc_connectivity', False)

        ionic_steps = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if 'ionic_steps' in f:
                    name, ext = os.path.splitext(f)
                    if ext in ('.gz', '.GZ', '.Z'):
                        with gzip.open(f, "rb") as gzipped:
                            d = json.loads(gzipped.read().decode("ascii"))
                    else:
                            d = loadfn(f)
                    ionic_steps.extend(d)

        structures = [step['structure'] for step in ionic_steps]

        data_dict = {}

        if calc_rdf:
            logger.info("LOGGER: Calculating radial distribution functions...")
            rdf = RadialDistributionFunction(structures=structures)
            rdf.get_radial_distribution_functions(nproc=4)
            cns = rdf.get_coordination_numbers()
            fs = rdf.first_coordination_shell_radius
            data_dict.update({'radial_distribution_functions': rdf.as_dict()})
            data_dict.update({'coordination_numbers': cns})

            if calc_cage:
                logger.info("LOGGER: Calculating cage correlation function...")
                ccf = CageCorrelationFunction(structures, fs)
                ccf.get_cage_correlation_function()
                # TODO: Make sure the CCFs work

            if calc_pmf:
                logger.info("LOGGER: Calculating the potential of mean force...")
                # TODO: Need to include the implementation of PMF here

        if calc_bad:
            logger.info("LOGGER: Calculating bond angle distribution functions...")
            bad = BondAngleDistribution(structures=structures)
            bad.get_bond_angle_distribution(nproc=4)
            data_dict.update({'bond_angle_distribution_functions': bad.as_dict()})

        if calc_voronoi:
            logger.info("LOGGER: Performing voronoi analysis...")
            va = VoronoiAnalyzer(structures)
            try:
                poly = va.analyze_structures()
                data_dict.update({'voronoi_polyhedra': poly})
            except MemoryError:
                logger.info("ERROR: Voronoi analysis failed due to insufficient memory...")

        if calc_connectivity:
            logger.info("LOGGER: Getting the connectivity motif distribution functions...")
            # TODO: Implement after writing connectivity function

        # write structural analysis results to json file and then zip it
        write_dir = os.path.join(os.getcwd(), 'structural_analysis')
        os.mkdir(write_dir)
        for k,v in data_dict.items():
            dumpfn(v, os.path.join(write_dir, '{}.json').format(k))
        gzip_dir(write_dir)

        return FWAction()


@explicit_serialize
class ParseSingleTask(FireTaskBase):
    """
    In contrast to ParseCheckpointsTask, this task is run at the end of a single MD simulation
    with no checkpointing. It writes the ionic_steps.json file to a folder called "analysis"
    in order to keep the functionality of the other analysis tasks more seamless.

    required_params:
        (none)

    optional_params:
        vasprun: (str) The path of the vasprun file to parse
                    Default: {$CURRENT_DIRECTORY}/vasprun.xml.gz
        write_dir: (str) The path of the directory in which to write the ionic_steps.json file
                    Default: {$CURRENT_DIRECTORY}/analysis
        filename: (str) The name of the file to which to write the ionic_steps data
                    Default: ionic_steps.json (recommended to keep this)
    """

    required_params = []
    optional_params = ['vasprun', 'write_dir', 'filename']

    def run_task(self, fw_spec):
        filepath = self.get('vasprun', os.path.join(os.getcwd(), 'vasprun.xml.gz'))
        write_dir = self.get('write_dir', os.path.join(os.getcwd(), 'analysis'))
        filename = self.get('filename', 'ionic_steps.json')

        os.mkdir(write_dir)

        ionic_steps = Vasprun(filepath).ionic_steps
        dumpfn(ionic_steps, os.path.join(write_dir, filename))
        compress_file(os.path.join(write_dir, filename))

        s = ionic_steps[0]['structure']
        composition = {{'composition': s.composition.to_data_dict}}
        composition.update({'density': float(s.density)})

        dumpfn(composition, os.path.join(write_dir, 'composition.json'))
        compress_file(os.path.join(write_dir, 'composition.json'))


@explicit_serialize
class ParseCheckpointsTask(FireTaskBase):
    """
    This function is used to assimilate an MD workflow into the current directory. For example,
    if you ran 3 separate 9 ps simulations, and those simulations were each split into 9 1-ps
    job executions, then this will assimilate the results into a single location.

    required_params:
        checkpoint_dirs: (dict) collection of directories that you are assimilating. Dictionary
                         format is {'Simulation_1': [array of checkpoint_dirs],
                         'Sumulation_2': [...], ...}

    optional_params:
        write_dir: (str) Name of the directory to be created in which to dump the trajectory files.
                    Default: current working directory

    """
    required_params = []
    optional_params = ['write_dir']

    def run_task(self, fw_spec):
        checkpoint_dirs = self.get('checkpoint_dirs')
        write_dir = self.get('write_dir', False)

        if not write_dir:
            write_dir = os.getcwd()

        # write each md run (comprised of n checkpoints) to a json file and zip it
        logger.info("LOGGER: Assimilating checkpoint data...")
        ionic_steps = []
        for directory in checkpoint_dirs:
            ionic_steps.extend(Vasprun(os.path.join(directory, "vasprun.xml.gz")).ionic_steps)
        dumpfn(ionic_steps, os.path.join(write_dir, 'ionic_steps.json'))
        compress_file(os.path.join(write_dir, 'ionic_steps.json'))

        # get composition info
        s = ionic_steps[0]['structure']
        composition = {{'composition': s.composition.to_data_dict}}
        composition.update({'density': float(s.density)})

        # write composition info to json and zip
        dumpfn(composition, os.path.join(write_dir, 'composition.json'))
        compress_file(os.path.join(write_dir, 'composition.json'))
