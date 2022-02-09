import imp
from pollination_dsl.dag import Inputs, DAG, task, Outputs
from dataclasses import dataclass
from pollination.honeybee_radiance.translate import CreateRadianceFolderGrid
from pollination.honeybee_radiance.grid import SplitGridFolder
from pollination.honeybee_radiance.octree import CreateOctree, CreateOctrees, \
    CreateOctreeWithSky
from pollination.honeybee_radiance.sky import CreateSkyDome, CreateSkyMatrix
from pollination.honeybee_radiance.sun import CreateSunMatrix, ParseSunUpHours
from pollination.honeybee_radiance.multiphase import DaylightMatrixGrouping, \
    MultiPhaseCombinations

from ._view_matrix import ViewMatrixRayTracing
from ._daylight_matrix import DaylightMtxRayTracing
from ._multiply_matrix import MultiplyMatrixDag

from .two_phase.entry import TwoPhaseEntryPoint

@dataclass
class ThreePhaseEntryPoint(DAG):
    """Three phase entry point."""

    # inputs
    north = Inputs.float(
        default=0,
        description='A number for rotation from north.',
        spec={'type': 'number', 'minimum': 0, 'maximum': 360}
    )

    cpu_count = Inputs.int(
        default=50,
        description='The maximum number of CPUs for parallel execution. This will be '
        'used to determine the number of sensors run by each worker.',
        spec={'type': 'integer', 'minimum': 1}
    )

    min_sensor_count = Inputs.int(
        description='The minimum number of sensors in each sensor grid after '
        'redistributing the sensors based on cpu_count. This value takes '
        'precedence over the cpu_count and can be used to ensure that '
        'the parallelization does not result in generating unnecessarily small '
        'sensor grids. The default value is set to 1, which means that the '
        'cpu_count is always respected.', default=50,
        spec={'type': 'integer', 'minimum': 1}
    )

    radiance_parameters = Inputs.str(
        description='The radiance parameters for ray tracing.',
        default='-ab 1 -ad 100 -lw 0.01'
    )

    grid_filter = Inputs.str(
        description='Text for a grid identifier or a pattern to filter the sensor grids '
        'of the model that are simulated. For instance, first_floor_* will simulate '
        'only the sensor grids that have an identifier that starts with '
        'first_floor_. By default, all grids in the model will be simulated.',
        default='*'
    )

    model = Inputs.file(
        description='A Honeybee model in HBJSON file format.',
        extensions=['json', 'hbjson']
    )

    wea = Inputs.file(
        description='Wea file.',
        extensions=['wea']
    )

    @task(template=CreateRadianceFolderGrid)
    def _create_rad_folder(self, input_model=model, grid_filter=grid_filter):
        """Translate the input model to a radiance folder."""
        return [
            {
                'from': CreateRadianceFolderGrid()._outputs.model_folder,
                'to': 'model'
            },
            {
                'from': CreateRadianceFolderGrid()._outputs.bsdf_folder,
                'to': 'model/bsdf'
            },
            {
                'from': CreateRadianceFolderGrid()._outputs.sensor_grids,
                'description': 'Sensor grids information.'
            },
            {
                'from': CreateRadianceFolderGrid()._outputs.model_sensor_grids_file,
                'to': 'results/2_phase/_model_grids_info.json'
            },
            {
                'from': CreateRadianceFolderGrid()._outputs.sensor_grids_file,
                'to': 'results/2_phase/_info.json'
            },
            {
                'from': CreateRadianceFolderGrid()._outputs.receivers,
                'description': 'Receiver apertures information.'
            }
        ]

    # generate sun and skies
    @task(template=CreateSunMatrix)
    def generate_sunpath(self, north=north, wea=wea):
        """Create sunpath for sun-up-hours."""
        return [
            {'from': CreateSunMatrix()._outputs.sunpath,
             'to': 'resources/sky_vectors/sunpath.mtx'},
            {
                'from': CreateSunMatrix()._outputs.sun_modifiers,
                'to': 'resources/sky_vectors/suns.mod'
            }
        ]

    @task(template=ParseSunUpHours, needs=[generate_sunpath])
    def parse_sun_up_hours(self, sun_modifiers=generate_sunpath._outputs.sun_modifiers):
        return [
            {
                'from': ParseSunUpHours()._outputs.sun_up_hours,
                'to': 'resources/sky_vectors/sun-up-hours.txt'
            }
        ]

    @task(template=CreateSkyDome)
    def create_sky_dome(self):
        """Create sky dome for daylight coefficient studies."""
        return [
            {
                'from': CreateSkyDome()._outputs.sky_dome,
                'to': 'resources/sky_vectors/sky_dome.rad'
            }
        ]

    @task(template=CreateSkyMatrix)
    def create_total_sky(self, north=north, wea=wea, sun_up_hours='sun-up-hours'):
        return [
            {
                'from': CreateSkyMatrix()._outputs.sky_matrix,
                'to': 'resources/sky_vectors/sky.mtx'
            }
        ]

    @task(template=CreateSkyMatrix)
    def create_direct_sky(
        self, north=north, wea=wea, sky_type='sun-only', sun_up_hours='sun-up-hours'
    ):
        return [
            {
                'from': CreateSkyMatrix()._outputs.sky_matrix,
                'to': 'resources/sky_vectors/sky_direct.mtx'
            }
        ]

    # split grids
    @task(
        template=SplitGridFolder,
        needs=[_create_rad_folder],
        sub_paths={'input_folder': 'grid'}
    )
    def split_grid_folder(
        self, input_folder=_create_rad_folder._outputs.model_folder,
        cpu_count=cpu_count, cpus_per_grid=3, min_sensor_count=min_sensor_count
    ):
        """Split sensor grid folder based on the number of CPUs"""
        return [
            {
                'from': SplitGridFolder()._outputs.output_folder,
                'to': 'resources/grid'
            },
            {
                'from': SplitGridFolder()._outputs.sensor_grids,
                'description': 'Sensor grids information.'
            }
        ]

    # Create all the octrees as needed
    @task(
        template=CreateOctreeWithSky, needs=[
            generate_sunpath, _create_rad_folder]
    )
    def create_octree_with_suns(
        self, model=_create_rad_folder._outputs.model_folder,
        sky=generate_sunpath._outputs.sunpath
    ):
        """Create octree from radiance folder and sunpath for direct studies."""
        return [
            {
                'from': CreateOctreeWithSky()._outputs.scene_file,
                'to': 'resources/octrees/scene_with_suns.oct'
            }
        ]

    @task(template=CreateOctree, needs=[_create_rad_folder])
    def create_octree(
        self, model=_create_rad_folder._outputs.model_folder,
        include_aperture='exclude', black_out='default'
    ):
        """Create octree from radiance folder."""
        return [
            {
                'from': CreateOctree()._outputs.scene_file,
                'to': 'resources/octrees/main.oct'
            }
        ]

    @task(template=CreateOctrees, needs=[_create_rad_folder, generate_sunpath])
    def create_octrees(
        self, model=_create_rad_folder._outputs.model_folder,
        sunpath=generate_sunpath._outputs.sunpath
    ):
        """Create octrees from radiance folder."""
        return [
            {
                'from': CreateOctrees()._outputs.scene_folder,
                'to': 'resources/octrees/2_phase'
            },
            {
                'from': CreateOctrees()._outputs.scene_info
            }
        ]

    @task(
        template=TwoPhaseEntryPoint,
        loop=create_octrees._outputs.scene_info,
        needs=[
            _create_rad_folder, create_octrees, split_grid_folder,
            create_total_sky, create_direct_sky, create_sky_dome,
            generate_sunpath
        ],
        sub_folder='calcs/2_phase/{{item.identifier}}',
        sub_paths={
            'octree': '{{item.octree}}',
            'octree_direct': '{{item.octree_direct}}',
            'octree_direct_sun': '{{item.octree_direct_sun}}'
        }
    )
    def calculate_two_phase_matrix(
        self,
        identifier='{{item.identifier}}',
        radiance_parameters=radiance_parameters,
        sensor_grids_info=split_grid_folder._outputs.sensor_grids,
        sensor_grids_folder=split_grid_folder._outputs.output_folder,
        octree=create_octrees._outputs.scene_folder,
        octree_direct=create_octrees._outputs.scene_folder,
        octree_direct_sun=create_octrees._outputs.scene_folder,
        sky_dome=create_sky_dome._outputs.sky_dome,
        total_sky=create_total_sky._outputs.sky_matrix,
        direct_sky=create_direct_sky._outputs.sky_matrix,
        sun_modifiers=generate_sunpath._outputs.sun_modifiers,
        bsdf_folder=_create_rad_folder._outputs.bsdf_folder,
        results_folder='../../../results/2_phase'
    ):
        pass

    @task(
        template=ViewMatrixRayTracing,
        needs=[_create_rad_folder, create_octree],
        loop=_create_rad_folder._outputs.receivers,
        sub_folder='calcs/3_phase/view_mtx',
        sub_paths={
            'sensor_grid': 'grid/{{item.identifier}}.pts',
            'receiver_file': 'receiver/{{item.path}}',
            'receivers_folder': 'aperture_group'
        }
    )
    def calculate_view_matrix(
        self,
        radiance_parameters=radiance_parameters,
        sensor_count='{{item.count}}',
        receiver_file=_create_rad_folder._outputs.model_folder,
        sensor_grid=_create_rad_folder._outputs.model_folder,
        scene_file=create_octree._outputs.scene_file,
        receivers_folder=_create_rad_folder._outputs.model_folder,
        bsdf_folder=_create_rad_folder._outputs.bsdf_folder,
        fixed_radiance_parameters='-aa 0.0 -I -c 1 -o vmtx/{{item.identifier}}..%%s.vtmx',
    ):
        pass

    @task(
        template=DaylightMatrixGrouping,
        needs=[_create_rad_folder, create_octree, create_sky_dome]
    )
    def group_apertures(
        self,
        model_folder=_create_rad_folder._outputs.model_folder,
        scene_file=create_octree._outputs.scene_file,
        sky_dome=create_sky_dome._outputs.sky_dome
    ):
        return [
            {
                'from': DaylightMatrixGrouping()._outputs.grouped_apertures_folder,
                'to': 'model/sender'
            },
            {
                'from': DaylightMatrixGrouping()._outputs.grouped_apertures
            }
        ]

    @task(
        template=DaylightMtxRayTracing,
        sub_folder='calcs/3_phase/daylight_mtx',
        loop=group_apertures._outputs.grouped_apertures,
        needs=[_create_rad_folder, create_octree,
               create_sky_dome, group_apertures],
        sub_paths={
            'sender_file': '{{item.identifier}}.rad',
            'senders_folder': 'aperture_group'
        }
    )
    def daylight_mtx_calculation(
        self,
        name='{{item.identifier}}',
        radiance_parameters=radiance_parameters,
        receiver_file=create_sky_dome._outputs.sky_dome,
        sender_file=group_apertures._outputs.grouped_apertures_folder,
        senders_folder=_create_rad_folder._outputs.model_folder,
        scene_file=create_octree._outputs.scene_file,
        bsdf_folder=_create_rad_folder._outputs.bsdf_folder
    ):
        pass

    @task(
        template=MultiPhaseCombinations,
        needs=[group_apertures, _create_rad_folder],
        sub_paths={
            'sender_info': '_info.json',
            'states_info': 'aperture_group/states.json',
            'receiver_info': 'receiver/_info.json'
        }
    )
    def get_three_phase_combinations(
        self,
        sender_info=group_apertures._outputs.grouped_apertures_folder,
        receiver_info=_create_rad_folder._outputs.model_folder,
        states_info=_create_rad_folder._outputs.model_folder
    ):
        return [
            {
                'from': MultiPhaseCombinations()._outputs.results_mapper,
                'to': 'results/3_phase/_info.json'
            },
            {
                'from': MultiPhaseCombinations()._outputs.multiplication_info
            }
        ]

    # multiply all the matrices for all the states
    @task(
        template=MultiplyMatrixDag,
        sub_folder='results/3_phase',
        loop=get_three_phase_combinations._outputs.multiplication_info,
        needs=[get_three_phase_combinations, daylight_mtx_calculation, create_total_sky],
        sub_paths={
            'view_matrix': '{{item.vmtx}}',
            't_matrix': '{{item.tmtx}}',
            'daylight_matrix': '{{item.dmtx}}'
        }
    )
    def multiply_matrix(
        self,
        identifier='{{item.identifier}}',
        sky_vector=create_total_sky._outputs.sky_matrix,
        view_matrix='calcs/3_phase/view_mtx',
        t_matrix='model/bsdf',
        daylight_matrix='calcs/3_phase/daylight_mtx'
    ):
        pass


results = Outputs.folder(source='results')
