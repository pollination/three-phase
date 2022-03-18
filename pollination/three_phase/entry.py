from pollination_dsl.dag import Inputs, DAG, task, Outputs
from dataclasses import dataclass
from pollination.honeybee_radiance.translate import CreateRadianceFolderGrid
from pollination.honeybee_radiance.sky import CreateSkyDome, CreateSkyMatrix
from pollination.honeybee_radiance.sun import CreateSunMatrix, ParseSunUpHours
from pollination.honeybee_radiance.multiphase import CreateOctreesGrids

from .two_phase.entry import TwoPhaseEntryPoint
from .three_phase.preparation import ThreePhaseInputsPreparation
from .three_phase.calculation import ThreePhaseMatrixCalculation


@dataclass
class RecipeEntryPoint(DAG):
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

    view_mtx_rad_params = Inputs.str(
        description='The radiance parameters for ray tracing.',
        default='-ab 2 -ad 5000 -lw 2e-05'
    )

    daylight_mtx_rad_params = Inputs.str(
        description='The radiance parameters for ray tracing.',
        default='-ab 2 -ad 5000 -lw 2e-05'
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

    dmtx_group_params = Inputs.str(
        description='A string to change the parameters for aperture grouping for '
        'daylight matrix calculation. Valid keys are -s for aperture grid size, -t for '
        'the threshold that determines if two apertures/aperture groups can be '
        'clustered, and -ad for ambient divisions used in view factor calculation '
        'The default is -s 0.2 -t 0.001 -ad 1000. The order of the keys is not '
        'important and you can include one or all of them. For instance if you only '
        'want to change the aperture grid size to 0.5 you should use -s 0.5 as the '
        'input.', default='-s 0.2 -t 0.001 -ad 1000'
    )

    @task(template=CreateRadianceFolderGrid)
    def create_rad_folder(self, input_model=model, grid_filter=grid_filter):
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
                'to': 'results/sun-up-hours.txt'
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

    @task(template=CreateOctreesGrids, needs=[create_rad_folder, generate_sunpath])
    def create_octrees_grids(
        self, model=create_rad_folder._outputs.model_folder,
        sunpath=generate_sunpath._outputs.sunpath, phase='3', cpu_count=cpu_count,
        cpus_per_grid=3, min_sensor_count=min_sensor_count
    ):
        return [
            {
                'from': CreateOctreesGrids()._outputs.scene_folder,
                'to': 'resources/octrees'
            },
            {
                'from': CreateOctreesGrids()._outputs.grid_folder,
                'to': 'resources/grid'
            },
            {
                'from': CreateOctreesGrids()._outputs.scene_info
            },
            {
                'from': CreateOctreesGrids()._outputs.two_phase_info
            }
        ]

    @task(
        template=TwoPhaseEntryPoint,
        loop=create_octrees_grids._outputs.two_phase_info,
        needs=[
            create_rad_folder, create_octrees_grids,
            create_total_sky, create_direct_sky, create_sky_dome,
            generate_sunpath
        ],
        sub_folder='calcs/2_phase/{{item.identifier}}',
        sub_paths={
            'octree': '{{item.octree}}',
            'octree_direct': '{{item.octree_direct}}',
            'octree_direct_sun': '{{item.octree_direct_sun}}',
            'sensor_grids_folder': '{{item.light_path}}'
        }
    )
    def calculate_two_phase_matrix(
        self,
        identifier='{{item.identifier}}',
        radiance_parameters=radiance_parameters,
        sensor_grids_info='{{item.sensor_grids_info}}',
        sensor_grids_folder=create_octrees_grids._outputs.grid_folder,
        octree=create_octrees_grids._outputs.scene_folder,
        octree_direct=create_octrees_grids._outputs.scene_folder,
        octree_direct_sun=create_octrees_grids._outputs.scene_folder,
        sky_dome=create_sky_dome._outputs.sky_dome,
        total_sky=create_total_sky._outputs.sky_matrix,
        direct_sky=create_direct_sky._outputs.sky_matrix,
        sun_modifiers=generate_sunpath._outputs.sun_modifiers,
        bsdf_folder=create_rad_folder._outputs.bsdf_folder,
        results_folder='../../../results/2_phase'
    ):
        pass

    @task(
        template=ThreePhaseInputsPreparation,
        needs=[
            create_rad_folder, create_octrees_grids,
            create_total_sky, create_sky_dome
        ],
        sub_folder='calcs/3_phase/',
        sub_paths={
            'octree': '__three_phase__.oct'
        }
    )
    def prepare_three_phase(
        self,
        model_folder=create_rad_folder._outputs.model_folder,
        octree=create_octrees_grids._outputs.scene_folder,
        sky_dome=create_sky_dome._outputs.sky_dome,
        bsdf_folder=create_rad_folder._outputs.bsdf_folder,
        dmtx_group_params=dmtx_group_params
    ):
        return [
            {
                'from': ThreePhaseInputsPreparation()._outputs.multiplication_info
            },
            {
                'from': ThreePhaseInputsPreparation()._outputs.grouped_apertures_info
            },
            {
                'from': ThreePhaseInputsPreparation()._outputs.grouped_apertures_folder,
                'to': '../../model/sender'
            },
            {
                'from': ThreePhaseInputsPreparation()._outputs.results_info,
                'to': '../../results/3_phase/_info.json'
            }
        ]

    @task(
        template=ThreePhaseMatrixCalculation,
        needs=[
            create_rad_folder, create_octrees_grids,
            create_total_sky, create_sky_dome,
            prepare_three_phase
        ],
        sub_folder='calcs/3_phase',
        sub_paths={
            'octree': '__three_phase__.oct'
        }
    )
    def calculate_three_phase_matrix_total(
        self,
        model_folder=create_rad_folder._outputs.model_folder,
        grouped_apertures=prepare_three_phase._outputs.grouped_apertures_info,
        grouped_apertures_folder=prepare_three_phase._outputs.grouped_apertures_folder,
        multiplication_info=prepare_three_phase._outputs.multiplication_info,
        receivers=create_rad_folder._outputs.receivers,
        view_mtx_rad_params=view_mtx_rad_params,
        daylight_mtx_rad_params=daylight_mtx_rad_params,
        octree=create_octrees_grids._outputs.scene_folder,
        sky_dome=create_sky_dome._outputs.sky_dome,
        sky_matrix=create_total_sky._outputs.sky_matrix,
        bsdf_folder=create_rad_folder._outputs.bsdf_folder,
    ):
        return [
            {
                'from': ThreePhaseMatrixCalculation()._outputs.results,
                'to': '../../results/3_phase'
            }
        ]

results = Outputs.folder(source='results')
