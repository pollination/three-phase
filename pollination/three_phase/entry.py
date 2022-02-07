from pollination_dsl.dag import Inputs, DAG, task, Outputs
from dataclasses import dataclass
from pollination.honeybee_radiance.translate import CreateRadianceFolderGrid
from pollination.honeybee_radiance.octree import CreateOctree
from pollination.honeybee_radiance.sky import CreateSkyDome, CreateSkyMatrix
from pollination.honeybee_radiance.multiphase import DMTXGroup

from ._view_matrix import ViewMatrixRayTracing
from ._daylight_matrix import DaylightMtxRayTracing
from ._multiply_matrix import MultiplyMatrixDag

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
        'cpu_count is always respected.', default=1,
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
                'from': CreateRadianceFolderGrid()._outputs.states,
                'description': 'Dynamic blinds states information.'
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
                'to': 'octrees/main.oct'
            }
        ]

    @task(template=CreateSkyDome)
    def create_sky_dome(self):
        """Create sky dome for daylight coefficient studies."""
        return [
            {
                'from': CreateSkyDome()._outputs.sky_dome,
                'to': 'sky_vectors/sky_dome.rad'
            }
        ]

    @task(template=CreateSkyMatrix)
    def create_total_sky(self, north=north, wea=wea, sun_up_hours='sun-up-hours'):
        return [
            {
                'from': CreateSkyMatrix()._outputs.sky_matrix,
                'to': 'sky_vectors/sky.mtx'
            }
        ]


    @task(
        template=ViewMatrixRayTracing,
        needs=[_create_rad_folder, create_octree],
        loop=_create_rad_folder._outputs.sensor_grids,
        sub_folder='initial_results/view_mtx',
        sub_paths={
            'sensor_grid': 'grid/{{item.full_id}}.pts',
            'receiver_file': 'receivers/{{item.full_id}}..receiver.rad',
            'receivers_folder': 'aperture_group'
        }
    )
    def calculate_view_matrix(
        self,
        grid_name='{{item.full_id}}',
        radiance_parameters=radiance_parameters,
        sensor_count='{{item.count}}',
        receiver_file=_create_rad_folder._outputs.model_folder,
        sensor_grid=_create_rad_folder._outputs.model_folder,
        scene_file=create_octree._outputs.scene_file,
        receivers_folder=_create_rad_folder._outputs.model_folder,
        bsdf_folder=_create_rad_folder._outputs.bsdf_folder
    ):
        pass

    @task(template=DMTXGroup, needs=[_create_rad_folder, create_octree, create_sky_dome])
    def group_apertures(
        self,
        model_folder=_create_rad_folder._outputs.model_folder,
        scene_file=create_octree._outputs.scene_file,
        sky_dome=create_sky_dome._outputs.sky_dome
    ):
        return [
            {
                'from': DMTXGroup()._outputs.grouped_apertures_folder,
                'to': 'initial_results/grouped_apertures'
            },
            {
                'from': DMTXGroup()._outputs.matrix_mapping
            },
            {
                'from': DMTXGroup()._outputs.matrix_mapping_file,
                'to': 'results/info.json'
            },
            {
                'from': DMTXGroup()._outputs.grouped_apertures
            }
        ]

    @task(
        template=DaylightMtxRayTracing,
        sub_folder='initial_results/daylight_mtx',
        loop=group_apertures._outputs.grouped_apertures,
        needs=[_create_rad_folder, create_octree, create_sky_dome, group_apertures],
        sub_paths={
            'sender_file': '{{item}}.rad',
            'senders_folder': 'aperture_group'
        }
    )
    def daylight_mtx_calculation(
        self,
        name='{{item}}',
        radiance_parameters=radiance_parameters,
        receiver_file=create_sky_dome._outputs.sky_dome,
        sender_file=group_apertures._outputs.grouped_apertures_folder,
        senders_folder=_create_rad_folder._outputs.model_folder,
        scene_file=create_octree._outputs.scene_file,
        bsdf_folder=_create_rad_folder._outputs.bsdf_folder
    ):
        pass

    # multiply all the matrices for all the states
    @task(
        template=MultiplyMatrixDag,
        sub_folder='results',
        loop=group_apertures._outputs.matrix_mapping,
        needs=[group_apertures, daylight_mtx_calculation, create_total_sky],
        sub_paths={
            'view_matrix': '{{item.vmtx}}.vmtx',
            't_matrix': '{{item.tmtx}}',
            'daylight_matrix': '{{item.dmtx}}.dmtx'
        }
    )
    def multiply_matrix(
        self,
        identifier='{{item.identifier}}',
        sky_vector=create_total_sky._outputs.sky_matrix,
        view_matrix='initial_results/view_mtx',
        t_matrix='model/bsdf',
        daylight_matrix='initial_results/daylight_mtx'
    ):
        pass


results = Outputs.folder(source='results')
