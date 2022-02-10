from pollination_dsl.dag import Inputs, DAG, task, Outputs
from pollination_dsl.dag.inputs import ItemType
from dataclasses import dataclass

from pollination.honeybee_radiance.multiphase import DaylightMatrixGrouping, \
    MultiPhaseCombinations

from ._view_matrix import ViewMatrixRayTracing
from ._daylight_matrix import DaylightMtxRayTracing
from ._multiply_matrix import MultiplyMatrixDag


@dataclass
class ThreePhaseEntryPoint(DAG):
    """Three phase daylight simulation DAG."""

    model_folder = Inputs.folder(
        description='Radiance model folder', path='model'
    )

    receivers = Inputs.list(
        description='A list with receivers information.',
        items_type=ItemType.JSONObject
    )

    view_mtx_rad_params = Inputs.str(
        description='The radiance parameters for ray tracing.',
        default='-ab 2 -ad 5000 -lw 2e-05'
    )

    daylight_mtx_rad_params = Inputs.str(
        description='The radiance parameters for ray tracing.',
        default='-ab 2 -ad 5000 -lw 2e-05'
    )

    octree = Inputs.file(
        description='Octree that describes the scene for indirect studies. This octree '
        'includes all the scene with default modifiers except for the aperture groups '
        'other the one that is the source for this calculation will be blacked out.'
    )

    sky_dome = Inputs.file(
        description='A sky dome for daylight coefficient studies.'
    )

    sky_matrix = Inputs.file(
        description='Sky matrix for matrix multiplication.'
    )

    bsdf_folder = Inputs.folder(
        description='The folder from Radiance model folder that includes the BSDF files.'
        'You only need to include the in-scene BSDFs for the two phase calculation.',
        optional=True
    )

    @task(
        template=ViewMatrixRayTracing,
        loop=receivers,
        sub_folder='view_mtx',
        sub_paths={
            'sensor_grid': 'grid/{{item.identifier}}.pts',
            'receiver_file': 'receiver/{{item.path}}',
            'receivers_folder': 'aperture_group',

        }
    )
    def calculate_view_matrix(
        self,
        radiance_parameters=view_mtx_rad_params,
        sensor_count='{{item.count}}',
        receiver_file=model_folder,
        sensor_grid=model_folder,
        scene_file=octree,
        receivers_folder=model_folder,
        bsdf_folder=bsdf_folder,
        fixed_radiance_parameters='-aa 0.0 -I -c 1 -o vmtx/{{item.identifier}}..%%s.vtmx',
    ):
        pass

    @task(template=DaylightMatrixGrouping)
    def group_apertures(
        self,
        model_folder=model_folder,
        scene_file=octree,
        sky_dome=sky_dome
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
        sub_folder='daylight_mtx',
        loop=group_apertures._outputs.grouped_apertures,
        needs=[group_apertures],
        sub_paths={
            'sender_file': '{{item.identifier}}.rad',
            'senders_folder': 'aperture_group'
        }
    )
    def daylight_mtx_calculation(
        self,
        name='{{item.identifier}}',
        radiance_parameters=daylight_mtx_rad_params,
        receiver_file=sky_dome,
        sender_file=group_apertures._outputs.grouped_apertures_folder,
        senders_folder=model_folder,
        scene_file=octree,
        bsdf_folder=bsdf_folder
    ):
        pass

    @task(
        template=MultiPhaseCombinations,
        needs=[group_apertures],
        sub_paths={
            'sender_info': '_info.json',
            'states_info': 'aperture_group/states.json',
            'receiver_info': 'receiver/_info.json'
        }
    )
    def get_three_phase_combinations(
        self,
        sender_info=group_apertures._outputs.grouped_apertures_folder,
        receiver_info=model_folder,
        states_info=model_folder
    ):
        return [
            {
                'from': MultiPhaseCombinations()._outputs.results_mapper,
                'to': 'results/_info.json'
            },
            {
                'from': MultiPhaseCombinations()._outputs.multiplication_info
            }
        ]

    # multiply all the matrices for all the states
    @task(
        template=MultiplyMatrixDag,
        sub_folder='results',
        loop=get_three_phase_combinations._outputs.multiplication_info,
        needs=[get_three_phase_combinations, daylight_mtx_calculation],
        sub_paths={
            'view_matrix': '{{item.vmtx}}',
            't_matrix': 'bsdf/{{item.tmtx}}',
            'daylight_matrix': '{{item.dmtx}}'
        }
    )
    def multiply_matrix(
        self,
        identifier='{{item.identifier}}',
        sky_vector=sky_matrix,
        view_matrix='view_mtx',
        t_matrix=model_folder,
        daylight_matrix='daylight_mtx'
    ):
        pass

    results_folder = Outputs.folder(source='results')
