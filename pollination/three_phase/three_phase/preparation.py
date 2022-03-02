from pollination_dsl.dag import Inputs, DAG, task, Outputs
from pollination_dsl.dag.inputs import ItemType
from dataclasses import dataclass

from pollination.honeybee_radiance.multiphase import DaylightMatrixGrouping, \
    MultiPhaseCombinations

@dataclass
class ThreePhaseInputsPreparation(DAG):
    """A DAG to prepare the inputs for three phase simulation."""

    model_folder = Inputs.folder(
        description='Radiance model folder', path='model'
    )

    octree = Inputs.file(
        description='Octree that describes the scene for indirect studies. This octree '
        'includes all the scene with default modifiers except for the aperture groups '
        'other the one that is the source for this calculation will be blacked out.'
    )

    sky_dome = Inputs.file(
        description='A sky dome for daylight coefficient studies.'
    )

    bsdf_folder = Inputs.folder(
        description='The folder from Radiance model folder that includes the BSDF files.'
        'You only need to include the in-scene BSDFs for the two phase calculation.',
        optional=True
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

    @task(template=DaylightMatrixGrouping)
    def daylight_matrix_aperture_grouping(
        self,
        model_folder=model_folder,
        scene_file=octree,
        sky_dome=sky_dome,
        dmtx_group_params=dmtx_group_params
    ):
        return [
            {
                'from': DaylightMatrixGrouping()._outputs.grouped_apertures_folder,
                'to': 'model/sender'
            },
            {
                'from': DaylightMatrixGrouping()._outputs.grouped_apertures
            },
            {
                'from': DaylightMatrixGrouping()._outputs.grouped_apertures_file,
                'to': 'model/sender/_info.json'
            },
            
        ]

    @task(
        template=MultiPhaseCombinations,
        needs=[daylight_matrix_aperture_grouping],
        sub_paths={
            'sender_info': '_info.json',
            'states_info': 'aperture_group/states.json',
            'receiver_info': 'receiver/_info.json'
        }
    )
    def get_three_phase_combinations(
        self,
        sender_info=daylight_matrix_aperture_grouping._outputs.grouped_apertures_folder,
        receiver_info=model_folder,
        states_info=model_folder
    ):
        return [
            {
                'from': MultiPhaseCombinations()._outputs.results_mapper,
                'to': 'results/_info.json'
            },
            {
                'from': MultiPhaseCombinations()._outputs.multiplication_file,
                'to': 'multiplication_info.json'
            },
            {
                'from': MultiPhaseCombinations()._outputs.multiplication_info
            }
        ]

    grouped_apertures_folder = Outputs.folder(
        description='A folder with all the grouped apertures for daylight matrix '
        'calculation.', source='model/sender'
    )

    grouped_apertures_info = Outputs.list(
        description='List fo grouped apertures for daylight matrix calculation.',
        source=daylight_matrix_aperture_grouping._outputs.grouped_apertures_file,
        items_type=ItemType.JSONObject
    )

    multiplication_info = Outputs.list(
        description='A JSON file with matrix multiplication information.',
        source='multiplication_info.json', items_type=ItemType.JSONObject
    )

    results_info = Outputs.file(
        description='A JSON file with information for loading the results.',
        source='results/_info.json'
    )
