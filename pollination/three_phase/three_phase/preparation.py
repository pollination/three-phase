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

    size = Inputs.float(
        description='Aperture grid size. A lower number will give a finer grid and more '
        'accurate results but the calculation time will increase.', default=0.2
    )

    threshold = Inputs.float(
        description='A number that determines if two apertures/aperture groups can be '
        'clustered. A higher number is more accurate but will also increase the number '
        'of aperture groups', default=0.001
    )

    ambient_division = Inputs.int(
        description='Number of ambient divisions (-ad) for view factor calculation in '
        'rfluxmtx. Increasing the number will give more accurate results but also '
        'increase the calculation time.', default=1000
    )

    @task(template=DaylightMatrixGrouping)
    def daylight_matrix_aperture_grouping(
        self,
        model_folder=model_folder,
        scene_file=octree,
        sky_dome=sky_dome,
        size=size,
        threshold=threshold,
        ambient_division=ambient_division
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
