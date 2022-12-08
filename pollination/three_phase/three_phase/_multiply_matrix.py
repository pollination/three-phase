from pollination_dsl.dag import Inputs, DAG, task
from dataclasses import dataclass
from pollination.honeybee_radiance.matrix import MatrixMultiplicationThreePhase
from pollination.honeybee_radiance_postprocess.translate import BinaryToNpy


@dataclass
class MultiplyMatrixDag(DAG):

    identifier = Inputs.str(
        description='Aperture state identifier.'
    )

    light_path = Inputs.str(
        description='Aperture state identifier.'
    )

    grid_id = Inputs.str(
        description='Aperture state identifier.'
    )

    state_id = Inputs.str(
        description='Aperture state identifier.'
    )

    sky_vector = Inputs.file(
        description='Path to sky vector.'
    )

    view_matrix = Inputs.file(
        description='Path to view matrix.'
    )

    t_matrix = Inputs.file(
        description='Path to input matrix.'
    )

    daylight_matrix = Inputs.file(
        description='Path to daylight matrix.'
    )

    @task(template=MatrixMultiplicationThreePhase)
    def multiply_threephase_matrix(
        self, identifier=identifier, sky_vector=sky_vector,
        view_matrix=view_matrix, t_matrix=t_matrix,
        daylight_matrix=daylight_matrix,
        output_format='f',
        conversion='illuminance',
        header='keep'
    ):
        return [
            {
                'from': MatrixMultiplicationThreePhase()._outputs.output_matrix,
                'to': 'temp/{{identifier}}.ill'
            }
        ]

    @task(
        template=BinaryToNpy,
        needs=[multiply_threephase_matrix]
    )
    def binary_to_npy(
        self,
        matrix_file=multiply_threephase_matrix._outputs.output_matrix,
        light_path=light_path,
        state_id=state_id,
        grid_id=grid_id
    ):
        return [
            {
                'from': BinaryToNpy()._outputs.output_file,
                'to': '../../results/{{light_path}}/{{state_id}}/total/{{grid_id}}.npy'
            }
        ]
