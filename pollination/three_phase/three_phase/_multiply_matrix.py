from pollination_dsl.dag import Inputs, DAG, task
from dataclasses import dataclass
from pollination.honeybee_radiance.matrix import MatrixMultiplicationThreePhase


@dataclass
class MultiplyMatrixDag(DAG):

    identifier = Inputs.str(
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

    options = Inputs.str(
        description='Additional options for dctimestep', default=''
    )

    @task(template=MatrixMultiplicationThreePhase)
    def multiply_threephase_matrix(
        self, identifier=identifier, sky_vector=sky_vector,
        view_matrix=view_matrix, t_matrix=t_matrix,
        daylight_matrix=daylight_matrix, options=options
    ):
        return [
            {
                'from': MatrixMultiplicationThreePhase()._outputs.output_matrix,
                'to': '{{identifier}}.ill'
            }
        ]
