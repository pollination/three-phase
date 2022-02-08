from pollination_dsl.dag import Inputs, DAG, task
from dataclasses import dataclass
from pollination.honeybee_radiance.multiphase import ViewMatrix


@dataclass
class ViewMatrixRayTracing(DAG):

    radiance_parameters = Inputs.str(
        description='Radiance parameters. -I, -c 1 and -aa 0 are already included in '
        'the command.', default=''
    )

    fixed_radiance_parameters = Inputs.str(
        description='Radiance parameters. -I, -c 1 and -aa 0 are already included in '
        'the command.', default='-aa 0 -I -c 1'
    )

    sensor_count = Inputs.int(
        description='Number of sensors in each generated grid.',
        spec={'type': 'integer', 'minimum': 1}
    )

    receiver_file = Inputs.file(
        description='Path to a receiver file.',
        extensions=['rad']
    )

    sensor_grid = Inputs.file(
        description='Path to sensor grid files.',
        extensions=['pts']
    )

    scene_file = Inputs.file(
        description='Path to an octree file to describe the scene.',
        extensions=['oct']
    )

    receivers_folder = Inputs.folder(
        description='Folder containing any receiver files needed for ray tracing. '
        'This folder is usually the aperture group folder inside the model folder.'
    )

    bsdf_folder = Inputs.folder(
        description='Folder containing any BSDF files needed for ray tracing.',
        optional=True
    )

    @task(template=ViewMatrix)
    def calculate_view_matrix(
        self,
        radiance_parameters=radiance_parameters,
        fixed_radiance_parameters=fixed_radiance_parameters,
        sensor_count=sensor_count,
        receiver_file=receiver_file,
        sensor_grid=sensor_grid,
        scene_file=scene_file,
        receivers_folder=receivers_folder,
        bsdf_folder=bsdf_folder
    ):
        return [
            {
                'from': ViewMatrix()._outputs.view_mtx,
                'to': '.'
            }
        ]
