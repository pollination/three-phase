from pollination_dsl.dag import Inputs, DAG, task
from dataclasses import dataclass
from pollination.honeybee_radiance.multiphase import FluxTransfer


@dataclass
class DaylightMtxRayTracing(DAG):

    name = Inputs.str(
        description='Sender group name. This is useful to rename the final result '
        'file to sender {name}.'
    )

    radiance_parameters = Inputs.str(
        description='Radiance parameters. -I, -c 1 and -aa 0 are already included in '
        'the command.', default=''
    )

    receiver_file = Inputs.file(
        description='Path to a receiver file.',
        extensions=['rad']
    )

    sender_file = Inputs.file(
        description='Path to a sender file.',
        extensions=['rad']
    )

    senders_folder = Inputs.folder(
        description='Folder containing any sender files needed for ray tracing. '
        'This folder is usually the aperture group folder inside the model folder.'
    )

    scene_file = Inputs.file(
        description='Path to an octree file to describe the scene.',
        extensions=['oct']
    )

    bsdf_folder = Inputs.folder(
        description='Folder containing any BSDF files needed for ray tracing.',
        optional=True
    )

    @task(template=FluxTransfer)
    def calculate_daylight_matrix(
        self,
        name=name,
        radiance_parameters=radiance_parameters,
        receiver_file=receiver_file,
        sender_file=sender_file,
        senders_folder=senders_folder,
        scene_file=scene_file,
        bsdf_folder=bsdf_folder
    ):
        return [
            {
                'from': FluxTransfer()._outputs.flux_mtx,
                'to': '{{name}}.dmtx'
            }
        ]
