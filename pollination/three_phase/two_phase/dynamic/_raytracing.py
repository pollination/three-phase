"""Raytracing DAG for annual daylight."""

from pollination_dsl.dag import Inputs, DAG, task
from dataclasses import dataclass

from pollination.honeybee_radiance.contrib import DaylightContribution
from pollination.honeybee_radiance.coefficient import DaylightCoefficient
from pollination.honeybee_radiance.sky import AddRemoveSkyMatrix


@dataclass
class DynamicRayTracing(DAG):
    # inputs

    radiance_parameters = Inputs.str(
        description='The radiance parameters for ray tracing',
        default='-ab 2 -ad 5000 -lw 2e-05'
    )

    octree_file = Inputs.file(
        description='Octree that describes the scene for indirect studies. This octree '
        'includes all the scene with default modifiers except for the aperture groups '
        'other the one that is the source for this calculation will be blacked out.',
        extensions=['oct']
    )

    octree_file_direct = Inputs.file(
        description='Octree that describes the scene for direct studies. This octree '
        'is similar to the octree for indirect studies with the difference that the '
        'matrials for the scene are set to black.',
        extensions=['oct']
    )

    octree_file_with_suns = Inputs.file(
        description='A blacked out octree that includes the sunpath. This octree is '
        'used for calculating the contribution from direct sunlight.',
        extensions=['oct']
    )

    grid_name = Inputs.str(
        description='Sensor grid file name. This is useful to rename the final result '
        'file to {grid_name}.ill'
    )

    sensor_grid = Inputs.file(
        description='Sensor grid file.',
        extensions=['pts']
    )

    sensor_count = Inputs.int(
        description='Number of sensors in the input sensor grid.'
    )

    sun_modifiers = Inputs.file(
        description='A file with sun modifiers.'
    )

    sky_matrix = Inputs.file(
        description='Path to total sky matrix file.'
    )

    sky_matrix_direct = Inputs.file(
        description='Path to direct skymtx file (i.e. gendaymtx -d).'
    )

    sky_dome = Inputs.file(
        description='Path to sky dome file.'
    )

    bsdfs = Inputs.folder(
        description='Folder containing any BSDF files needed for ray tracing.',
        optional=True
    )

    @task(template=DaylightContribution)
    def direct_sunlight(
        self,
        radiance_parameters=radiance_parameters,
        fixed_radiance_parameters='-aa 0.0 -I -faf -ab 0 -dc 1.0 -dt 0.0 -dj 0.0 -dr 0',
        sensor_count=sensor_count,
        modifiers=sun_modifiers,
        sensor_grid=sensor_grid,
        scene_file=octree_file_with_suns,
        bsdf_folder=bsdfs
    ):
        return [
            {
                'from': DaylightContribution()._outputs.result_file,
                'to': 'direct_sunlight.ill'
            }
        ]

    @task(template=DaylightCoefficient)
    def direct_sky(
        self,
        radiance_parameters=radiance_parameters,
        fixed_radiance_parameters='-aa 0.0 -I -ab 1 -c 1 -faf',
        sensor_count=sensor_count,
        sky_matrix=sky_matrix_direct, sky_dome=sky_dome,
        sensor_grid=sensor_grid,
        scene_file=octree_file_direct,
        bsdf_folder=bsdfs
    ):
        return [
            {
                'from': DaylightCoefficient()._outputs.result_file,
                'to': 'direct_sky.ill'
            }
        ]

    @task(template=DaylightCoefficient)
    def total_sky(
        self,
        radiance_parameters=radiance_parameters,
        fixed_radiance_parameters='-aa 0.0 -I -c 1 -faf',
        sensor_count=sensor_count,
        sky_matrix=sky_matrix, sky_dome=sky_dome,
        sensor_grid=sensor_grid,
        scene_file=octree_file,
        bsdf_folder=bsdfs
    ):
        return [
            {
                'from': DaylightCoefficient()._outputs.result_file,
                'to': 'total_sky.ill'
            }
        ]

    @task(
        template=AddRemoveSkyMatrix,
        needs=[direct_sunlight, total_sky, direct_sky]
    )
    def output_matrix_math(
        self,
        name=grid_name,
        direct_sky_matrix=direct_sky._outputs.result_file,
        total_sky_matrix=total_sky._outputs.result_file,
        sunlight_matrix=direct_sunlight._outputs.result_file,
        conversion='47.4 119.9 11.6'
    ):
        return [
            {
                'from': AddRemoveSkyMatrix()._outputs.results_file,
                'to': '../final/{{self.name}}.ill'
            }
        ]
