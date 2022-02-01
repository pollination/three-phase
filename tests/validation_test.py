from pollination.three_phase.entry import ThreePhaseEntryPoint
from queenbee.recipe.dag import DAG


def test_three_phase():
    recipe = ThreePhaseEntryPoint().queenbee
    assert recipe.name == 'three-phase-entry-point'
    assert isinstance(recipe, DAG)
