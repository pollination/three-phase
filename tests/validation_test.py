from pollination.three_phase.entry import RecipeEntryPoint
from queenbee.recipe.dag import DAG


def test_three_phase():
    recipe = RecipeEntryPoint().queenbee
    assert recipe.name == 'recipe-entry-point'
    assert isinstance(recipe, DAG)
