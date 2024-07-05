"""Direct composition between models."""

from twpasolver.models.oneport import OnePortArray, OnePortModel
from twpasolver.models.twoportarrays import AnyModel, TwoPortArray


def compose(
    *args: AnyModel, parallel=False  # type:ignore
) -> TwoPortArray | OnePortArray:
    """Directly compose any number of OnePortModel or TwoPortModels."""
    if all(isinstance(mod, OnePortModel) for mod in args):
        return OnePortArray(cells=list(args), parallel=parallel)
    return TwoPortArray(cells=list(args))
