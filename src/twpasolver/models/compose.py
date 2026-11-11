"""Direct composition between models."""

from twpasolver.models.oneport import OnePortArray, OnePortModel
from twpasolver.models.twoportarrays import TwoPortArray
from twpasolver.twoport import TwoPortModel


def compose(
    *args: TwoPortModel, parallel=False, **kwargs
) -> TwoPortArray | OnePortArray:
    """
    Compose any number of OnePortModels or TwoPortModels into a ModelArray.

    Returns a OnePortArray if all inputs are OnePortModels with the same
    ``twoport_parallel`` setting, otherwise a TwoPortArray.

    Args:
        *args (TwoPortModel): Models to compose.
        parallel (bool, optional): Compose the OnePortModels in parallel (OnePortArray
            result only). Defaults to False.
        **kwargs: Passed to the OnePortArray or TwoPortArray constructor.
    """
    if not args:
        raise RuntimeError("Provide at least one Model to insert into the array.")
    if all(isinstance(mod, OnePortModel) for mod in args):
        if all(
            mod.twoport_parallel == args[0].twoport_parallel for mod in args  # type: ignore[attr-defined]
        ):
            return OnePortArray(cells=list(args), parallel=parallel, **kwargs)
    return TwoPortArray(cells=list(args), **kwargs)
