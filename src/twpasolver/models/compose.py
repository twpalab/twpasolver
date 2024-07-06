"""Direct composition between models."""

from twpasolver.models.oneport import OnePortArray, OnePortModel
from twpasolver.models.twoportarrays import AnyModel, TwoPortArray


def compose(
    *args: AnyModel, parallel=False, **kwargs  # type:ignore
) -> TwoPortArray | OnePortArray:
    """
    Directly compose any number of OnePortModel or TwoPortModels into a ModelArray.

    If all the provided instances are OnePortModels with the same parallel/series configuration
    in a two-port circuit, the resulting composition will be given by a OnePortArray.
    Otherwise a TwoPortArray will be returned.

    Args:
        *args (AnyModel): Any number of OnePortModel or TwoPortModel instances to be composed.
        parallel (bool, optional): Indicates if the OnePortModels should be composed in parallel.
                                   Defaults to False. Applies only if a composition to OnePortModel is possible.
        **kwargs: Additional keyword arguments to pass to the OnePortArray or TwoPortArray constructors.
    """
    if not args:
        raise RuntimeError("Provide at least one Model to insert into the array.")
    if all(isinstance(mod, OnePortModel) for mod in args):
        if all(
            mod.twoport_parallel == args[0].twoport_parallel for mod in args  # type: ignore[attr-defined]
        ):
            return OnePortArray(cells=list(args), parallel=parallel, **kwargs)
    return TwoPortArray(cells=list(args), **kwargs)
