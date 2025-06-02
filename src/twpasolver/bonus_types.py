"""Type annotations module."""

from typing import Any, Callable, TypeAlias

import numba as nb
import numpy as np
from pydantic import GetJsonSchemaHandler, NonNegativeFloat, NonNegativeInt
from pydantic.functional_validators import BeforeValidator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import Annotated


def all_subclasses(cls) -> list:
    """
    Recursively get all subclasses of a given class.

    Args:
        cls: The class for which to find all subclasses.

    Returns:
        list: A list of all subclasses of the given class.
    """
    return cls.__subclasses__() + [
        s for c in cls.__subclasses__() for s in all_subclasses(c)
    ]


def validate_impedance(Z: complex | float | str) -> complex | float:
    """
    Validate impedance value.

    Args:
        Z (complex | float | str): Input impedance value.
    """
    try:
        Z = complex(Z)
    except Exception as exc:
        raise ValueError(f"Cannot convert {type(Z)} {Z} to complex number.") from exc
    if np.real(Z) < 0:
        raise ValueError("Real part of impedance {Z} must be non-negative.")
    if np.imag(Z) == 0:
        Z = Z.real
    return Z


class _Impedance2PydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        """Pydantic annotation for impedance type."""
        from_float_schema = core_schema.chain_schema(
            [
                core_schema.float_schema(),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_float_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(complex | float),
                    from_float_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda x: x),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(core_schema.float_schema())


Impedance = Annotated[
    complex | float, _Impedance2PydanticAnnotation, BeforeValidator(validate_impedance)
]
FrequencyList = list[NonNegativeFloat]
FrequencyArange = tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat]
FrequencyLinspace = tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeInt]

ComplexArray: TypeAlias = np.ndarray[Any, np.dtype[np.complex128]]
FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]

nb_complex3d = nb.complex128[:, :, :]
nb_complex1d = nb.complex128[:]
nb_float1d = nb.float64[:]
nb_int2d = nb.int64[:, :]
