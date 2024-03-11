"""Type annotations module."""
from typing import Any, Callable, TypeAlias

import numpy as np
from pydantic import GetJsonSchemaHandler, ValidationError
from pydantic.functional_validators import AfterValidator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import Annotated


def validate_impedance(Z: complex | float) -> complex | float:
    """Validate impedance value."""
    if np.real(Z) < 0:
        raise ValidationError("Real part of impedance {Z} must be non-negative.")
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
    complex | float, _Impedance2PydanticAnnotation, AfterValidator(validate_impedance)
]

complex_array: TypeAlias = np.ndarray[Any, np.dtype[np.complex128]]
