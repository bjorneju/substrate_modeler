#from network_builder import Unit

from typing import Tuple, Any


# SEE VALIDATION FUNCTIONS BELOW
def validate_kwargs(**kwargs):
    """Validates keyword arguments using respective validation functions."""
    for arg_name, arg_value in kwargs.items():
        # Get validation function for this argument, if provided
        validation_func = VALIDATION_FUNCTIONS.get(arg_name)

        if validation_func:
            # Validate argument using its validation function
            if not validation_func(arg_value):
                raise ValueError(f"Invalid value for argument {arg_name}: {arg_value}")


def validate_unit(value: Any) -> Unit:
    """Validates that value is an object of the class Unit, and has the properties inputs, state and input_state."""
    if not isinstance(value, Unit):
        raise ValueError(f"value must be of the class Unit, but got {type(value)}")
    if not hasattr(value, "inputs"):
        raise ValueError(f"value must have an 'inputs' attribute")
    if not hasattr(value, "state"):
        value.state = (0,)
        print("Warning: state property was not present, it has been set to (0,)")
    if not hasattr(value, "input_state"):
        value.input_state = (0,) * len(value.inputs)
        print("Warning: input_state property was not present, it has been set to (0,) * len(inputs)")
    return value


def validate_floor(value: Any) -> float:
    """Validates that value is a float between 0 and 1, inclusive."""
    if not isinstance(value, float):
        raise ValueError(f"value must be a float, but got {type(value)}")
    if not 0 <= value <= 1:
        if value < 0:
            value = 0.
            print("Warning: floor value was less than 0, it has been set to 0")
        else:
            value = 1.
            print("Warning: floor value was greater than 1, it has been set to 1")
    return value


def validate_ceiling(value: Any) -> float:
    """Validates that value is a float between 0 and 1, inclusive."""
    if not isinstance(value, float):
        raise ValueError(f"value must be a float, but got {type(value)}")
    if not 0 <= value <= 1:
        if value < 0:
            value = 0.
            print("Warning: ceiling value was less than 0, it has been set to 0")
        else:
            value = 1.
            print("Warning: ceiling value was greater than 1, it has been set to 1")
    return value


def validate_input_weights(value: Any, unit: Unit) -> Tuple[float]:
    """Validates that value is a tuple of floats between 0 and 1, inclusive, and its length is the same as unit.inputs."""
    if not isinstance(value, tuple):
        raise ValueError(f"value must be a tuple, but got {type(value)}")
    if len(value) != len(unit.inputs):
        value = (1.,) * len(unit.inputs)
        print(f"Warning: input_weights must be a tuple of length {len(unit.inputs)}, it has been set to {(1,) * len(unit.inputs)}")
    for v in value:
        if not isinstance(v, float):
            raise ValueError(f"all values in the input_weights tuple must be floats, but got {type(v)}")
    return value


VALIDATION_FUNCTIONS = {
    "unit": validate_unit,
    "floor": validate_floor,
    "ceiling": validate_ceiling,
    "input_weights": validate_input_weights,
    # Add more validation functions for other arguments as needed
}

