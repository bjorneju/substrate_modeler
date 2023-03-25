"""
unit.py
=============
This module provides functionality for creating units that constitute substrate models in Integrated information theory.
The 'unit' module is separated into three sections:


Section 1 - Unit classes
------------------------
Defines three classes
    Class 1 -- BaseUnit
    Class 2 -- Unit
    Class 3 -- CompositeUnit
Please refer to the docstrings for information about each of these.

Section 2 - Unit (I/O) functions
--------------------------------
This section contains functions for creating unit TPMs. That is, it provides specific, predifined functions that can be used to define units.

Section 3 - Unit validation
---------------------------
This section contains functions for validating the creation of units and their TPMs.

"""

# TODO:
# - Fix the time spent by caching the unit TPM's (Maybe stick the TPM-dict in the base unit?)
# - allow for composite units
# - allow for non-binary units
# - make better "default values" for "UNIT_VALIDATION" (probably just update in unit functions)
# - REFACTOR to a atom -> unit -> substrate structure
# - Deal with modulation (put into unit params?)


from typing import Union, Tuple, List, Any

from itertools import product
import numpy as np
import pyphi

#from .validate import validate_kwargs
from .utils import map_to_floor_and_ceil, reshape_to_md


class BaseUnit:
    """
    Represents a basic unit in a substrate, with a binary state and input connections.

    Args:
        index (int): The index of the unit in the list of units.
        state (Union[int, Tuple[int,]]): The binary state of the unit, either as an int (0 or 1) or a tuple of ints.
        label (str, optional): A label for the unit. Defaults to None.
        inputs (tuple[int], optional): A tuple of indices of units that input to this unit. Defaults to (None,).
        input_state (tuple[int], optional): The binary state of the input units, as a tuple of ints. Defaults to (None,).

    Attributes:
        index (int): The index of the unit in the substrate.
        state (Tuple[int,]): The binary state of the unit, as a tuple (length 1) or int.
        label (str): A string label for the unit.
        inputs (tuple[int]): A tuple of indices of units that input to this unit.
        input_state (tuple[int]): The binary state of the input units, as a tuple of ints.
    """
    
    def __init__(
        self,
        index: int,
        state: Union[int, Tuple[int,]],
        label: str = None,
        inputs: tuple[int] = (None,),
        input_state: tuple[int] = (None,),
    ):

        # This unit's index in the list of units.
        self._index = index
        
        # Node labels used in the system
        self._label = label

        # List of indices that input to the unit (one pr mechanism).
        self._inputs = inputs

        # Set unit state
        self._state = state
        
        #  and input state
        self._input_state = input_state
        
    @property
    def index(self):
        """int: The index of the unit in the list of units."""
        return self._index
    
    @index.setter
    def index(self, index: int):
        self._index = index
        
    @property
    def state(self):
        """
        Tuple[int,]: The binary state of the unit, as a tuple of ints.

        Raises:
            ValueError: If the state is not a valid binary value (0 or 1).
        """
        if type(self._state)==int:
            if self._state not in (0, 1):
                raise ValueError("state must be 0 or 1")
            self._state = (self._state,)
        else:
            if self._state[0] not in (0, 1):
                raise ValueError("state must be 0 or 1")
        return self._state
    
    @state.setter
    def state(self, state: Union[int, Tuple[int,]]):
        self._state = state
        return self.state
        
    @property
    def label(self):
        """str: A label for the unit."""
        return self._label
    
    @label.setter
    def label(self, label: str=None):
        if not label == None:
            self._label = label
        else:
            self._label = str(self.index)
            
    @property
    def inputs(self):
        """tuple[int]: A tuple of indices of units that input to this unit."""
        return self._inputs
    
    @property
    def input_state(self):
        """
        tuple[int]: The binary state of the input units, as a tuple of ints.
        """
        if not all([s in (0, 1) for s in self._input_state]):
            raise ValueError("all input states must be 0 or 1")
        return self._input_state
    
    @input_state.setter
    def input_state(self, input_state: tuple[int]):
        """
        Set the binary state of the input units.

        Args:
            input_state (tuple[int]): The binary state of the input units, as a tuple of ints.

        Raises:
            ValueError: If any of the input states are not a valid binary value (0 or 1).
        """
        if not all([s in (0, 1) for s in input_state]):
            raise ValueError("all input states must be 0 or 1")
        self._input_state = input_state

    def __repr__(self):
        """
        Return a string representation of the unit.
        """
        return "Unit(label={}, state={})".format(
            self.label, self.state
        )

    def __str__(self):
        """
        Return a string representation of the unit.
        """
        return self.__repr__()


class Unit(BaseUnit):
    """
    Represents a functional unit in a system, with a binary state and input connections,
    as well as additional parameters specific to the unit's input-output mechanism.

    Args:
        index (int): The index of the unit in the list of units.
        state (Union[int, Tuple[int,]]): The binary state of the unit, either as an int (0 or 1) or a tuple of ints.
        inputs (tuple[int]): A tuple of indices of units that input to this unit.
        input_state (tuple[int]): The binary state of the input units, as a tuple of ints.
        mechanism (str): The type of unit mechanism (e.g., 'and', 'or', etc. see unit_functions).
        params (dict): A dictionary of parameters specifying to the unit's mechanism.
        label (str, optional): A label for the unit. Defaults to None.

    Attributes:
        index (int): The index of the unit in the list of units.
        state (Tuple[int,]): The binary state of the unit, as a tuple of ints.
        label (str): A label for the unit.
        inputs (tuple[int]): A tuple of indices of units that input to this unit.
        input_state (tuple[int]): The binary state of the input units, as a tuple of ints.
        params (dict): A dictionary of parameters specific to the unit's mechanism.
        mechanism (str): The type of unit mechanism (e.g., 'and', 'or', etc.).
        tpm (numpy.ndarray): The truth table of the unit, as a numpy array of binary values.
    """
    def __init__(
        self,
        index: int,
        inputs: tuple[int],
        mechanism: str,
        state: Union[int, Tuple[int,]] = None,
        input_state: tuple[int] = None,
        params: dict = None,
        label: str = None,
    ):
        super().__init__(
            index,
            state,
            label,
            inputs,
            input_state,
        )
        
        # Store the parameters
        self._params = params
        
        # Store the type of unit
        self._mechanism = mechanism
        
        # set tpm
        self.tpm

        # validate unit
        assert self.validate(), "Unit did not pass validation"
        
    @property
    def params(self):
        """
        dict: A dictionary of parameters specific to the unit's mechanism.
        """
        return self._params
    
    @params.setter
    def params(self, params: dict):
        self._params = params
        return self.params
        
    @property
    def mechanism(self):
        """
        str: The type of unit mechanism (e.g., 'and', 'or', etc.).
        """
        return self._mechanism
    
    @mechanism.setter
    def mechanism(self, mechanism: str):
        self._mechanism = mechanism
        
    @property
    def tpm(self):
        """
        numpy.ndarray: The truth table of the unit, as a numpy array of binary values.
        """
        if self._params is None:
            self._tpm = None
        else:
            func = UNIT_FUNCTIONS[self.mechanism]
            self._tpm = func(self, **self.params)
        return self._tpm
    
    def state_dependent_tpm(self, substrate_state: tuple[int]):
        """
        Set the binary state of the input units and return the truth table of the unit.

        Args:
            substrate_state (tuple[int]): The binary state of the substrate units, as a tuple of ints.

        Returns:
            numpy.ndarray: The truth table of the unit, as a numpy array of binary values.
        """
        self.state = substrate_state[self.index]
        self.input_state = tuple([
            substrate_state[i] for i in self.inputs
        ])
        return self.tpm
    
    def validate(self):
        """Return whether the specifications for the unit are valid.

        The checks for validity are defined in the local UNIT_FUNCTIONS object.
        """
        if type(self.params) == dict:
            unit_type = self.mechanism

            if not unit_type == "composite":
                assert (
                    unit_type in UNIT_FUNCTIONS.keys()
                ), "Unit mechanism '{}' is not valid".format(unit_type)

        return True

    def __repr__(self):
        return "Unit(type={}, label={}, state={})".format(
            self.mechanism, self.label, self.state
        )

    def __eq__(self, other):
        """Return whether this unitis identical to another.

        Two nodes are equal if they have the same TPMs, states, and inputs.

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (
            np.array_equal(self.tpm, other.tpm)
            and self.state == other.state
            and self.inputs == other.inputs
        )

    def __copy__(self):
        return Unit(
            self.index,
            self.inputs,
            input_state=self.input_state,
            mechanism=self.mechanism,
            params=self.params,
            label=self.label,
            state=self.state,
        )


    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index


class CompositeUnit(Unit):
    """
    Represents a composite unit in a system, composed of multiple individual units.

    Args:
        index (int): The index of the composite unit in the list of units.
        state (Union[int, Tuple[int,]]): The binary state of the composite unit, either as an int (0 or 1) or a tuple of ints.
        units (List[Unit]): A list of individual units that compose the composite unit.
        label (str, optional): The label for the composite unit.
        mechanism_combination (Union[str, np.ndarray, dict], optional): The mechanism(s) used to combine the individual units' truth tables into a composite truth table.

    Attributes:
        inputs (tuple[int]): The indices of the input units to the composite unit.
        input_state (tuple[int]): The binary states of the input units to the composite unit.
        tpm (numpy.ndarray): The truth table of the composite unit, as a numpy array of binary values.

    Methods:
        state_dependent_tpm(substrate_state): Set the binary state of the input units and return the truth table of the composite unit.

    """

    def __init__(
        self,
        index: int,
        units: List[Unit],
        state: Union[int, Tuple[int,]],
        label: str = None,
        mechanism_combination: Union[str, np.ndarray, dict] = None,
    ):
        # store the list of `Unit` objects that make up this `CompositeUnit`
        self.units = units  
        
        # Store the waythe tpms from the component `Unit`s combine to give the composit I/O function
        self.mechanism_combination = mechanism_combination
        
        # Determine the input indices of the `CompositeUnit`
        self._inputs = self.inputs
        
        # Determine the input state of the `CompositeUnit`
        self._input_state = self.input_state
        
        # Initialize the unit object
        super().__init__(
            index = index,
            state = state,
            inputs = self._inputs,
            input_state = self._input_state,
            mechanism = 'composite: {}'.format('+'.join([unit.mechanism for unit in self.units])),
            params = None,
            label = label,
        )
        
        # get the TPM of the composite unit
        self.tpm
        
    @property
    def inputs(self):
        """
        tuple[int]: The indices of the input units to the composite unit.
        """
        all_inputs = tuple(set([ix for unit in self.units for ix in unit._inputs]))
        self._inputs = tuple(sorted(all_inputs))
        return self._inputs
    
    @property
    def input_state(self):
        """
        tuple[int]: The binary states of the input units to the composite unit.
        """
        state_dict = dict()
        for unit in self.units:
            for ix, state in zip(unit._inputs, unit._input_state):
                # NOTE! if there are conflicts, the latter will be used
                state_dict[ix] = state
                
        self._input_state = tuple([state_dict[ix] for ix in self._inputs])
        return self._input_state
    
    @input_state.setter
    def input_state(self, input_state: Tuple[int]):
        """
        Set the binary states of the input units to the composite unit.

        Args:
            input_state (tuple[int]): The binary states of the input units, as a tuple of ints.
        """
        for unit in self.units:
            unit.input_state = tuple([
                state for state, i in zip(input_state, self._inputs)
                if i in unit._inputs 
            ])
        self._input_state = input_state
        return self.input_state
        
    @property
    def tpm(self):
        """
        numpy.ndarray: The truth table of the composite unit, as a numpy array of binary values.
        """
        tpms = [unit.tpm for unit in self.units]
        return self.combine_unit_tpms(tpms)
    
    def state_dependent_tpm(self, substrate_state: tuple[int]):
        """
        Set the binary state of the input units and return the truth table of the unit.

        Args:
            substrate_state (tuple[int]): The binary state of the substrate units, as a tuple of ints.

        Returns:
            numpy.ndarray: The truth table of the unit, as a numpy array of binary values.
        """
        self.state = substrate_state[self.index]
        self.input_state = tuple([
            substrate_state[i] for i in self.inputs
        ])
        
        return self.tpm

    def combine_unit_tpms(self, tpms):
        # Check this
        expanded_tpm = self.expand_tpms(tpms)

        # combine subunit TPMs into composite unit tpm
        return self.apply_tpm_combination(expanded_tpm)

    def expand_tpms(self, tpms):
        def get_subset_state(state, subset_indices):
            """tuple[int (binary)]: the state of a subset of indices.

            Args:
                state (tuple[int(binary)]): The (binary) state of the full set of inputs.
                subset_indices (tuple[int]): The indices (relative to the state) for the subset.
            """
            return tuple([state[ix] for i, ix in enumerate(subset_indices)])

        expanded_tpms = []
        for tpm, unit in zip(tpms, self.units):
            
            # get indices of unit inputs among the composite unit inputs
            unit_specific_inputs = tuple([
                unit._inputs.index(i) for i in self._inputs 
                if i in unit._inputs
            ])
            
            # get mechanism activation probabilities for all potential input states
            mechanism_activation = []
            for state in pyphi.utils.all_states(len(self._inputs)):
                P = tpm[get_subset_state(state, unit_specific_inputs)]
                
                if type(P) == np.ndarray:
                    P = P[0]
                mechanism_activation.append(P)

            expanded_tpms.append(mechanism_activation)

        # make the TPMs into an array of correct shape
        return np.array(expanded_tpms).T
    
    
    def apply_tpm_combination(self, expanded_tpms):
        if self.mechanism_combination == "selective":

            def get_selective(P):
                Q = np.array([np.abs(p - 0.5) for p in P])
                return P[np.argmax(Q)]

            tpm = reshape_to_md(
                np.array(
                    [
                        [get_selective(activation_probabilities)]
                        for activation_probabilities in expanded_tpms
                    ]
                )
            )
        elif self.mechanism_combination == "average":

            tpm = reshape_to_md(
                np.array(
                    [
                        [np.mean(activation_probabilities)]
                        for activation_probabilities in expanded_tpms
                    ]
                )
            )
        elif self.mechanism_combination == "maximal":

            tpm = reshape_to_md(
                np.array(
                    [
                        [np.max(activation_probabilities)]
                        for activation_probabilities in expanded_tpms
                    ]
                )
            )
        elif self.mechanism_combination == "first_necessary":

            def first_necessary(ap):
                # non-primary units boost activation probability as a function of the primary unit's activation probability
                primary = ap[0]

                if primary > 0.5:
                    non_primary = np.prod([1 - p for p in ap[1:]])
                    max_boost = 1 - primary
                    boost = max_boost / (1 + np.e ** (-5 * (1 - non_primary - 0.5)))
                    return primary + boost
                else:
                    return primary

            tpm = reshape_to_md(
                np.array(
                    [
                        [(first_necessary(activation_probabilities))]
                        for activation_probabilities in expanded_tpms
                    ]
                )
            )
        elif self.mechanism_combination == "integrator":

            def get_cumulated_probability(activation_probabilities):
                cumsum = np.sum(activation_probabilities)
                if cumsum > 1.0:
                    return 1.0
                elif cumsum < 0.0:
                    return 0.0
                else:
                    return cumsum

            tpm = reshape_to_md(
                np.array(
                    [
                        [get_cumulated_probability(activation_probabilities)]
                        for activation_probabilities in expanded_tpms
                    ]
                )
            )
        elif self.mechanism_combination == "serial":

            def serial_func(P):
                remainder = 1
                for p in P:
                    remainder -= p * remainder
                return 1 - remainder

            tpm = reshape_to_md(
                np.array(
                    [
                        [serial_func(activation_probabilities)]
                        for activation_probabilities in expanded_tpms
                    ]
                )
            )
        return tpm
    
    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index

# UNIT FUNCTIONS

def sigmoid(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
    input_weights: Tuple[float] = None,
    determinism: float = 5.0,
    threshold: float = 0.0,
    ising: bool = True,
):
    # validate all kwargs
    validate_kwargs(locals())
    
    def LogFunc(state, input_weights, determinism, threshold):

        if ising:
            state = tuple([s * 2 - 1 for s in state])
        total_input = sum(state * np.array([input_weights[n] for n in range(n_nodes)]))
        y = 1 / (1 + np.e ** (-determinism * (total_input - threshold)))
        return map_to_floor_and_ceil(y,floor,ceiling)

    n_nodes = len(input_weights)

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                LogFunc(
                    state,
                    input_weights,
                    determinism,
                    threshold,
                )
            ]
            for state in pyphi.utils.all_states(n_nodes)
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def sor_gate(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
    pattern_selection: 
    Tuple[Tuple[int]] = None, 
    selectivity: float = 2.0,
):

    # get state of

    # Ensure states are tuples
    pattern_selection = list(map(tuple, pattern_selection))

    # Ensure nceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    # Ensure selectivity is larger than 1
    if not selectivity > 1:
        print(
            "Selectivity for SOR gates must be bigger than 1, adjusting to inverse of value given."
        )
        selectivity = 1 / selectivity

    # Ensure unit has input state
    if unit.input_state == None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.params["mechanism"], unit.label
            )
        )
        unit.set_input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state == None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    # the tpm is uniform for all states except the input state (i.e. short term plasticity)
    tpm = np.ones([2] * (len(unit.input_state))) * floor

    # if the input state matches a pattern in the patternselction, activation probability given that state is ceiling, otherwise, it is increased by a fraction of the difference between floor and ceiling (given by selectivity)
    pattern = floor + (ceiling - floor) / selectivity
    for state in pattern_selection:
        tpm[state] = pattern
    if unit.input_state in pattern_selection:
        tpm[unit.input_state] = ceiling
    else:
        tpm[unit.input_state] = 0.0

    return tpm


def resonnator(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
    input_weights: Tuple[float] = None,
    determinism=None,
    threshold=None,
    weight_scale_mapping=None,
):

    # Ensure nceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    # Ensure unit has input state
    if unit.input_state == None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.mechanism, unit.label
            )
        )
        unit.input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state == None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.state((0,))

    # the tpm is uniform for all states except the input state (i.e. short term plasticity)
    tpm = np.ones([2] * (len(unit.input_state))) * floor

    # if the input state matches a pattern in the patternselction, activation probability given that state is ceiling, otherwise, it is increased by a fraction of the difference between floor and ceiling (given by selectivity)

    n_nodes = len(input_weights)
    unit_state = unit.state[0]

    # alter weight to make it push unit towards its state, weighted using weight_scale_mapping
    w = [
        input_weights[i] * weight_scale_mapping[(unit_state, s)]
        if s == unit_state
        else -input_weights[i] * weight_scale_mapping[(unit_state, s)]
        for i, s in enumerate(unit.input_state)
    ]

    def resonnatorFunc(state, input_weights, determinism, threshold, unit_state):
        # make state interpreted as ising
        state = [2 * s - 1 for s in state]

        total_input = np.sum(state * np.array(w))
        y = 1 / (1 + np.e ** (-determinism * (total_input - threshold)))
        return y

    # producing transition probability matrix
    tpm = np.array(
        [
            [resonnatorFunc(state, input_weights, determinism, threshold, unit_state)]
            for state in pyphi.utils.all_states(n_nodes)
        ]
    )

    # make it between floor and ceiling
    tpm = map_to_floor_and_ceil(tpm,floor,ceiling)

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def mismatch_pattern_detector(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
    pattern_selection: Tuple[Tuple[int]] = None, 
    selectivity: float = 1.0
):
    # This mechanism is selective to certain inputs (i.e. they turn it ON with P=ceiling, while the remaining possible input patterns turn it OFF with P=floor). However, it's selectivity (probability of turning on) depends the state of the unit: if the unit is already in the state that matches the pattern, then the effect of the inputs is reduced by the selectivity factor. That is, if the unit is ON, and one of its patterns are on its inputs, then the probability that *this mechanism* will turn keep it ON in the next step i P=0.5+(ceiling-0.5)/selectivity.
    # The mechanism is supposed to mimic short-term plasticity mechanisms (or other short term adaptive changes in the function of cells) that make them strongly responsive to mismatching/unpredicted inputs, but weakly coupled to inputs that provide no new information (inputs that match the predicted state of things).

    # Ensure states are tuples
    pattern_selection = list(map(tuple, pattern_selection))

    # Ensure nceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    if floor == None:
        floor = 1.0 - ceiling

    # Ensure selectivity is larger than 1
    if not selectivity > 1:
        print(
            "Selectivity for SOR gates must be bigger than 1, adjusting to inverse of value given."
        )
        selectivity = 1 / selectivity

    # Ensure unit has input state
    if unit.input_state == None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.params["mechanism"], unit.label
            )
        )
        unit.set_input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state == None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    # Check if the unit is ON
    if unit.state == (1,):
        # since it is ON, it will only respond strongly if a non-pattern is on its inputs
        P_pattern = 0.5 + (ceiling - 0.5) / selectivity
        P_no_pattern = floor
    else:
        # since it is OFF, it will only respond strongly if a pattern is on its inputs
        P_pattern = ceiling
        P_no_pattern = 0.5 - (0.5 - floor) / selectivity

    # the tpm is uniform for all states except the input state (i.e. short term plasticity)
    N = len(unit.input_state)
    tpm = np.ones([2] * N)

    for state in pyphi.utils.all_states(N):
        if state in pattern_selection:
            tpm[state] = P_pattern
        else:
            tpm[state] = P_no_pattern

    return tpm


def copy_gate(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    tpm = np.ones([2]) * floor
    tpm[1] = ceiling
    return tpm


def and_gate(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    tpm = np.ones((2, 2)) * floor
    tpm[(1, 1)] = ceiling
    return tpm


def or_gate(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    tpm = np.ones((2, 2)) * ceiling
    tpm[(0, 0)] = floor
    return tpm


def xor_gate(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    tpm = np.ones((2, 2)) * floor
    tpm[(0, 1)] = ceiling
    tpm[(1, 0)] = ceiling
    return tpm


def weighted_mean(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
    weights: List[float] = [],
):

    weights = [w / np.sum(weights) for w in weights]
    N = len(weights)

    tpm = np.ones((2,) * N)
    for state in pyphi.utils.all_states(N):
        weighted_mean = (
            sum([(1 + w * (s * 2 - 1)) / 2 for w, s in zip(weights, state)]) / N
        )
        tpm[state] = weighted_mean * (ceiling - floor) + floor

    return tpm


def democracy(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
):

    N = len(unit.inputs)

    tpm = np.ones((2,) * N)
    for state in pyphi.utils.all_states(N):
        avg_vote = np.mean(state)
        tpm[state] = avg_vote * (ceiling - floor) + floor

    return tpm


def majority(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
):

    N = len(unit.inputs)

    tpm = np.ones((2,) * N)
    for state in pyphi.utils.all_states(N):
        avg_vote = round(np.mean(state))
        tpm[state] = avg_vote * (ceiling - floor) + floor

    return tpm


def mismatch_corrector(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
    bias: float = 0.0
):

    # Ensure unit has input state
    if unit.input_state == None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.params["mechanism"], unit.label
            )
        )
        unit.set_input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state == None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    # Ensure unit has state
    if bias > 1:
        print("bias must be below 1, setting to 1.".format(unit.label))
        bias = 1

    # Ensure that there is only one unit
    if len(unit.inputs) > 1:
        print(
            "Unit {} has too many inputs for mechanism of typ {}. Using only first input.".format(
                unit.label
            )
        )
        unit.set_inputs(tuple(unit.inputs[[0]]))

    # check whether state of unit matches its input, and create tpm accordingly
    if unit.state == unit.input_state:
        tpm = np.ones([2]) * 0.5 - (unit.state[0] * 2 - 1) * bias * 0.5
    else:
        tpm = np.array([floor, ceiling])

    return tpm


def modulated_sigmoid(
    unit: Unit,
    input_weights: List[float],
    modulation: dict,
    floor: float = 0.0,
    ceiling: float = 1.0,
    determinism: float = 2.0,
    threshold: float = 0.0,
):
    # modulation must be a dict like {'modulator': tuple(index), 'threshold': float, 'determinism': float}
    # modulation will update the sigmoid function as indicated by the functions, and depending on whether the unit is ON or OFF

    # Ensure unit has state
    if unit.state == None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    def LogFunc(
        input_state, modulation_state, unit_state, weights, determinism, threshold
    ):
        total_input = sum(input_state * np.array([weight for weight in weights]))
        # count how many of the modulators are ON
        mods_on = sum(modulation_state)

        # modulate threshold based on unit state and the state of the modulators
        new_threshold = threshold + unit_state * mods_on * modulation["threshold"]

        # modulate determinism based on unit state and the state of the modulators
        new_determinism = determinism + unit_state * mods_on * modulation["determinism"]

        y = ceiling * (
            floor
            + (1 - floor)
            / (1 + np.e ** (-new_determinism * (total_input - new_threshold)))
        )
        return y

    # first inputs will be interpreted as true inputs, while the last will be modulator inputs
    n_mods = len(modulation["modulator"])
    n_inputs = len(input_weights)
    n_nodes = n_inputs + n_mods

    unit_state = (
        unit.state[0] * 2 - 1
    )  # making unit state "ising" rather than binary, to make modulation symmetric

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                LogFunc(
                    input_state,
                    modulation_state,
                    unit_state,
                    input_weights,
                    determinism,
                    threshold,
                )
            ]
            for modulation_state, input_state in product(
                pyphi.utils.all_states(n_mods), pyphi.utils.all_states(n_inputs)
            )
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def stabilized_sigmoid(
    unit: Unit,
    input_weights: list,
    determinism: float,
    threshold: float,
    modulation: dict,
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    # modulation must be a dict like {'modulator': tuple(index), 'threshold': float, 'determinism': float}
    # modulation will update the sigmoid function as indicated by the functions, and depending on whether the unit is ON or OFF

    # Ensure unit has state
    if unit.state == None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    def LogFunc(
        input_state, modulation_state, unit_state, weights, determinism, threshold
    ):
        total_input = sum(input_state * np.array([weight for weight in weights]))

        # The modulation should work in such a way as to always "stabilize" the current state of the unit, but it should be stronger, the more modulation is "active"

        # count how many of the modulators are ON
        mods_on = sum(modulation_state)
        if mods_on == 0:
            mods_on = 1 / modulation["selectivity"]

        ising_state = (
            unit_state * 2 - 1
        )  # making unit state "ising" rather than binary, to make modulation symmetric

        # modulate threshold based on unit state and the state of the modulators
        new_threshold = threshold - ising_state * mods_on * modulation["threshold"]

        # modulate determinism based on unit state and the state of the modulators
        new_determinism = (
            determinism
            if (mods_on == 0 or modulation["determinism"] == 0)
            else (
                determinism * float(mods_on * modulation["determinism"]) ** ising_state
            )
        )

        y = ceiling * (
            floor
            + (1 - floor)
            / (1 + np.e ** (-new_determinism * (total_input - new_threshold)))
        )
        return y

    # first inputs will be interpreted as true inputs, while the last will be modulator inputs
    n_mods = len(modulation["modulator"])
    n_inputs = len(input_weights)
    n_nodes = n_inputs + n_mods

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                LogFunc(
                    input_state,  # tuple(s * 2 - 1 for s in input_state),
                    modulation_state,
                    unit.state[0],
                    input_weights,
                    determinism,
                    threshold,
                )
            ]
            for input_state, modulation_state in product(
                pyphi.utils.all_states(n_inputs), pyphi.utils.all_states(n_mods)
            )
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def biased_sigmoid(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,
    input_weights: Tuple[float] = None,
    determinism: float = 2.0,
    threshold: float = 0.0,
):
    # A sigmoid unit that is biased in its activation by the last unit in the inputs.
    # The bias consists in a rescaling of the activation probability to make it more in line with the biasing unit. The biasing unit is assumed to be the last one of the inputs.
    # For example, if the biased unit is OFF, the sigmoid activation probability is divided by the factor given in the last value of input_weights. If the unit is ON, 1 - the activation probability is divided by the factor (in essence reducing the probability that it will NOT activate).

    def LogFunc(total_input, determinism, threshold):
        y = ceiling * (
            floor
            + (1 - floor) / (1 + np.e ** (-determinism * (total_input - threshold)))
        )
        return y

    n_nodes = len(input_weights)

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                LogFunc(
                    sum(
                        state[:-1] * np.array([weight for weight in input_weights[:-1]])
                    ),
                    determinism,
                    threshold,
                )
                / input_weights[-1]
                if state[-1] == 0
                else 1
                - (
                    1
                    - LogFunc(
                        sum(
                            state[:-1]
                            * np.array([weight for weight in input_weights[:-1]])
                        ),
                        determinism,
                        threshold,
                    )
                )
                / input_weights[-1]
            ]
            for state in pyphi.utils.all_states(n_nodes)
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def gabor_gate(
    unit: Unit,
    floor: float = 0.0,
    ceiling: float = 1.0,  
    preferred_states: Tuple[Tuple[int]] = None, 
):

    # Ensure states are tuples
    preferred_states = list(map(tuple, preferred_states))
    anti_states = [tuple([int(1 - s) for s in state]) for state in preferred_states]

    # Ensure nceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    # Ensure unit has input state
    if unit.input_state == None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.params["mechanism"], unit.label
            )
        )
        unit.set_input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state == None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    # if the unit is ON, its tpm should indicate that the past state was likely one of its preferred_states
    # if the unit is OFF, its tpm should indicate that the past state was likely one of its anti_states

    # the tpm is uniform for all states except the input state (i.e. short term plasticity)
    tpm = np.ones([2] * (len(unit.input_state))) * 0.5
    for input_state in pyphi.utils.all_states(len(unit.inputs)):
        if input_state in preferred_states:
            tpm[input_state] = ceiling
        elif input_state in anti_states:
            tpm[input_state] = floor
    """
    if unit.state[0]:
        for input_state in pyphi.utils.all_states(len(unit.inputs)):
            if input_state in preferred_states:
                tpm[input_state] = ceiling
            elif input_state in anti_states:
                tpm[input_state] = floor
    else:
        for input_state in pyphi.utils.all_states(len(unit.inputs)):
            if input_state in preferred_states:
                tpm[input_state] = floor
            elif input_state in anti_states:
                tpm[input_state] = ceiling
                """
    return tpm



### UNITS 
UNIT_FUNCTIONS = {
    "gabor": gabor_gate,
    "sigmoid":sigmoid,
    "resonnator": resonnator,
    "sor": sor_gate,
    "mismatch_pattern_detector": mismatch_pattern_detector,
    "copy": copy_gate,
    "and": and_gate,
    "or": or_gate,
    "xor": xor_gate,
    "weighted_mean": weighted_mean,
    "democracy": democracy,
    "majority": majority,
    "mismatch_corrector": mismatch_corrector,
    "modulated_sigmoid": modulated_sigmoid,
    "stabilized_sigmoid": stabilized_sigmoid,
    "biased_sigmoid": biased_sigmoid,
}

# VALIDATION FUNCTIONS

# SEE VALIDATION FUNCTIONS BELOW
def validate_kwargs(kwargs):
    """Validates keyword arguments using respective validation functions."""
    for arg_name, arg_value in kwargs.items():
        # Get validation function for this argument, if provided
        validation_func = VALIDATION_FUNCTIONS.get(arg_name)

        if validation_func:
            # Validate argument using its validation function
            if not validation_func(kwargs["unit"], arg_value):
                raise ValueError(f"Invalid value for argument {arg_name}: {arg_value}")


def validate_unit(unit: Unit, value: Any) -> Unit:
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


def validate_floor(unit: Unit, value: Any) -> float:
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


def validate_ceiling(unit: Unit, value: Any) -> float:
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


def validate_input_weights(unit: Unit, value: Any) -> Tuple[float]:
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

