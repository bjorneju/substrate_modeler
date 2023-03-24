# TODO:
# - allow for composite units
# - allow for non-binary units
# - make better "default values" for "UNIT_VALIDATION" (probably just update in unit functions)
# - REFACTOR to a atom -> unit -> substrate structure
# - Deal with modulation (put into unit params?)

from typing import Dict, Union, Tuple

import numpy as np
from copy import deepcopy
import pyphi
import networkx as nx
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from units.unit_functions import *

# LOCAL VARIABLES, used to validate the objects created.


PROGRESS_BAR_THRESHOLD = 2**10

UNIT_VALIDATION = {
    "gabor": dict(
        default_params=dict(preferred_states=[], floor=0.0, ceiling=1.0),
        function=gabor_gate,
    ),
    "sigmoid": dict(
        default_params=dict(
            input_weights=[],
            determinism=4,
            threshold=0.5,
            floor=0.0,
            ceiling=1.0,
            ising=True,
        ),
        function=sigmoid,
    ),
    "resonnator": dict(
        default_params=dict(
            input_weights=[],
            determinism=4,
            threshold=0.5,
            weight_scale_mapping=dict(),
            floor=0.0,
            ceiling=1.0,
        ),
        function=resonnator,
    ),
    "sor": dict(
        default_params=dict(
            pattern_selection=[], selectivity=10, floor=0.0, ceiling=1.0
        ),
        function=sor_gate,
    ),
    "mismatch_pattern_detector": dict(
        default_params=dict(
            pattern_selection=[], selectivity=10, floor=0.0, ceiling=1.0
        ),
        function=mismatch_pattern_detector,
    ),
    "copy": dict(default_params=dict(floor=0.0, ceiling=1.0), function=copy_gate),
    "and": dict(default_params=dict(floor=0.0, ceiling=1.0), function=and_gate),
    "or": dict(default_params=dict(floor=0.0, ceiling=1.0), function=or_gate),
    "xor": dict(default_params=dict(floor=0.0, ceiling=1.0), function=xor_gate),
    "weighted_mean": dict(
        default_params=dict(weights=[], floor=None, ceiling=None),
        function=weighted_mean,
    ),
    "democracy": dict(
        default_params=dict(floor=None, ceiling=None), function=democracy
    ),
    "majority": dict(default_params=dict(floor=None, ceiling=None), function=majority),
    "mismatch_corrector": dict(
        default_params=dict(floor=0.0, ceiling=1.0, bias=0.0),
        function=mismatch_corrector,
    ),
    "modulated_sigmoid": dict(
        default_params=dict(
            input_weights=[],
            determinism=4,
            threshold=0.5,
            modulation={"modulator": tuple([]), "determinism": 0.0, "threshold": 0.0},
            floor=0.0,
            ceiling=1.0,
        ),
        function=modulated_sigmoid,
    ),
    "stabilized_sigmoid": dict(
        default_params=dict(
            input_weights=[],
            determinism=4,
            threshold=0.5,
            modulation={
                "modulator": tuple([]),
                "determinism": 0.0,
                "threshold": 0.0,
                "selectivity": 0.0,
            },
            floor=0.0,
            ceiling=1.0,
        ),
        function=stabilized_sigmoid,
    ),
    "biased_sigmoid": dict(
        default_params=dict(
            input_weights=[], determinism=4, threshold=0.5, floor=0.0, ceiling=1.0
        ),
        function=biased_sigmoid,
    ),
}


def reshape_to_md(tpm):
    N = int(np.log2(len(tpm)))
    try:
        return tpm.reshape([2] * N + [1], order="F").astype(float)
    except:
        return pyphi.convert.to_md(tpm)


class TPMDict(dict):
    def __init__(self, unit):
        self.unit = unit
    
    def __missing__(self, state):
        value = self.yield_tpm(state)
        return value
    
    def yield_tpm(self, state):
        
        # if unit state and input state
        if type(state[0]) == tuple: 
            u_state = (state[self.unit.index],)
            i_state = tuple([state[i] for i in self.unit.inputs])
            # check if there already is a "all unit tpm" representation (TO BE REMOVED)
        if hasattr(self.unit,'all_unit_tpm') and (u_state, i_state) in self.unit.all_unit_tpm:
            return self.unit.all_unit_tpm[(u_state, i_state)]
        else:
            ixs = (self.unit.index,) + self.unit.inputs
            state = u_state + i_state
            substate = tuple(
                [
                    0 if i not in ixs else state[ixs.index(i)]
                    for i in range(len(self.unit.substrate_state))
                ]
            )
            self.unit.input_state = i_state
            self.unit.set_unit_tpm(substate)
            return self.unit.tpm


class Unit:
    """A unit that can constitute a substrate.

    Represents the unit and holds auxilary data about it to construct a
    substrate.

    Args:
        index (int): Integer identifier of the unit.
        inputs (tuple[int]): A tuple of integers specifying the identities of
            the units providing iniputs.

    Keyword Args:
        params (dict or np.ndarray): Contains a specification of the mechanism
            of the unit. Given either explicitly by a TPM (ndarray) or implicitly as
            specifications in a dict. The supported unit tpyes can be found in the
            UNIT_VALIDATION object.
        label (str): Human readable name for the unit
        state (int): Indicates the current state of the unit: 1 means it is ON,
            0 OFF. For now, only binary units are supported.
        input_state (tuple[int]): Binary tuple (consisting of 1s and 0s) that
            indicates the current state of the inputs. This should be understood
            as the state of the inputs to the unit, and not the past state of
            the units that provide input to the unit.
    """

    def __init__(
        self,
        index: int,
        state: Union[int, Tuple[int,]],
        inputs: tuple[int],
        input_state: tuple[int],
        mechanism: str,
        params: dict,
        label: str = None,
    ):
        # Storing the parameters
        self._params = params

        # This unit's index in the list of units.
        self._index = index
        
        # Node labels used in the system
        self._label = label
        
        # Store the type of unit
        self._mechanism = mechanism

        # List of indices that input to the unit (one pr mechanism).
        self._inputs = inputs

        # Set unit state
        self._state = state
        
        #  and input state
        self._input_state = input_state
        
        # set tpm (None means I am not overriding with a substrate state)
        #self._tpm = None

        # validating unit
        assert self.validate(), "Unit did not pass validation"

        
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params: dict):
        self._params = params
        
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index: int):
        self._index = index
        
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, label: str=None):
        
        if not label == None:
            self._label = label
        else:
            self._label = str(self.index)
        
    @property
    def mechanism(self):
        return self._mechanism
    
    @mechanism.setter
    def mechanism(self, mechanism: str):
        self._mechanism = mechanism
        
    @property
    def inputs(self):
        return self._inputs
    
    @inputs.setter
    def inputs(self, inputs: tuple[int]):
        self._inputs = inputs
        
    @property
    def state(self):
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
        return self._state
        
    @property
    def input_state(self):
        return self._input_state
    
    @input_state.setter
    def input_state(self, input_state: tuple[int]):
        if not all([s in (0, 1) for s in input_state]):
            raise ValueError("all input states must be 0 or 1")
        self._input_state = input_state
        
    @property
    def tpm(self):
        func = UNIT_VALIDATION[self.mechanism]["function"]
        self._tpm = func(self, **self.params)
        return self._tpm
    
    # this might be bad (neet to use as unit.tpm = substrate_state)
    @tpm.setter
    def tpm(self, substrate_state: tuple[int]=None):
        
        if not substrate_state==None:
            self.state = substrate_state[self.index]
            self.input_state = tuple([
                substrate_state[i] for i in self.inputs
            ])
        return self.tpm
    
    
    def all_tpm(self, substrate_state: tuple[int]):
        self.state = substrate_state[self.index]
        self.input_state = tuple([
            substrate_state[i] for i in self.inputs
        ])
        return self.tpm
    
    def set_unit_tpm(self, substrate_state=None):
        """np.ndarray: The unit TPM.

        A multidimensional array, containing the probabilities for the unit
        turning ON, given the possible states of the inputs.
        """
        if not self.type == "composite":
            # if params include a specification of a unit
            func = UNIT_VALIDATION[self.type]["function"]
            self.tpm = func(self, **self.params)
            
        else:
            # TODO fix after refactoring (code from old version)
            c_unit = self.params["CompositeUnit"]
            self.tpm = CompositeUnit(
                c_unit.index,
                c_unit.all_inputs,
                params=c_unit.params,
                label=c_unit.label,
                state=c_unit.state,
                input_state=c_unit.input_state,
                mechanism_combination=c_unit.mechanism_combination,
                substrate_state=self.substrate_state
                if substrate_state is None
                else substrate_state,
                substrate_indices=self.substrate_indices,
                modulation=self.modulation,
                all_tpm=True,
            ).tpm
            self.type = "Composite unit: {}".format(
                [unit.type for unit in c_unit.units]
            )
            
            
    def validate(self):
        """Return whether the specifications for the unit are valid.

        The checks for validity are defined in the local UNIT_VALIDATION object.
        """
        if type(self.params) == dict:
            unit_type = self.mechanism

            if not unit_type == "composite":
                assert (
                    unit_type in UNIT_VALIDATION.keys()
                ), "Unit mechanism '{}' is not valid".format(unit_type)
                # check that all required params are present
                for key, value in UNIT_VALIDATION[unit_type]["default_params"].items():
                    if not key in self.params:
                        print(
                            "Unit {} missing {} params, defaulting to {}".format(
                                self.label, key, value
                            )
                        )
                        self.params[key] = value

        return True

    def __repr__(self):
        return "Unit(type={}, label={}, state={})".format(
            self.mechanism, self.label, self.state
        )

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same subsystem and have the
        same index (their TPMs must be the same in that case, so this method
        doesn't need to check TPM equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (
            self.index == other.index
            and np.array_equal(self.tpm, other.tpm)
            and self.state == other.state
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def __copy__(self):
        return Unit(
            self.index,
            self.inputs,
            params=self.params,
            label=self.label,
            state=self.state,
            input_state=self.input_state,
            tpm=self.tpm,
            substrate_state=self.substrate_state,
            substrate_indices=self.substrate_indices,
            modulation=self.modulation,
        )

    def get_all_tpms(self):

        if True:
            self.all_tpm = AllTPMDict(self)
        else:
            orig_unit_state = self.state
            orig_input_state = self.input_state
            ixs = (self.index,) + self.inputs

            self.all_unit_tpm = dict()
            for u_state in pyphi.utils.all_states(1):
                self.state = u_state
                all_states = list(pyphi.utils.all_states(len(self.inputs)))
                for i_state in all_states:
                    state = u_state + i_state
                    substate = tuple(
                        [
                            0 if i not in ixs else state[ixs.index(i)]
                            for i in range(len(self.substrate_state))
                        ]
                    )
                    self.input_state = i_state
                    self.set_unit_tpm(substate)
                    self.all_unit_tpm[(u_state, i_state)] = self.tpm
            self.input_unit_tpm = self.all_unit_tpm

            self.all_tpm = AllTPMDict(self)

            '''
            for s_state in pyphi.utils.all_states(len(self.substrate_state)):
                u_state = (s_state[self.index],)
                i_state = tuple([s_state[i] for i in self.inputs])
                self.all_tpm[s_state] = self.all_unit_tpm[(u_state, i_state)]'''

            self.state = orig_unit_state
            self.input_state = orig_input_state

    def set_unit_tpm(self, substrate_state=None):
        """np.ndarray: The unit TPM.

        A multidimensional array, containing the probabilities for the unit
        turning ON, given the possible states of the inputs.
        """
        if hasattr(self, 'all_tpm') and substrate_state in self.all_tpm:
            self.tpm = self.all_tpm[substrate_state]
        else:
            if type(self.params) == np.ndarray:
                # if params just include a TPM
                self.tpm = self.params * 1.0

            elif type(self.params) == dict:
                if self.params["mechanism"] == "composite":
                    c_unit = self.params["CompositeUnit"]
                    self.tpm = CompositeUnit(
                        c_unit.index,
                        c_unit.all_inputs,
                        params=c_unit.params,
                        label=c_unit.label,
                        state=c_unit.state,
                        input_state=c_unit.input_state,
                        mechanism_combination=c_unit.mechanism_combination,
                        substrate_state=self.substrate_state
                        if substrate_state is None
                        else substrate_state,
                        substrate_indices=self.substrate_indices,
                        modulation=self.modulation,
                        all_tpm=True,
                    ).tpm
                    self.type = "Composite unit: {}".format(
                        [unit.type for unit in c_unit.units]
                    )

                else:
                    if (
                        not self.modulation == None
                        and not type(not self.modulation) == list
                    ):
                        if self.modulation["type"] == "actual":
                            self.actual_modulation()

                        elif self.modulation["type"] == "virtual":
                            self.virtual_modulation()

                    else:
                        # if params include a specification of a unit
                        func = UNIT_VALIDATION[self.params["mechanism"]]["function"]
                        self.tpm = func(self, **self.params["params"])

    def set_input_state(self, state):
        """The current state of the inputs.

        Args:
            state (tuple[int]): The state of the inputs to the unit

        A tuple of 0's and 1's (One pr input) indicating the current state of the inputs.
        """
        self.input_state = state

    def set_substrate_state(self, state):
        """The current state of the substrate.

        Args:
            state (tuple[int]): The state of the inputs to the unit

        A tuple of 0's and 1's (One pr input) indicating the current state of the inputs.
        """
        self.substrate_state = state
        self.set_state(self.get_substate((self.index,)))
        self.set_unit_tpm(substrate_state=state)

    def set_state(self, state):
        """The current state of the unit.

        Args:
            state (int): The state of the unit

        A number (0 or 1) indicating the current state of the unit
        """
        self.state = state

    def set_inputs(self, input_indices):
        """The inputs to the unit.

        Args:
            input_indices (tuple[int]): The indices of inputs to the unit

        A tuple of integers (One pr input) indicating the identity of the inputs.
        """
        self.inputs = input_indices

    def set_type(self, type_description):
        """The type unit.

        Args:
            type_description (str): The type of unit

        Just a descriptive string used to identify the kind of unit.
        """
        self.type = type_description

    def get_substate(self, unit_indices):

        return tuple([self.substrate_state[i] for i in unit_indices])

    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index


class CompositeUnit:
    """A composite unit, consisting of multiple interacting sub-units.
    It contains within it a Unit object constructed by combining the sub-units according to the user specifications.

    Args:
        index (int): Integer identifier of the unit.
        inputs (list[tuple[int]]): A list of tuples of integers specifying the
            identities of the units providing iniputs. The number of elements in
            the list must match the number of units the CompositeUnit is
            composed of.

    Keyword Args:
        params (list[dict] or list[np.ndarray]): The list contains
            specifications of the mechanism of each of the subunits. Given
            either explicitly by a TPM (ndarray) or implicitly as specifications
            in a dict.  The supported unit tpyes can be found in the
            UNIT_VALIDATION object.
        label (str): Human readable name for the unit
        state (int): Indicates the current state of the unit: 1 means it is ON,
            0 OFF. For now, only binary units are supported.
        input_state (list[tuple[int]]): A list of binary tuples (consisting of
            1s and 0s) that indicates the current state of the inputs to each of the
            subunits. This should be understood as the present state of the inputs
            to the subunit, and not the past state of the units that provide input
            to the unit.
        mechanism_combination (list[dict] or list[np.ndarray]): Like the
            "params" kwarg. This kwarg provides the specifications for how the
            activations of the distinct subunits combine to yield an output for the
            CompositUnit as a whole.
    """

    def __init__(
        self,
        index,
        inputs,
        params=None,
        label=None,
        state=None,
        input_state=None,
        mechanism_combination=None,
        substrate_state=None,
        substrate_indices=None,
        modulation=None,
        all_tpm=True,
    ):

        # Construct Units for each of the subunits constituting the CompositeUnit
        if input_state == None:
            input_state = [input_state] * len(inputs)
        if modulation == None:
            modulation = [modulation] * len(inputs)
        if substrate_state == None and substrate_indices == None:
            substrate_indices = range(max([i for ix in inputs for i in ix]) + 1)

        units = [
            Unit(
                index,
                input_ixs,
                params=param,
                label=label,
                state=state,
                input_state=input_s,
                substrate_state=substrate_state,
                substrate_indices=substrate_indices,
                modulation=mod,
                all_tpm=all_tpm,
            )
            for input_ixs, input_s, param, mod in zip(
                inputs, input_state, params, modulation
            )
        ]
        self.units = units

        # Store the list of indices that input to the unit (one pr subunit).
        # update self.inputs to include modulators, if modulation is virtual
        all_inputs = []
        for in_ix, mod in zip(inputs, modulation):
            if not mod == None and mod["type"] == "virtual":
                all_inputs.append(
                    in_ix + tuple([m for m in mod["modulators"] if m not in in_ix])
                )
            else:
                all_inputs.append(in_ix)

        self.inputs = all_inputs
        self.modulation = modulation

        self.substrate_state = units[0].substrate_state
        self.substrate_indices = units[0].substrate_indices

        # set the TPM defining the combination of subunit activations into a output state for the composite unit.
        self.set_mechanism_combination(mechanism_combination)

        # get composite tpm based on the subunits and the mechanism combination
        composite_tpm = self.combine_unit_tpms([unit.tpm for unit in units])

        # Storing some properties of the unit
        self.type = [unit.type for unit in units]
        self.label = label
        self.state = state
        self.index = index
        self.params = params
        self.input_state = input_state

        # create the Unit object
        self.Unit = Unit(
            index,
            self.composite_inputs(units),
            params={"mechanism": "composite", "CompositeUnit": self},
            label=label,
            state=state,
            input_state=self.composite_input_state(units, input_state),
            tpm=composite_tpm,
            substrate_state=substrate_state,
            substrate_indices=substrate_indices,
            modulation=modulation,
            all_tpm=all_tpm,
        )
        self.Unit.set_type(self.__repr__())

        self.tpm = self.Unit.tpm

    def __repr__(self):
        return "CompositeUnit(type={}, label={}, state={})".format(
            self.type, self.label, self.state
        )

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same subsystem and have the
        same index (their TPMs must be the same in that case, so this method
        doesn't need to check TPM equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (
            self.index == other.index
            and np.array_equal(self.tpm, other.tpm)
            and self.state == other.state
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def set_mechanism_combination(self, mechanism_combination):
        """np.ndarray: The composite unit TPM.

        Args:
            mechanism_combination (dict or np.ndarray): A specification of the
                way the composite unit should translate activations of its subunits
                into a single output from the composite unit.

        A multidimensional array, containing the probabilities for the unit
        turning ON, given the possible activations of the subunits.
        """

        num_subunits = len(self.inputs)
        if mechanism_combination is None:
            # if no function is given for how to combine mechanisms, OR logic is used.
            # that is, the activation of any number of subunits results in activation of the composite unit.
            print("No combination defined, using OR-logic")
            self.mechanism_combination = np.array(
                [
                    0 if i == 0 else 1
                    for i in range(len(list(pyphi.utils.all_states(num_subunits))))
                ]
            ).reshape((2,) * num_subunits)

        elif type(mechanism_combination) == dict:
            # if specification is given as a dict, we create the TPM as if a Unit specification.
            func = UNIT_VALIDATION[mechanism_combination["mechanism"]]["function"]
            self.mechanism_combination = func(
                self, 0, **mechanism_combination["params"]
            )

        elif type(mechanism_combination) == np.ndarray:
            # if specification is explicit as a TPM, it is just set as is
            self.mechanism_combination = mechanism_combination

        elif type(mechanism_combination) == str:
            # if specification is explicit as a TPM, it is just set as is
            self.mechanism_combination = mechanism_combination

    def combine_unit_tpms(self, tpms):
        """
        Returns:
            tpm (np.ndarray): The composite unit TPM, specifying the activation probability (probability to turn ON) given all possible input states.

        Args:
            tpms (list[np.ndarray]): A specification of the way the composite unit should translate activations of its subunits into a single output from the composite unit.

        A multidimensional array, containing the probabilities for the unit
        turning ON, given the possible activations of the subunits.
        """

        # getting all the possible states of the individual mechanisms
        # constituting the unit
        MECHANISM_ACTIVATIONS = list(pyphi.utils.all_states(len(tpms)))

        # a local function that returns the unit activation given some state of its constituent mechanisms
        def combined_activation_probability(
            activation_probabilities, mechanism_combination
        ):
            """Return the probability of the composite unit activating given the
            activation of subunits due to a particular input state.

            Args:
                tpms (list[np.ndarray]): A specification of the way the
                    composite unit should translate activations of its subunits into
                    a single output from the composite unit.

            Returns:
                tpm (np.ndarray): The composite unit TPM, specifying the
                activation probability (probability to turn ON) given all
                possible input states.

            A multidimensional array, containing the probabilities for the unit
            turning ON, given the possible activations of the subunits.
            """

            unit_activation = np.sum(
                [
                    np.prod(
                        [
                            p if s == 1 else 1 - p
                            for p, s in zip(activation_probabilities, state)
                        ]
                    )
                    * mechanism_combination[state]
                    for state in MECHANISM_ACTIVATIONS
                ]
            )

            return unit_activation

        # Expand all TPMs to be the length of the unit's full set of inputs
        # all_inputs = sorted(tuple(set([i for indices in self.inputs for i in indices])))
        all_inputs = self.composite_inputs(
            self.units
        )  # tuple(sorted(set([i for indices in self.inputs for i in indices]))) composite
        all_input_states = list(pyphi.utils.all_states(len(all_inputs)))

        expanded_tpms = []
        for tpm, inputs in zip(tpms, self.inputs):
            # get mechanism activation probabilities for all potential input states
            mechanism_activation = []
            for state in all_input_states:
                P = tpm[
                    self.get_subset_state(
                        state, tuple([all_inputs.index(i) for i in inputs])
                    )
                ]
                if type(P) == np.ndarray:
                    P = P[0]
                mechanism_activation.append(P)

            expanded_tpms.append(mechanism_activation)

        # make the TPMs into an array of correct shape
        expanded_tpms = np.array(expanded_tpms).T

        # combine subunit TPMs into composite unit tpm
        if not type(self.mechanism_combination) == str:
            tpm = reshape_to_md(
                np.array(
                    [
                        [
                            combined_activation_probability(
                                activation_probabilities, self.mechanism_combination
                            )
                        ]
                        for activation_probabilities in expanded_tpms
                    ]
                )
            )
        else:

            # here we can add specific cases of "string specified unit combinations" that do not adhere to the above logic of using another TPM to combine them
            # For example, we can have one where the Unit matches the output of the "strongest" subunit.
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

        # set some final properties
        self.all_inputs = self.inputs * 1
        self.inputs = tuple(all_inputs)
        return tpm

    def composite_inputs(self, units):
        """tuple[int]: The IDs of units providing inputs to the composite units.

        Args:
            units (list[Unit]): all the units that constitute the composite unit.

        """

        inputs = []
        for unit in units:
            for ix in unit.inputs:
                if ix not in inputs:
                    inputs.append(ix)

        return tuple(inputs)

    def composite_input_state(self, units, input_states):
        """tuple[int]: the input state to the composite unit as a whole (must be congruent among shared inputs to subunits).

        Args:
            units (list[Unit]): all the units that constitute the composite unit.

        """

        # first get the IDs of units that input to subunits
        composite_inputs = self.composite_inputs(units)

        # get/set the state of the inputs (the state recorded locally, by the unit)
        # TODO: consider whether subunits can have incongruent inputs from the same unit (noisy channels?)

        if any([i_s == None for i_s in input_states]):
            input_state = self.get_subset_state(self.substrate_state, composite_inputs)
        else:
            states = dict()
            for unit, input_state in zip(units, input_states):
                for i, i_s in zip(unit.inputs, input_state):
                    states[i] = i_s

            input_state = tuple([states[i] for i in composite_inputs])

        # # go through units to pick out their input states. If the same unit appears multiply, then the state is forced to be whichever state appeared first.
        # input_unit_states = dict()
        # for unit in units:
        #     for input_unit, input_state in zip(unit.inputs, unit.input_state):
        #         if input_unit in input_unit_states:
        #             # check that input state of repeated unit is congruent
        #             if not input_unit_states[input_unit] == input_state:
        #                 print(
        #                     "incongruency in input state of unit {}, forcing state to {}".format(
        #                         unit.label, input_unit_states[input_unit]
        #                     )
        #                 )
        #         else:
        #             # storing the state of the input unit
        #             input_unit_states[input_unit] = input_state

        # # create the state tuple by looking through the dict constructed before
        # input_state = tuple(
        #     [input_unit_states[input_unit] for input_unit in composite_inputs]
        # )
        return input_state

    # THIS IS WRONG! IT DOES NOT GET THE RIGHT STATE BECAUSE IT DOESNT CARE
    # ABOUT THE FULL SET OF INDICES
    def get_subset_state(self, state, subset_indices):
        """tuple[int (binary)]: the state of a subset of indices.

        Args:
            state (tuple[int(binary)]): The (binary) state of the full set of inputs.
            subset_indices (tuple[int]): The indices (relative to the state) for the subset.
        """
        return tuple([state[ix] for i, ix in enumerate(subset_indices)])

    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index


class Substrate:
    """A model of a substrate, constituted of units

    Args:
        units (list): a list of unit objects

    Attributes:
        tpm (np.ndarray): The node TPM is an array w
        cm (np.ndarray): The node TPM is an array w
        state (tuple)
    """

    def __init__(
        self, units: list[Unit], state: tuple[int] = None, all_tpm: bool = False
    ):

        # store units
        self.units = units

        # set the state of the substrate
        self.set_state(state)

        # Set indices of the inputs.
        self.node_indices = tuple([unit.index for unit in units])

        # Node labels used in the system
        self.node_labels = pyphi.labels.NodeLabels(
            [unit.label for unit in units], self.node_indices
        )

        self.all_tpm = dict()
        if all_tpm:
            self.set_all_tpms()

        self.set_tpm(state)

        self.dynamic_tpm = self.get_dynamic_tpm()

        # substrate_tpm = []
        # if self.state == None and explicit_tpm:
        #     # running through all possible substrate states
        #     for state in tqdm(list(pyphi.utils.all_states(len(units)))):

        #         # setting auxillary substrate state
        #         self.state = state

        #         # Force the state of (state dependent) units
        #         # and their inputs to match the substrate state
        #         units = self.validate_unit_states(units)

        #         # Combine Unit TPMs to substrate TPM
        #         # adding relevant row from the matrix to the substrate tpm
        #         substrate_tpm.append(self.combine_unit_tpms(units, state))

        #     self.tpm = reshape_to_md(np.array(substrate_tpm))
        #     self.state = None

        #     # validating unit
        #     assert self.validate(), "Substrate did not pass validation"

        # else:
        #     # running through all possible substrate states
        #     for past_state in tqdm(list(pyphi.utils.all_states(len(units)))):

        #         # check that the state of (state dependent) units
        #         # and their inputs match the substrate state
        #         units = self.validate_unit_states(units)

        #         # Combine Unit TPMs to substrate TPM
        #         # adding relevant row from the matrix to the substrate tpm
        #         substrate_tpm.append(self.combine_unit_tpms(units, past_state))

        #     self.tpm = reshape_to_md(np.array(substrate_tpm))

        #     # validating unit
        #     assert self.validate(), "Substrate did not pass validation"

        # self.tpm = self.tpm  # to check for issues due to rounding
        # storing the units

        self.units = units

        # This node's index in the list of nodes.
        self.create_cm(units)

    def validate_unit_states(self, units):

        new_units = []
        for unit in units:
            if (
                type(unit.tpm) == np.ndarray
                or type(unit.params) == dict
                or type(unit.params) == np.ndarray
            ):

                # CHECK ALL OF THIS!!!
                substrate_unit_state = self.get_subset_state((unit.index,))
                substrate_input_state = self.get_subset_state(unit.inputs)

                if (
                    not unit.state == substrate_unit_state
                    or not unit.input_state == substrate_input_state
                ):
                    # print(
                    #     "Redefining unit {} to match substrate state {},".format(
                    #         unit.label, self.state
                    #     )
                    # )

                    # redefining unit
                    if (
                        type(unit.params) == dict
                        and unit.params["mechanism"] == "composite"
                    ):
                        substrate_input_state = [
                            self.get_subset_state(u.inputs)
                            for u in unit.params["CompositeUnit"].units
                        ]
                        unit = CompositeUnit(
                            unit.index,
                            unit.params["CompositeUnit"].all_inputs,
                            params=unit.params["CompositeUnit"].params,
                            label=unit.label,
                            state=substrate_unit_state,
                            input_state=substrate_input_state,
                            substrate_state=self.state,
                            substrate_indices=self.node_indices,
                            mechanism_combination=unit.params[
                                "CompositeUnit"
                            ].mechanism_combination,
                            modulation=unit.params["CompositeUnit"].modulation,
                        ).Unit
                    elif unit.type == "TPM":
                        unit = Unit(
                            unit.index,
                            unit.inputs,
                            params=unit.tpm,
                            label=unit.label,
                            state=substrate_unit_state,
                            input_state=substrate_input_state,
                            substrate_state=self.state,
                            substrate_indices=self.node_indices,
                            modulation=unit.modulation,
                        )

                    else:
                        unit = Unit(
                            unit.index,
                            unit.inputs,
                            params=unit.params,
                            label=unit.label,
                            state=substrate_unit_state,
                            input_state=substrate_input_state,
                            substrate_state=self.state,
                            substrate_indices=self.node_indices,
                            modulation=unit.modulation,
                        )

            new_units.append(unit)

        return new_units

    def update_substrate_tpm(self, state):

        orig_state = self.state
        dynamic_tpm = []
        # running through all possible substrate states
        all_states = list(pyphi.utils.all_states(len(self.units)))

        if all([len(u.all_tpm.keys()) > 0 for u in self.units]):

            for state in (
                tqdm(all_states)
                if len(all_states) > PROGRESS_BAR_THRESHOLD
                else all_states
            ):

                # Combine Unit TPMs to substrate TPM
                # adding relevant row from the matrix to the substrate tpm
                dynamic_tpm.append(self.combine_unit_tpms(self.units, state, state))

            self.state = orig_state
            return reshape_to_md(np.array(dynamic_tpm))

    def get_dynamic_tpm(self):
        orig_state = self.state
        dynamic_tpm = []
        # running through all possible substrate states
        all_states = list(pyphi.utils.all_states(len(self.units)))

        if True:#all([len(u.all_tpm.keys()) > 0 for u in self.units]):

            for state in (
                tqdm(all_states)
                if len(all_states) > PROGRESS_BAR_THRESHOLD
                else all_states
            ):

                # Combine Unit TPMs to substrate TPM
                # adding relevant row from the matrix to the substrate tpm
                dynamic_tpm.append(self.combine_unit_tpms(self.units, state, state))

            self.state = orig_state
            return reshape_to_md(np.array(dynamic_tpm))

        else:
            print("Consider putting all tpms in units")
            for state in (
                tqdm(all_states)
                if len(all_states) > PROGRESS_BAR_THRESHOLD
                else all_states
            ):

                # setting auxillary substrate state

                self.update_state(state)
                units = self.validate_unit_states(units)

                # Combine Unit TPMs to substrate TPM
                # adding relevant row from the matrix to the substrate tpm
                dynamic_tpm.append(self.combine_unit_tpms(units, state, state))

            self.state = orig_state
            return reshape_to_md(np.array(dynamic_tpm))

    def validate(self):
        return True

    def get_subset_state(self, subset_indices):
        return tuple([self.state[i] for i in subset_indices])

    def __repr__(self):
        return "Substrate({})".format("|".join(self.node_labels))

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.units)

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same subsystem and have the
        same index (their TPMs must be the same in that case, so this method
        doesn't need to check TPM equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (
            self.index == other.index
            and np.array_equal(self.tpm, other.tpm)
            and self.state == other.state
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def update_state(self, state):
        if all([state in u.all_tpm for u in self.units]):
            self.set_tpm(state)
            self.all_tpm[state] = self.tpm
            return self

        else:
            substrate = Substrate(self.units, state=state)
            self.all_tpm[state] = substrate.tpm
            return self

    def combine_unit_tpms(self, units, past_state, present_state):

        unit_response = []
        # going through each unit to find its state-dependent activation probability
        for unit in units:
            if True:#present_state in unit.all_tpm:
                unit_response.append(
                    float(
                        unit.all_tpm[present_state][
                            tuple([past_state[i] for i in unit.inputs])
                        ]
                    )
                )
            else:
                unit_response.append(
                    float(unit.tpm[tuple([past_state[i] for i in unit.inputs])])
                )
        # # sorting the activation probability in an ascending order
        # full_tpm = np.array(unit_response)
        # substrate_tpm = []
        # for ix in sorted(set(self.node_indices)):
        #     substrate_tpm.append(full_tpm[ix])

        # return substrate_tpm
        return unit_response

    def create_cm(self, units):
        cm = np.zeros((len(self.node_indices), len(self.node_indices)))

        for unit in units:
            cm[unit.inputs, unit.index] = 1

        self.cm = cm

        return

    def set_state(self, state):
        self.state = state

    def set_tpm(self, state):

        if state in self.all_tpm:
            self.tpm = self.all_tpm[state]
        elif all([state in u.all_tpm for u in self.units]):
            # set the state of the substrate
            self.set_state(state)

            all_states = list(pyphi.utils.all_states(len(self.units)))
            substrate_tpm = []
            for past_state in all_states:
                # Combine Unit TPMs to substrate TPM
                # adding relevant row from the matrix to the substrate tpm
                substrate_tpm.append(
                    self.combine_unit_tpms(self.units, past_state, state)
                )

            self.tpm = reshape_to_md(np.array(substrate_tpm))

            self.all_tpm[state] = self.tpm

        elif state is None:
            self.tpm = None
        else:
            # set the state of the substrate
            self.set_state(state)
            self.units = self.validate_unit_states(self.units)

            all_states = list(pyphi.utils.all_states(len(self.units)))
            substrate_tpm = []
            for past_state in all_states:
                # Combine Unit TPMs to substrate TPM
                # adding relevant row from the matrix to the substrate tpm
                substrate_tpm.append(
                    self.combine_unit_tpms(self.units, past_state, state)
                )

            self.tpm = reshape_to_md(np.array(substrate_tpm))

            self.all_tpm[state] = self.tpm

    def set_all_tpms(self):

        orig_state = self.state
        # running through all possible substrate states
        all_states = list(pyphi.utils.all_states(len(self.units)))
        for present_state in (
            tqdm(all_states) if len(all_states) > PROGRESS_BAR_THRESHOLD else all_states
        ):
            # setting auxillary substrate state
            self.set_state(present_state)
            self.set_tpm(present_state)

        self.state = orig_state

    def get_network(self):
        if self.state == None:
            return pyphi.network.Network(self.dynamic_tpm, self.cm, self.node_labels)
        else:
            return pyphi.network.Network(self.tpm, self.cm, self.node_labels)

    def get_subsystem(self, state=None, nodes=None):
        if nodes == None:
            nodes = self.node_indices

        # make sure the substrate is state_specific
        if state == None and not self.state == None:
            return pyphi.subsystem.Subsystem(self.get_network(), self.state, nodes)
        elif not state == None:

            if state in self.all_tpm:
                return pyphi.subsystem.Subsystem(
                    pyphi.network.Network(
                        self.all_tpm[state], self.cm, self.node_labels
                    ),
                    state,
                    nodes,
                )
            else:

                self = self.update_state(state)
                subsystem = pyphi.subsystem.Subsystem(self.get_network(), state, nodes)
                self = self.update_state(state)
                return subsystem

    def get_model(self, state=None):
        rows, cols = np.where(self.cm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_edges_from(edges)
        gr.graph["params"] = "test"

        return gr

    def plot_model(self, state=None):

        gr = self.get_model(state=None)
        nodes = gr.nodes

        if state == None:
            state = (1,) * len(self.node_indices)

        nx.draw(
            gr,
            node_size=500,
            labels={i: l for i, l in enumerate(self.node_labels)},
            with_labels=True,
            node_color=["blue" if state[s] == 1 else "gray" for s in nodes],
        )
        plt.show()

    def simulate(self, initial_state=None, timesteps=1000, clamp=False, evoked=False):

        rng = np.random.default_rng(0)
        if not clamp:
            # Just simulating from initial state
            if initial_state == None:
                initial_state = tuple(rng.integers(0, 2, len(self)))
            states = [initial_state]
            for t in range(timesteps):
                P_next = self.dynamic_tpm[states[-1]]
                comparison = rng.random(len(initial_state))
                states.append(
                    tuple([1 if P > c else 0 for P, c in zip(P_next, comparison)])
                )

        else:
            # simulating with some units clamped to a state
            clamped_ix = list(clamp.keys())[0]
            clamped_state = list(clamp.values())[0]
            if not evoked:

                if initial_state == None:
                    initial_state = list(rng.integers(0, 2, len(self)))

                for ix, s in zip(clamped_ix, clamped_state):
                    initial_state[ix] = s
                states = [tuple(initial_state)]

                for t in range(timesteps):
                    P_next = self.dynamic_tpm[states[-1]]
                    comparison = rng.random(len(initial_state))

                    state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                    for ix, s in zip(clamped_ix, clamped_state):
                        state[ix] = s
                    states.append(tuple(state))
            elif type(evoked) == int:
                print('hey',flush=True)

                states = []
                for initial_state in tqdm(
                    list(pyphi.utils.all_states(len(self) - len(clamped_ix)))
                ):
                    initial_state = list(initial_state)
                    
                    for i in clamped_ix:
                        initial_state.insert(i,np.random.randint(0,2))
                        
                    trial = [tuple(initial_state)]

                    for t in range(timesteps):
                        P_next = self.dynamic_tpm[trial[-1]]
                        comparison = rng.random(len(initial_state))

                        state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                        if t >= evoked:
                            for ix, s in zip(clamped_ix, clamped_state):
                                state[ix] = s
                        trial.append(tuple(state))

                    states.append(trial)
            elif type(evoked) == list:

                states = []
                for initial_state in tqdm(
                    list(pyphi.utils.all_states(len(self) - len(clamped_ix)))
                ):
                    initial_state = list(initial_state)
                    
                    for i in clamped_ix:
                        initial_state.insert(i,np.random.randint(0,2))
                        
                    trial = [tuple(initial_state)]

                    for t in range(timesteps):
                        P_next = self.dynamic_tpm[trial[-1]]
                        comparison = rng.random(len(initial_state))

                        state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                        if t >= evoked[0] and t < evoked[1]:
                            for ix, s in zip(clamped_ix, clamped_state):
                                state[ix] = s
                        trial.append(tuple(state))

                    states.append(trial)
                    
            elif evoked == 'all':

                states = [(0,)*len(self)]
                
                for clamp in tqdm(
                    list(pyphi.utils.all_states(len(clamped_ix)))
                ):
                    
                    for t in range(timesteps):
                        P_next = self.dynamic_tpm[states[-1]]
                        comparison = rng.random(len(self))

                        state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                        for ix, s in zip(clamped_ix, clamp):
                            state[ix] = s
                        
                        states.append(tuple(state))
                        
            else:

                states = []
                for initial_state in tqdm(
                    list(pyphi.utils.all_states(len(self) - len(clamped_ix)))
                ):
                    initial_state = list(initial_state)
                    for i, s in zip(clamped_ix, clamped_state):
                        initial_state.insert(i, s)
                    trial = [tuple(initial_state)]

                    for t in range(timesteps):
                        P_next = self.dynamic_tpm[trial[-1]]
                        comparison = rng.random(len(initial_state))

                        state = [1 if P > c else 0 for P, c in zip(P_next, comparison)]
                        for ix, s in zip(clamped_ix, clamped_state):
                            state[ix] = s
                        trial.append(tuple(state))

                    states.append(trial)

        return states

    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index


class System(pyphi.subsystem.Subsystem):
    def __init__(
        self,
        units: list[Unit],
        substrate_ixs: tuple[int],
        substrate_state: tuple[int],
        system_ixs: tuple[int],
    ):
        self.units = units
        self.state = tuple([unit.state[0] for unit in self.units])
        self.node_indices = tuple([unit.index for unit in self.units])
        self.node_labels = tuple([unit.label for unit in self.units])

        print(list(self.units[0].all_tpm.keys())[0], flush=True)
        system_units = self.subsystem_units(substrate_ixs, substrate_state, system_ixs)

        print(list(system_units[0].all_tpm.keys())[0], flush=True)
        system_state = tuple([unit.state[0] for unit in system_units])
        self.substrate = Substrate(system_units, state=system_state)
        self.subsystem = self.substrate.get_subsystem()

        self.units = system_units
        self.state = tuple([unit.state[0] for unit in self.units])
        self.node_indices = tuple([unit.index for unit in self.units])
        self.node_labels = tuple([unit.label for unit in self.units])

    def __repr__(self):
        return "System {} in {}".format(
            "|".join([u.label for u in self.units]),
            "".join([str(u.state[0]) for u in self.units]),
        )

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.units)

    def subsystem_units(
        self,
        substrate_ixs: tuple[int],
        substrate_state: tuple[int],
        system_ixs: tuple[int],
    ):
        # NOTE: add checks to make sure all indices are represented etc.

        # all indices inputing to the substrate units
        input_ixs = set([index for unit in self.units for index in unit.inputs])
        # prepare to marginalize out any input from units not present in the substrate
        marginalize_ixs = input_ixs - set(substrate_ixs)
        # prepare to condition unit tpms on inputs from outside the system
        condition_ixs = input_ixs - set(system_ixs) - marginalize_ixs

        substrate_units = []
        for u in tqdm(self.units, desc="Updating units"):

            if u.index in system_ixs:

                # copy unit to avoid destroying
                unit = deepcopy(u)

                self.update_unit_state(unit, substrate_ixs, substrate_state)

                # create tpm object
                tpm = pyphi.tpm.ExplicitTPM(unit.tpm)

                tpm = self.get_unit_tpm(
                    unit,
                    tpm,
                    substrate_ixs,
                    marginalize_ixs,
                    condition_ixs,
                    substrate_state,
                )

                # remaining "live" inputs
                live_inputs = tuple(
                    [
                        system_ixs.index(i)
                        for i in unit.inputs
                        if not i in marginalize_ixs.union(condition_ixs)
                    ]
                )

                # update all_tpms
                long_keys = []
                if type(unit.all_tpm) == dict:
                    new_all_tpm = dict()
                    for sub_state in pyphi.utils.all_states(len(substrate_ixs)):
                        long_state = tuple(
                            [
                                0
                                if not i in unit.inputs + (unit.index,)
                                else sub_state[substrate_ixs.index(i)]
                                for i in range(len(u.substrate_state))
                            ]
                        )
                        new_tpm = pyphi.tpm.ExplicitTPM(unit.all_tpm[long_state])
                        new_tpm = self.get_unit_tpm(
                            unit,
                            new_tpm,
                            substrate_ixs,
                            marginalize_ixs,
                            condition_ixs,
                            substrate_state,
                        )
                        new_all_tpm[sub_state] = new_tpm

                # get the updated unit
                substrate_unit = Unit(
                    system_ixs.index(unit.index),
                    live_inputs,
                    params=tpm,
                    tpm=tpm,
                    label=unit.label,
                    state=unit.state,
                    all_tpm=True,
                    inherited_tpm=new_all_tpm,
                )
                substrate_units.append(substrate_unit)

        return substrate_units

    def update_unit_state(
        self, unit: Unit, substrate_ixs: tuple[int], substrate_state: tuple[int]
    ):

        # check which substrate and state is stored in the unit
        old_substrate_state = self.state
        old_substrate_indices = self.node_indices

        # update substrate state with the state from kwargs
        new_substrate_state = tuple(
            [
                substrate_state[substrate_ixs.index(i)] if i in substrate_ixs else s
                for s, i in zip(old_substrate_state, old_substrate_indices)
            ]
        )

        # recreate unit with correct substrate state
        unit.set_substrate_state(new_substrate_state)

    def get_unit_tpm(
        self, unit, tpm, substrate_ixs, marginalize_ixs, condition_ixs, substrate_state
    ):

        # marginalize out non-substrate units from inputs
        non_substrate_inputs = [
            i for i, ix in enumerate(unit.inputs) if ix in marginalize_ixs
        ]
        tpm = tpm.marginalize_out(non_substrate_inputs)

        # condition on non-system units in inputs
        non_system_input_mapping = {
            unit.inputs.index(ix): s
            for ix, s in zip(substrate_ixs, substrate_state)
            if ix in condition_ixs and ix in unit.inputs
        }
        tpm = tpm.condition_tpm(non_system_input_mapping)

        # get tpm of only "live" inputs
        tpm = np.squeeze(tpm.tpm)[..., np.newaxis]

        return tpm

    def pyphi_kwargs(self, units: list[Unit]):

        unit_tpms = [
            np.concatenate((1 - unit.tpm, unit.tpm), axis=len(unit.tpm.shape) - 1)
            for unit in units
        ]
        node_labels = tuple([unit.label for unit in units])
        cm = np.array(
            [
                [1 if i in unit.inputs else 0 for i in range(len(units))]
                for unit in units
            ]
        )

        return dict(tpms=unit_tpms, node_labels=node_labels, cm=cm)

    def get_pyphi_kwargs(
        self,
        units: list[Unit] = None,
        substrate_indices: tuple[int] = None,
        substrate_state: tuple[int] = None,
        system_indices: tuple[int] = None,
    ):
        if units == None:
            units = self.units
        if substrate_indices == None:
            substrate_indices = tuple([unit.index for unit in units])
        if substrate_state == None:
            substrate_state = tuple([unit.state[0] for unit in units])
        if system_indices == None:
            system_indices = substrate_indices

        return self.pyphi_kwargs(
            self.subsystem_units(substrate_indices, substrate_state, system_indices)
        )
