# TODO:
# - Fix the time spent by caching the unit TPM's
# - allow for composite units
# - allow for non-binary units
# - make better "default values" for "UNIT_VALIDATION" (probably just update in unit functions)
# - REFACTOR to a atom -> unit -> substrate structure
# - Deal with modulation (put into unit params?)

from typing import Dict, Union, Tuple, List
from functools import cached_property

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
            func = UNIT_VALIDATION[self.mechanism]["function"]
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
        state: Union[int, Tuple[int,]] = None,
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


class Substrate:
    """A substrate of `Unit's.

    Attributes:
        units (List[Unit]): List of `Unit`s that make up the substrate.
        state (Tuple[int]): The binary state of the substrate units, as a tuple of ints.

    Properties:
        node_indices (Tuple[int]): The indices of the `Unit`s in the substrate, as a tuple of ints.
        node_labels (pyphi.labels.NodeLabels): The labels of the `Unit`s in the substrate.
        tpm (numpy.ndarray): The truth table of the substrate, as a numpy array of binary values.
        dynamic_tpm (numpy.ndarray): The dynamic truth table of the substrate, as a numpy array of binary values.
        cm (numpy.ndarray): The connectivity matrix of the substrate, as a numpy array of binary values.

    Methods:
        combine_unit_tpms(units, past_state, present_state): Combine the truth tables of `Unit`s in the substrate to compute the substrate's truth table.
        get_network(state=None): Get a `pyphi.network.Network` corresponding to the current or a specified state of the substrate.
        get_subsystem(state=None,nodes=None): Get a `pyphi.subsystem.Subsystem`

    """

    def __init__(self, units: List[Unit], state: Tuple[int] = None):
        """Initialize a new `Substrate` object. """

        self._units = units
        self.state = state
        
    @property
    def state(self):
        """Tuple[int]: The binary state of the substrate units, as a tuple of ints."""

        if self._state is not None:
            for s, u in zip(self._state, self._units):
                u.state = s
        else:
            self._state = tuple([u.state[0] for u in self._units])
        return self._state
    
    @state.setter
    def state(self, state):
        self._state = state
        return self.state
        
    @cached_property
    def node_indices(self):
        """Tuple[int]: The indices of the `Unit`s in the substrate, as a tuple of ints."""
        
        return tuple([unit.index for unit in self._units])

    @cached_property
    def node_labels(self):
        """pyphi.labels.NodeLabels: The labels of the `Unit`s in the substrate."""
        
        return pyphi.labels.NodeLabels(
            [unit.label for unit in self._units], self.node_indices
        )
    
    @property
    def tpm(self):
        """numpy.ndarray: The truth table of the substrate, as a numpy array of binary values."""
        
        # running through all possible input states
        all_states = list(pyphi.utils.all_states(len(self._units)))
        
        return reshape_to_md(np.array([
            self.combine_unit_tpms(self._units, past_state, self.state)
            for past_state in (
                tqdm(all_states)
                if len(all_states) > PROGRESS_BAR_THRESHOLD
                else all_states
            )
        ]))
            
    
    def state_dependent_tpm(self, state):
        """numpy.ndarray: The truth table of the substrate, as a numpy array of binary values."""
        self.state = state
        return self.tpm
    
    def combine_unit_tpms(self, units, past_state, present_state):
        """Combine the truth tables of `Unit`s in the substrate to compute the substrate's truth table.

        Args:
            units (List[Unit]): List of `Unit`s in the substrate.
            past_state (Tuple[int]): The binary state of the substrate at the previous time step, as a tuple of ints.
            present_state (Tuple[int]): The binary state of the substrate at the current time step, as a tuple of ints.

        Returns:
            List[float]: The combined truth table of the substrate, as a list of float values.
        """

        return [
            float(
                unit.state_dependent_tpm(present_state)[tuple([past_state[i] for i in unit.inputs])]
            )
            for unit in units
        ]
    
    @cached_property
    def dynamic_tpm(self):
        orig_state = self._state
        # running through all possible substrate states
        all_states = list(pyphi.utils.all_states(len(self._units)))
        
        dynamic_tpm = [
            self.combine_unit_tpms(self._units, state, state)
            for state in (
                tqdm(all_states)
                if len(all_states) > PROGRESS_BAR_THRESHOLD
                else all_states
            )
        ]
        self._state = orig_state
        return reshape_to_md(np.array(dynamic_tpm))
        
    @cached_property
    def cm(self):
        connectivity = np.zeros((len(self.node_indices), len(self.node_indices)))

        for unit in self._units:
            connectivity[unit.inputs, unit.index] = 1

        return connectivity

    def __repr__(self):
        return "Substrate({})".format("|".join(self.node_labels))

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self._units)

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

    def get_network(self, state: Tuple[int]=None):
        if self.state is None:
            return pyphi.network.Network(self.dynamic_tpm, self.cm, self.node_labels)
        elif not state is None:
            self.state = state
            return pyphi.network.Network(self.tpm, self.cm, self.node_labels)
            
        else:
            return pyphi.network.Network(self.tpm, self.cm, self.node_labels)

    def get_subsystem(
        self, 
        state: Tuple[int] = None, 
        nodes: Tuple[int] = None
    ):
        if nodes is None:
            nodes = self.node_indices

        # make sure the substrate is state_specific
        if state is None and self.state is not None:
            return pyphi.subsystem.Subsystem(self.get_network(), self.state, nodes)
        elif state is not None:
            return pyphi.subsystem.Subsystem(
                pyphi.network.Network(
                    self.state_dependent_tpm(state), self.cm, self.node_labels
                ),
                state,
                nodes,
            )

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

        print(list(self.units[0].state_dependent_tpm.keys())[0], flush=True)
        system_units = self.subsystem_units(substrate_ixs, substrate_state, system_ixs)

        print(list(system_units[0].state_dependent_tpm.keys())[0], flush=True)
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

                # update state_dependent_tpms
                long_keys = []
                if type(unit.state_dependent_tpm) == dict:
                    new_state_dependent_tpm = dict()
                    for sub_state in pyphi.utils.all_states(len(substrate_ixs)):
                        long_state = tuple(
                            [
                                0
                                if not i in unit.inputs + (unit.index,)
                                else sub_state[substrate_ixs.index(i)]
                                for i in range(len(u.substrate_state))
                            ]
                        )
                        new_tpm = pyphi.tpm.ExplicitTPM(unit.state_dependent_tpm[long_state])
                        new_tpm = self.get_unit_tpm(
                            unit,
                            new_tpm,
                            substrate_ixs,
                            marginalize_ixs,
                            condition_ixs,
                            substrate_state,
                        )
                        new_state_dependent_tpm[sub_state] = new_tpm

                # get the updated unit
                substrate_unit = Unit(
                    system_ixs.index(unit.index),
                    live_inputs,
                    params=tpm,
                    tpm=tpm,
                    label=unit.label,
                    state=unit.state,
                    state_dependent_tpm=True,
                    inherited_tpm=new_state_dependent_tpm,
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
