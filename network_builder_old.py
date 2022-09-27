"""
TODO:
- FIGURE OUT WEIRD PROBABILITIES
- consider making the CompositeUnit somehow also an instance of the Unit. Just a special case?
- consider how to make MechanismCombination dependent on the probabilities
- write validate function
- put validate function in pyphi
- document functions
- allow for sending in function
"""

import numpy as np
import pandas as pd
import string
import pyphi
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm

from unit_functions import *

""" 
    LOCAL VARIABLES, used to validate the objects created.
"""

UNIT_VALIDATION = {
    "sigmoid": dict(
        default_params=dict(
            input_weights=[], determinism=4, threshold=0.5, floor=0.0, ceiling=1.0
        ),
        function=sigmoid,
    ),
    "sor": dict(
        default_params=dict(
            pattern_selection=[], selectivity=10, floor=0.0, ceiling=1.0
        ),
        function=sor_gate,
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
        default_params=dict(floor=0.0, ceiling=1.0), function=mismatch_corrector
    ),
    "modulated_sigmoid": dict(
        default_params=dict(
            input_weights=[], determinism=4, threshold=0.5, floor=0.0, ceiling=1.0
        ),
        function=modulated_sigmoid,
    ),
    "biased_sigmoid": dict(
        default_params=dict(
            input_weights=[], determinism=4, threshold=0.5, floor=0.0, ceiling=1.0
        ),
        function=biased_sigmoid,
    ),
}


class Unit:
    """A unit that can constitute a substrate.

    Represents the unit and holds auxilary data about it to construct a substrate.

    Args:
        index (int): Integer identifier of the unit.

        inputs (tuple[int]): A tuple of integers specifying the identities of the units providing iniputs.

    Keyword Args:
        params (dict or np.ndarray): Contains a specification of the mechanism of the unit. Given either explicitly by a TPM (ndarray) or implicitly as specifications in a dict. The supported unit tpyes can be found in the UNIT_VALIDATION object.

        label (str): Human readable name for the unit

        state (int): Indicates the current state of the unit: 1 means it is ON, 0 OFF. For now, only binary units are supported.

        input_state (tuple[int]): Binary tuple (consisting of 1s and 0s) that indicates the current state of the inputs. This should be understood as the state of the inputs to the unit, and not the past state of the units that provide input to the unit.


    Example:
        .
    """

    def __init__(
        self,
        index,
        inputs,
        params=None,
        label=None,
        state=None,
        input_state=None,
        tpm=None,
        substrate_state=None,
        substrate_indices=None,
    ):
        # Storing the parameters
        self.params = params

        # Store the type of unit
        if not type(tpm) == np.ndarray:
            self.type = params["mechanism"]
        else:
            self.type = "TPM"

        # This unit's index in the list of units.
        self.index = index

        # List of indices that input to the unit (one pr mechanism).
        self.inputs = tuple(inputs)

        # sset substrate indices and state
        self.substrate_indices = (
            tuple(range(np.max((index,) + tuple(inputs)) + 1))
            if substrate_indices is None
            else substrate_indices
        )
        self.substrate_state = (
            (0,) * len(self.substrate_indices)
            if substrate_state is None
            else substrate_state
        )

        # Node labels used in the system
        if not label is None:
            self.label = label
        else:
            self.label = str(self.index)

        # Setting unit and input state (always congruent with the substrate state)
        self.state = self.get_substate((index,))
        self.input_state = self.get_substate(tuple(inputs))
        
        # validating unit
        assert self.validate(), "Unit did not pass validation"

        # Store the type of unit
        if not type(tpm) == np.ndarray:
            self.set_unit_tpm()
        else:
            self.tpm = tpm

    def validate(self):
        """Return whether the specifications for the unit are valid.

        The checks for validity are defined in the local UNIT_VALIDATION object.
        """
        if type(self.params) == dict:
            unit_type = self.params["mechanism"]

            if not unit_type == "composite":
                assert (
                    unit_type in UNIT_VALIDATION.keys()
                ), "Unit mechanism '{}' is not valid".format(unit_type)
                # check that all required params are present
                for key, value in UNIT_VALIDATION[unit_type]["default_params"].items():
                    if not key in self.params["params"]:
                        print(
                            "Unit {} missing {} params, defaulting to {}".format(
                                self.label, key, value
                            )
                        )
                        self.params["params"][key] = value

        return True

    def __repr__(self):
        return "Unit(type={}, label={}, state={})".format(
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

    def set_unit_tpm(self):
        """np.ndarray: The unit TPM.

        A multidimensional array, containing the probabilities for the unit
        turning ON, given the possible states of the inputs.
        """

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
                    substrate_state=self.substrate_state,
                    substrate_indices=self.substrate_indices,
                ).tpm
                self.type = "Composite unit: {}".format(
                    [unit.type for unit in c_unit.units]
                )

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
            
        return tuple(
            [
                self.substrate_state[i]
                for i in unit_indices
            ]
        )

    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index


class CompositeUnit:
    """A composite unit, consisting of multiple interacting sub-units.
    It contains within it a Unit object constructed by combining the sub-units according to the user specifications.

    Args:
        index (int): Integer identifier of the unit.

        inputs (list[tuple[int]]): A list of tuples of integers specifying the identities of the units providing iniputs. The number of elements in the list must match the number of units the CompositeUnit is composed of.

    Keyword Args:
        params (list[dict] or list[np.ndarray]): The list contains specifications of the mechanism of each of the subunits. Given either explicitly by a TPM (ndarray) or implicitly as specifications in a dict. The supported unit tpyes can be found in the UNIT_VALIDATION object.

        label (str): Human readable name for the unit

        state (int): Indicates the current state of the unit: 1 means it is ON, 0 OFF. For now, only binary units are supported.

        input_state (list[tuple[int]]): A list of binary tuples (consisting of 1s and 0s) that indicates the current state of the inputs to each of the subunits. This should be understood as the present state of the inputs to the subunit, and not the past state of the units that provide input to the unit.

        mechanism_combination (list[dict] or list[np.ndarray]): Like the "params" kwarg. This kwarg provides the specifications for how the activations of the distinct subunits combine to yield an output for the CompositUnit as a whole.

    Example:
        .
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
    ):

        # Construct Units for each of the subunits constituting the CompositeUnit
        if input_state==None:
            input_state = [input_state]*len(inputs)
            
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
            )
            for input_ixs, input_s, param in zip(inputs, input_state, params)
        ]

        # Store the list of indices that input to the unit (one pr subunit).
        self.inputs = inputs
        
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
            input_state=self.composite_input_state(units),
            tpm=composite_tpm,
            substrate_state=substrate_state,
            substrate_indices=substrate_indices,
        )
        self.Unit.set_type(self.__repr__())

        self.tpm = self.Unit.tpm

        self.units = units

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
            mechanism_combination (dict or np.ndarray): A specification of the way the composite unit should translate activations of its subunits into a single output from the composite unit.

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

        # getting all the possible states of the individual mechanisms constituting the unit
        MECHANISM_ACTIVATIONS = list(pyphi.utils.all_states(len(tpms)))

        # a local function that returns the unit activation given some state of its constituent mechanisms
        def combined_activation_probability(
            activation_probabilities, mechanism_combination
        ):
            """the probability of the composite unit activating given the activation of subunits due to a particular input state.
            Returns:
                tpm (np.ndarray): The composite unit TPM, specifying the activation probability (probability to turn ON) given all possible input states.

            Args:
                tpms (list[np.ndarray]): A specification of the way the composite unit should translate activations of its subunits into a single output from the composite unit.

            A multidimensional array, containing the probabilities for the unit
            turning ON, given the possible activations of the subunits.
            """

            for state in MECHANISM_ACTIVATIONS:
                
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
        #all_inputs = sorted(tuple(set([i for indices in self.inputs for i in indices])))
        all_inputs = tuple(set([i for indices in self.inputs for i in indices]))
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
            tpm = pyphi.convert.to_md(
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
            if self.mechanism_combination == "strongest":
                def get_strongest(P):
                    Q = np.array([np.abs(p-0.5) for p in P])
                    return P[np.argmax(Q)]
                    
                tpm = pyphi.convert.to_md(
                    np.array(
                        [
                            [
                                get_strongest(
                                    activation_probabilities
                                )
                            ]
                            for activation_probabilities in expanded_tpms
                        ]
                    )
                )
            elif self.mechanism_combination == "average":
                def get_strongest(P):
                    Q = np.array([np.abs(p-0.5) for p in P])
                    return P[np.argmax(Q)]
                    
                tpm = pyphi.convert.to_md(
                    np.array(
                        [
                            [
                                np.mean(
                                    activation_probabilities
                                )
                            ]
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
        #inputs = sorted(tuple(set([ix for unit in units for ix in unit.inputs])))
        inputs = tuple(set([ix for unit in units for ix in unit.inputs]))
        return inputs

    def composite_input_state(self, units):
        """tuple[int]: the input state to the composite unit as a whole (must be congruent among shared inputs to subunits).

        Args:
            units (list[Unit]): all the units that constitute the composite unit.

        """

        # first get the IDs of units that input to subunits
        composite_inputs = self.composite_inputs(units)
        input_state = self.get_subset_state(self.substrate_state, composite_inputs)

        '''
        # go through units to pick out their input states. If the same unit appears multiply, then the state is forced to be whichever state appeared first.
        input_unit_states = dict()
        for unit in units:
            for input_unit, input_state in zip(unit.inputs, unit.input_state):
                if input_unit in input_unit_states:
                    # check that input state of repeated unit is congruent
                    if not input_unit_states[input_unit] == input_state:
                        print(
                            "incongruency in input state of unit {}, forcing state to {}".format(
                                unit.label, input_unit_states[input_unit]
                            )
                        )
                else:
                    # storing the state of the input unit
                    input_unit_states[input_unit] = input_state

        # create the state tuple by looking through the dict constructed before
        input_state = tuple(
            [input_unit_states[input_unit] for input_unit in composite_inputs]
        )
        '''
        
        return input_state

    # THIS IS WRONG! IT DOES NOT GET THE RIGHT STATE BECAUSE IT DOESNT CARE ABOUT THE FULL SET OF INDICES
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


### SUBSTRATES
class Substrate:
    """A model of a substrate, constituted of units

    Args:
        units (list): a list of unit objects

    Attributes:
        tpm (np.ndarray): The node TPM is an array w
        cm (np.ndarray): The node TPM is an array w
        state (tuple)
    """

    def __init__(self, units, state=None):

        # Set indices of the inputs.
        self.node_indices = tuple([unit.index for unit in units])

        # Node labels used in the system
        self.node_labels = tuple([unit.label for unit in units])

        # This node's index in the list of nodes.
        self.create_cm(units)
        
        # storing the units
        self.units = units

        substrate_tpm = []
        
        if state==None:
            # running through all possible substrate states
            for state in tqdm(list(pyphi.utils.all_states(len(units)))):

                # setting auxillary substrate state
                self.state = state

                # check that the state of (state dependent) units
                # and their inputs match the substrate state
                units = self.validate_unit_states(units)

                # Combine Unit TPMs to substrate TPM
                # adding relevant row from the matrix to the substrate tpm
                substrate_tpm.append(self.combine_unit_tpms(units, state))

            self.tpm = pyphi.convert.to_md(np.array(substrate_tpm))
            self.state = None

            # validating unit
            assert self.validate(), "Substrate did not pass validation"
        
        else:
            self.state = state
            # running through all possible substrate states
            for past_state in tqdm(list(pyphi.utils.all_states(len(units)))):

                # check that the state of (state dependent) units
                # and their inputs match the substrate state
                units = self.validate_unit_states(units)

                # Combine Unit TPMs to substrate TPM
                # adding relevant row from the matrix to the substrate tpm
                substrate_tpm.append(self.combine_unit_tpms(units, past_state))

            self.tpm = pyphi.convert.to_md(np.array(substrate_tpm))

            # validating unit
            assert self.validate(), "Substrate did not pass validation"

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
                    """print(
                        "Redefining unit {} to match substrate state {},".format(
                            unit.label, self.state
                        )
                    )"""
                    # redefining unit
                    if unit.params['mechanism'] == 'composite':
                        unit = CompositeUnit(
                            unit.index,
                            unit.params['CompositeUnit'].all_inputs,
                            params=unit.params['CompositeUnit'].params,
                            label=unit.label,
                            state=substrate_unit_state,
                            input_state=substrate_input_state,
                            substrate_state=self.state,
                            substrate_indices=self.node_indices,
                            mechanism_combination=unit.params['CompositeUnit'].mechanism_combination,
                        ).Unit
                        
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
                        )

            new_units.append(unit)

        return new_units

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
        
    def combine_unit_tpms(self, units, past_state):
        
        unit_response = []
        # going through each unit to find its state-dependent activation probability
        for unit in units:
            unit_response.append(float(unit.tpm[tuple([past_state[i] for i in unit.inputs])]))
        '''# sorting the activation probability in an ascending order
        full_tpm = np.array(unit_response)
        substrate_tpm = []
        for ix in sorted(set(self.node_indices)):
            substrate_tpm.append(full_tpm[ix])

        return substrate_tpm'''
        return unit_response

    def create_cm(self, units):
        cm = np.zeros((len(self.node_indices), len(self.node_indices)))

        for unit in units:
            cm[unit.inputs, unit.index] = 1

        self.cm = cm

        return

    def get_network(self):
        return pyphi.network.Network(self.tpm, self.cm, self.node_labels)

    def get_subsystem(self, state, nodes):
        # make sure the substrate is state_specific
        if self.state==state:
            return pyphi.subsystem.Subsystem(self.get_network(), state, nodes)
        else:
            print('remaking substrate to enforce correct state dependence')
            substrate = Substrate(self.units, state=state)
            return pyphi.subsystem.Subsystem(substrate.get_network(), state, nodes)

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
        
    def simulate(self,initial_state=None,timesteps=1000):

        rng = np.random.default_rng(0)
        if initial_state==None:
            initial_state = tuple(rng.integers(0,2,len(self)))
        states = [initial_state]

        for t in range(timesteps):
            P_next = self.tpm[states[-1]]
            comparison = rng.random(len(initial_state))

            states.append(tuple([1 if P>c else 0 for P,c in zip(P_next, comparison)]))

        return states

    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index
