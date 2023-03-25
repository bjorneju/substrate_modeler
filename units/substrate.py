from .Unit import Unit
from .utils import reshape_to_md

from typing import Tuple, List
from functools import cached_property
import pyphi
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


PROGRESS_BAR_THRESHOLD = 2**10


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
        elif state is not None:
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

                if initial_state is None:
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
                        initial_state.insert(i, np.random.randint(0,2))
                        
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
                        initial_state.insert(i, np.random.randint(0,2))
                        
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
