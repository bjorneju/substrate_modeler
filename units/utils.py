import numpy as np
import pyphi


def map_to_floor_and_ceil(y, floor, ceiling):
    return floor + (ceiling - floor) * y


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