import numpy as np
import pyphi

"""
TODO:
- move all validation checks to validation function

"""


def sigmoid(
    unit,
    i=0,
    input_weights=None,
    determinism=None,
    threshold=None,
    floor=None,
    ceiling=None,
):
    def LogFunc(x, determinism, threshold):
        y = ceiling * (
            floor + (1 - floor) / (1 + np.e ** (-determinism * (x - threshold)))
        )
        return y

    n_nodes = len(input_weights)

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                LogFunc(
                    sum(state * np.array([input_weights[n] for n in range(n_nodes)])),
                    determinism,
                    threshold,
                )
            ]
            for state in pyphi.utils.all_states(n_nodes)
        ]
    )

    return pyphi.convert.to_multidimensional(tpm)


def sor_gate(
    unit, i=0, pattern_selection=None, selectivity=None, floor=None, ceiling=None
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
    not_pattern = floor + (ceiling - floor) / selectivity
    tpm[unit.input_state] = (
        ceiling if unit.input_state in pattern_selection else not_pattern
    )

    return tpm


def copy_gate(unit, i=0, floor=None, ceiling=None):
    tpm = np.ones([2]) * floor
    tpm[1] = ceiling
    return tpm


def and_gate(unit, i=0, floor=None, ceiling=None):
    tpm = np.ones((2, 2)) * floor
    tpm[(1, 1)] = ceiling
    return tpm


def or_gate(unit, i=0, floor=None, ceiling=None):
    tpm = np.ones((2, 2)) * ceiling
    tpm[(0, 0)] = floor
    return tpm


def xor_gate(unit, i=0, floor=None, ceiling=None):
    tpm = np.ones((2, 2)) * floor
    tpm[(0, 1)] = ceiling
    tpm[(1, 0)] = ceiling
    return tpm


def weighted_mean(unit, i=0, weights=[], floor=None, ceiling=None):
    
    weights = [w/np.sum(weights) for w in weights]
    N = len(weights)
    
    tpm = np.ones((2,)*N)
    for state in pyphi.utils.all_states(N):
        weighted_mean = sum([(1 + w*(s*2-1))/2 for w,s in zip(weights, state)])/N
        tpm[state] = weighted_mean*(ceiling-floor) + floor
    
    return tpm


def democracy(unit, i=0, floor=None, ceiling=None):
    
    N = len(unit.inputs)
    
    tpm = np.ones((2,)*N)
    for state in pyphi.utils.all_states(N):
        avg_vote = np.mean(state)
        tpm[state] = avg_vote*(ceiling-floor) + floor
        
    return tpm


def majority(unit, i=0, floor=None, ceiling=None):
    
    N = len(unit.inputs)
    
    tpm = np.ones((2,)*N)
    for state in pyphi.utils.all_states(N):
        avg_vote = round(np.mean(state))
        tpm[state] = avg_vote*(ceiling-floor) + floor
        
    return tpm

def mismatch_corrector(unit, i=0, floor=None, ceiling=None):

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
        tpm = np.ones([2]) * 0.5
    else:
        tpm = np.array([floor, ceiling])

    return tpm


def modulated_sigmoid(
    unit,
    i=0,
    input_weights=None,
    determinism=None,
    threshold=None,
    floor=None,
    ceiling=None,
):
    # modulation by the last unit in the inputs.
    # modulation consists in a linear shift in the threshold of the sigmoid, distance given by last value in input_weights

    def LogFunc(x, determinism, threshold):
        y = ceiling * (
            floor + (1 - floor) / (1 + np.e ** (-determinism * (x - threshold)))
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
                    threshold - input_weights[-1] * state[-1],
                )
            ]
            for state in pyphi.utils.all_states(n_nodes)
        ]
    )

    return pyphi.convert.to_multidimensional(tpm)


def biased_sigmoid(
    unit,
    i=0,
    input_weights=None,
    determinism=None,
    threshold=None,
    floor=None,
    ceiling=None,
):
    # A sigmoid unit that is biased in its activation by the last unit in the inputs.
    # The bias consists in a rescaling of the activation probability to make it more in line with the biasing unit. The biasing unit is assumed to be the last one of the inputs.
    # For example, if the biased unit is OFF, the sigmoid activation probability is divided by the factor given in the last value of input_weights. If the unit is ON, 1 - the activation probability is divided by the factor (in essence reducing the probability that it will NOT activate).

    def LogFunc(x, determinism, threshold):
        y = ceiling * (
            floor + (1 - floor) / (1 + np.e ** (-determinism * (x - threshold)))
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

    return pyphi.convert.to_multidimensional(tpm)
