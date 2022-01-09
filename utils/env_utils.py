import numpy as np

## normolize states
def process_state_constructor(en):

    def process_state(state, normalize=True): #FIXME: This doesn't need to be an inner function. Unnecessary loss of performance
        # states = np.array([state['position'], state['vel']])
        if normalize:
            if en == "Acrobot":
                states = np.array([state[0], state[1], state[2], state[3], state[4], state[5]])
                states[0] = (states[0] + 1) / (2)
                states[1] = (states[1] + 1) / (2)
                states[2] = (states[2] + 1) / (2)
                states[3] = (states[3] + 1) / (2)
                states[4] = (states[4] + (4 * np.pi)) / (2 * 4 * np.pi)
                states[5] = (states[5] + (9 * np.pi)) / (2 * 4 * np.pi)
            elif en == "LunarLander":
                states = np.array([state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]])
                mean = [0, 0.9, 0, -0.6, 0, 0, 0, 0]
                deviation = [0.35, 0.6, 0.7, 0.6, 0.5, 0.5, 1.0, 1.0] #QSTN: why are we doing this to normalize the input. Is there any paper out there that does this? If so why? 
                states[0] = (states[0] - mean[0]) / (deviation[0])
                states[1] = (states[1] - mean[1]) / (deviation[1])
                states[2] = (states[2] - mean[2]) / (deviation[2])
                states[3] = (states[3] - mean[3]) / (deviation[3])
                states[4] = (states[4] - mean[4]) / (deviation[4])
                states[5] = (states[5] - mean[5]) / (deviation[5])

            elif en == "cartpole":
                states = np.array([state[0], state[1], state[2], state[3]])
                states[0] = states[0]
                states[1] = states[1]
                states[2] = states[2]
                states[3] = states[3]

            elif en == "Mountaincar":
                states = np.array([state[0], state[1]])
                states[0] = (states[0] + 1.2) / (0.6 + 1.2)
                states[1] = (states[1] + 0.07) / (0.07 + 0.07)

            elif en == "catcher":
                states = np.array([state['player_x'], state['player_vel'], state['fruit_x'], state['fruit_y']])
                states[0] = (states[0] - 25.5) / 26
                states[1] = states[1] / 10
                states[2] = (states[2] - 30) / 22
                states[3] = (states[3] - 18.5) / 47
            elif en == 'gridhard':
                coef = 255.
                states = state
                states = (2./coef)*states - 1.

        return states

    return process_state
