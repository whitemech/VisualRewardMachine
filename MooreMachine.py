class MooreMachine:
    def __init__(self, transition_function, output_function):
        self.transition_function = transition_function
        self.output_function = output_function
        assert len(transition_function.keys()) == len(output_function)
        self.numb_of_states = len(output_function)
        self.numb_of_actions = len(transition_function[0].keys())
        self.numb_of_rewards = len(set(output_function))

    def output(self, string):
        if string == '':
            return self.output_function[0]
        return self.output_from_state(0, string)

    def output_from_state(self, state, string):
        assert string != ''

        a = string[0]
        next_state = self.transition_function[state][a]

        if len(string) == 1:
            return self.output_function[next_state]

        return self.output_from_state(next_state, string[1:])

t_function = {0:{0:2, 1:5, 2:0, 3:1, 4:0}, 1:{0:1, 1:1, 2:1, 3:1, 4:1}, 2:{0:2, 1:3, 2:2, 3:1, 4:2}, 3:{0:3, 1:3, 2:4, 3:1, 4:3}, 4:{0:4, 1:4, 2:4, 3:1, 4:4},
              5:{0:3, 1:5, 2:5, 3:1, 4:5}}
o_function = [3,4,2,1,0,2]

MinecraftMoore = MooreMachine(t_function, o_function)
