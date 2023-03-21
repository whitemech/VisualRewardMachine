import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dot2pythomata, transacc2pythomata
from Random_DFA import Random_DFA

# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
device = 'cpu'

print(device)

sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x/temp)

class LSTMAutoma(nn.Module):

    def __init__(self, hidden_dim, vocab_size, tagset_size):
        super(LSTMAutoma, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(vocab_size, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):

        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        return tag_space

    def predict(self, sentence):
        tag_space = self.forward(sentence)
        out = F.softmax(tag_space, dim=1)[-1]
        return out



class ProbabilisticAutoma(nn.Module):
    def __init__(self, numb_of_actions, numb_of_states, numb_of_rewards, initialization="gaussian"):
        super(ProbabilisticAutoma, self).__init__()
        self.numb_of_actions = numb_of_actions
        self.alphabet = [str(i) for i in range(numb_of_actions)]
        self.numb_of_states = numb_of_states
        self.numb_of_rewards = numb_of_rewards
        self.reward_values = torch.Tensor(list(range(numb_of_rewards)))
        self.activation = sftmx_with_temp
        #if initialization == "gaussian":
        #standard gaussian noise initialization
        self.trans_prob = torch.normal(0, 0.1, size=( numb_of_actions, numb_of_states, numb_of_states), requires_grad=True, device=device)
        self.rew_matrix = torch.normal(0, 0.1, size=( numb_of_states, numb_of_rewards), requires_grad=True, device=device)
        if initialization == "random_DFA":
            random_dfa = Random_DFA(self.numb_of_states, self.numb_of_actions)
            transitions = random_dfa.transitions
            final_states = []
            for s in range(self.numb_of_states):
                if random_dfa.acceptance[s]:
                    final_states.append(s)
            self.initFromDfa(transitions, final_states)

    #input: sequence of actions (batch, length_seq, num_of_actions)
    def forward(self, action_seq, temp, current_state= None):
        batch_size = action_seq.size()[0]
        length_size = action_seq.size()[1]

        pred_states = torch.zeros((batch_size, length_size, self.numb_of_states))
        pred_rew = torch.zeros((batch_size, length_size, self.numb_of_rewards))

        if current_state == None:
            s = torch.zeros((batch_size,self.numb_of_states)).to(device)
            #initial state is 0 for construction
            s[:,0] = 1.0
        else:
            s = current_state
        for i in range(length_size):
            a = action_seq[:,i, :]

            s, r = self.step(s, a, temp)

            pred_states[:,i,:] = s
            pred_rew[:,i,:] = r

        return pred_states, pred_rew

    def step(self,state, action, temp):
        
        if type(action) == int:
            action= torch.IntTensor([action])
        #activation
        trans_prob = self.activation(self.trans_prob, temp)
        rew_matrix = self.activation(self.rew_matrix, temp)

      
        trans_prob = trans_prob.unsqueeze(0)
        state = state.unsqueeze(1).unsqueeze(-2)
      
        selected_prob = torch.matmul(state, trans_prob)
       

        next_state = torch.matmul(action.unsqueeze(1), selected_prob.squeeze())
      
        next_reward = torch.matmul(next_state, rew_matrix)
       
        return next_state.squeeze(1), next_reward.squeeze(1)

    def step_(self, state, action, temp):

        print("##############################")
        print("state: ", state)
        print("state size: ", state.size())
        print("action :", action)
        print("action size :", action.size())

        print("trans prob size:", self.trans_prob.size())
        print("trans prob:", self.trans_prob)

        if type(action) == int:
            action = torch.IntTensor([action])


        #no activation
        trans_prob = self.trans_prob
        rew_matrix = self.rew_matrix

        print("trans_prob activated size: ", trans_prob.size())
        print("trans_prob activated: ", trans_prob)
        print("rew matrix size:", self.rew_matrix.size())
        print("rew matrix:", self.rew_matrix)
        print("rew_matrix activated size: ", rew_matrix.size())
        print("rew_matrix activated: ", rew_matrix)

        trans_prob = trans_prob.unsqueeze(0)
        state = state.unsqueeze(1).unsqueeze(-2)

        print("transprob size: ", trans_prob.size())
        print("state size: ", state.size())

        selected_prob = torch.matmul(state, trans_prob)

        print("selected prob size: ", selected_prob.size())
        print("selected prob: ", selected_prob)

        next_state = torch.matmul(action.unsqueeze(1), selected_prob.squeeze())

        print("next_state size:", next_state.size())
        print("next_state :", next_state)
        print("rew_matrix:", rew_matrix)

        next_reward = torch.matmul(next_state, rew_matrix)

        print("next reward:", next_reward)
        print("next_rew size: ", next_reward.size())


        return next_state.squeeze(1), next_reward.squeeze(1)

    def net2dfa(self, min_temp):

        trans_prob = self.activation(self.trans_prob, min_temp)
        rew_matrix = self.activation(self.rew_matrix, min_temp)

        trans_prob = torch.argmax(trans_prob, dim= 2)
        rew_matrix = torch.argmax(rew_matrix, dim=1)

        #2transacc
        trans = {}
        for s in range(self.numb_of_states):
            trans[s] = {}
        acc = []
        for i, rew in enumerate(rew_matrix):
                if rew == 0:
                    acc.append(True)
                else:
                    acc.append(False)
        for a in range(trans_prob.size()[0]):
            for s, s_prime in enumerate(trans_prob[a]):
                    trans[s][str(a)] = s_prime.item()

     
        pyautomaton = transacc2pythomata(trans, acc, self.alphabet)
       

        pyautomaton = pyautomaton.reachable()
        

        pyautomaton = pyautomaton.minimize()
       

        return pyautomaton


    def initFromDfa(self, reduced_dfa, outputs, weigth=100):
        with torch.no_grad():
            #zeroing transition probabilities
            for a in range(self.numb_of_actions):
                for s1 in range(self.numb_of_states):
                    for s2 in range(self.numb_of_states):
                        self.trans_prob[a, s1, s2] = 0.0

            #zeroing reward matrix
            for s in range(self.numb_of_states):
                for r in range(self.numb_of_rewards):
                    self.rew_matrix[s,r] = 0.0


        #set the transition probabilities as the one in the dfa
        for s in reduced_dfa:
            for a in reduced_dfa[s]:
                with torch.no_grad():
                    self.trans_prob[a, s, reduced_dfa[s][a]] = weigth

        #set reward matrix
        for s in range(len(reduced_dfa.keys())):
                with torch.no_grad():
                    self.rew_matrix[s, outputs[s]] = weigth

