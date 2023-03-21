import torch
import random
from numpy.random import RandomState
import os
import numpy as np
from copy import deepcopy
from pythomata import SymbolicAutomaton, PropositionalInterpretation, SimpleDFA
import pickle
from VRM_loss_functions import sat_current_output
#from DeepAutoma import sftmx_with_temp
# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
device = 'cpu'

sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x/temp)

def set_seed(seed: int) -> RandomState:
    """ Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state


def eval_learnt_DFA_acceptance_no_batch(automa, dataset, automa_implementation='logic_circuit', temp=1.0, alphabet=None):

    #automa implementation =
    #   - 'dfa' use the discretized probabilistic automaton 
    #   - 'logic_circuit'
    #   - 'lstm' use the lstm model in automa

    total = 0
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for i in range(len(dataset[0])):
            sym = dataset[0][i]
            label = dataset[1][i]

            if automa_implementation == 'logic_circuit' or automa_implementation == 'lstm':
                sym = sym.unsqueeze(0)
                pred_acceptace = automa(sym, temp)
                output = torch.argmax(pred_acceptace).item()
            elif automa_implementation == 'dfa':
                sym_trace = tensor2symtrace(sym, alphabet)
                output = int(automa.accepts(sym_trace))
            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)

            total += 1


            correct += int(output==label)


            accuracy = 100. * correct/(float)(total)

    return accuracy

def eval_learnt_DFA_acceptance(automa, dataset, automa_implementation='logic_circuit', temp=1.0, alphabet=None):

    #automa implementation =
    #   - 'dfa' use the discretized probabilistic automaton #TODO
    #   - 'logic_circuit'
    #   - 'lstm' use the lstm model in automa

    total = 0
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for i in range(len(dataset[0])):
            sym = dataset[0][i].to(device)
            if automa_implementation != "dfa":
                label = dataset[1][i].to(device)
            else:
                label = dataset[1][i]

            if automa_implementation == 'logic_circuit' or automa_implementation == 'lstm':
                pred_acceptace = automa(sym, temp)
                output = torch.argmax(pred_acceptace, dim= 1)
            elif automa_implementation == 'dfa':

                output = torch.zeros((sym.size()[0]), dtype=torch.int)
                for k in range(sym.size()[0]):

                    sym_trace = tensor2string(sym[k])
                    output[k] = int(automa.accepts(sym_trace))

            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)
            total += output.size()[0]


            correct += sum(output==label).item()


            accuracy = 100. * correct/(float)(total)

    return accuracy

def tensor2symtrace(tensor, alphabet):
    truth_value = {}

    for c in alphabet:
        truth_value[c] = False

    symtrace = []
    tensor=tensor.tolist()

    for sym in tensor:
      
        step = truth_value.copy()
        step["c"+str(sym)] = True
        symtrace.append(step)

    return symtrace

def tensor2string(tensor):
    string = ""
    tensor=tensor.tolist()

    for sym in tensor:
        string += str(sym)

    return string

def dot2pythomata(dot_file_name, action_alphabet):

        fake_action = "(~"+action_alphabet[0]
        for sym in action_alphabet[1:]:
            fake_action+=" & ~"+sym
        fake_action+=") | ("+action_alphabet[0]
        for sym in action_alphabet[1:]:
            fake_action+=" & "+sym
        fake_action+=")"

        file1 = open(dot_file_name, 'r')
        Lines = file1.readlines()

        count = 0
        states = set()

        for line in Lines:
            count += 1
            if count >= 11:
                if line.strip()[0] == '}':
                    break
                action = line.strip().split('"')[1]
                states.add(line.strip().split(" ")[0])
            else:
                if "doublecircle" in line.strip():
                    final_states = line.strip().split(';')[1:-1]

        automaton = SymbolicAutomaton()
        state_dict = dict()
        state_dict['0'] = 0
        for state in states:
            if state == '0':
                continue
            state_dict[state] = automaton.create_state()

        final_state_list = []
        for state in final_states:
            state = int(state)
            state = str(state)
            final_state_list.append(state)

        for state in final_state_list:
            automaton.set_accepting_state(state_dict[state], True)

        count = 0
        for line in Lines:
            count += 1
            if count >= 11:
                if line.strip()[0] == '}':
                    break
                action = line.strip().split('"')[1]
                action_label = action
                for sym in action_alphabet:
                    if sym != action:
                        action_label += " & ~"+sym
                init_state = line.strip().split(" ")[0]
                final_state = line.strip().split(" ")[2]
                automaton.add_transition((state_dict[init_state], action_label, state_dict[final_state]))
                automaton.add_transition((state_dict[init_state], fake_action, state_dict[init_state]))

        automaton.set_initial_state(state_dict['0'])

   
        return automaton

def from_dfainductor_2_transacc(picklepath):
    with open(picklepath, "rb") as f:
        dfa = pickle.load(f)
    print("dfa_ind:")
    print(dfa.__dict__)
    trans = {}
    acc = []
    dfa = dfa.__dict__["_states"]

    for s in dfa:
        trans[s.id_] = {}
        acc.append(int(s.is_accepting()))
        for action in s.children.keys():
            action_int = int(action)
            trans[s.id_][action_int] = s.children[action].id_
    print("trans acc")
    print(trans)
    print(acc)
    return trans, acc

def transacc2pythomata_old(trans, acc, action_alphabet):

    automaton = SymbolicAutomaton()
    state_dict = dict()
    states = trans.keys()
    states = [str(s) for s in states]

    state_dict['0'] = 0
    for state in states[1:]:
        state_dict[state] = automaton.create_state()


    for s in range(len(acc)):
        state = str(s)
        if acc[s] > 0:
            automaton.set_accepting_state(state_dict[state], True)

    #automaton.set_initial_state(state_dict['0'])

    fake_action = "(~" + action_alphabet[0]
    for sym in action_alphabet[1:]:
        fake_action += " & ~" + sym
    fake_action += ") | (" + action_alphabet[0]
    for sym in action_alphabet[1:]:
        fake_action += " & " + sym
    fake_action += ")"

    for s0 in trans.keys():
        for action in trans[s0].keys():
            s1= trans[s0][action]

            action = "c"+str(action)
            action_label = action
            for sym in action_alphabet:
                if sym != action:
                    action_label += " & ~" + sym
            init_state = str(s0)
            final_state = str(s1)
            automaton.add_transition((state_dict[init_state], action_label, state_dict[final_state]))
            automaton.add_transition((state_dict[init_state], fake_action, state_dict[init_state]))

    return automaton

def transacc2pythomata(trans, acc, action_alphabet):
    accepting_states = set()
    for i in range(len(acc)):
        if acc[i]:
            accepting_states.add(i)

    automaton = SimpleDFA.from_transitions(0, accepting_states, trans)

    return automaton


def dataset_from_dict(path):
    with open(path, "rb") as f:
        ds_dict = pickle.load(f)

    strings = []
    labels = []

    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))
    print(sorted_ds_dict[:10])
    len0 = 0
    batch_size = 64

    for string,label in sorted_ds_dict:
        if string=='':
            continue
        l = len(string)
        if l > len0:
            len0 = l
            strings.append(torch.zeros((0,len(string)),dtype=torch.int))
            labels.append([])
        #else:
        strings[-1] = torch.cat((strings[-1], torch.zeros((1, len(string)),dtype=torch.int)))
        labels[-1].append(label)

        for i, char in enumerate(string):
            strings[-1][-1][i] = int(char)

    labels = [torch.LongTensor(label) for label in labels]
    print("-----statistics------")
    print([s.size()[0] for s in strings])
    return strings, labels

def dataset_from_dict_list_of_tens(path, num_of_symbols = 2):
    with open(path, "rb") as f:
        ds_dict = pickle.load(f)

    strings = []
    labels = []

    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))

    for string,label in sorted_ds_dict:
        if string=='':
            continue

        strings.append(torch.zeros((len(string), num_of_symbols)))
        labels.append(int(label))

        for i, char in enumerate(string):
            strings[-1][i][int(char)] = 1

    return strings, labels

def dataset_from_dict_list_of_tens_reward(ds_dict, mooreMachine, num_of_symbols = 2):

    strings = []
    labels = []
    num_outputs = mooreMachine.numb_of_rewards
    indices = [[] for i in range(num_outputs)]
    lenstr = len(list(ds_dict.keys())[0])
    batch_size = round(64 / lenstr)
    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))
    for string,label in sorted_ds_dict:
        if string=='':
            continue

        strings.append(torch.zeros((len(string), num_of_symbols)))
        labels.append(torch.zeros((len(string)), dtype=torch.int))

        for i, char in enumerate(string):
            strings[-1][i][int(char)] = 1
            labels[-1][i] = mooreMachine.output(string[:i+1])
        final_output = mooreMachine.output(string)
        indices[final_output].append(len(strings) -1 )


    current_indices = [0 for i in range(num_outputs)]
    num_samples_in_each_batch = [round(batch_size / num_outputs) for i in range(num_outputs - 1)]
    num_samples_in_each_batch.append(batch_size - sum(num_samples_in_each_batch))

    X = []
    y = []
    all_data_seen = [False for _ in range(num_outputs)]
    while sum(all_data_seen) != num_outputs:
        batch_X = torch.zeros((batch_size, lenstr, num_of_symbols))
        batch_y = torch.zeros((batch_size, lenstr))
        curr_batch_ind = 0
        for o in range(num_outputs):
            indices_o =indices[o][current_indices[o]: current_indices[o] + num_samples_in_each_batch[o]]
            rest = num_samples_in_each_batch[o] - len(indices_o)
            while rest > 0:
                indices_o += indices[o][:rest]
                rest = num_samples_in_each_batch[o] - len(indices_o)
            if current_indices[o] + num_samples_in_each_batch[o] >= len(indices[o]):
                all_data_seen[o] = True
            current_indices[o] = (current_indices[o] + num_samples_in_each_batch[o]) % len(indices[o])
            for ind in indices_o:
                batch_X[curr_batch_ind] = strings[ind]
                batch_y[curr_batch_ind] = labels[ind]
                X.append(batch_X)
                y.append(batch_y)
                curr_batch_ind+=1

    return X, y

def dataset_with_errors_from_dict(path, error_rate):
    with open(path, "rb") as f:
        ds_dict = pickle.load(f)

    strings = []
    labels = []

    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))
    print(sorted_ds_dict[:10])

    if sorted_ds_dict[0][0] == '':
        sorted_ds_dict = sorted_ds_dict[1:]

    len_ds = len(sorted_ds_dict)

    n_errors = round(error_rate*len_ds)
    errors = random.sample(list(range(len_ds)), n_errors)

    len0 = 0


    for i in range(len_ds):
        string, label = sorted_ds_dict[i]
        if i in errors:
            label = not label
        if string=='':
            continue
        l = len(string)
        if l > len0:
            len0 = l
            strings.append(torch.zeros((0,len(string)),dtype=torch.int))
            labels.append([])
        #else:
        strings[-1] = torch.cat((strings[-1], torch.zeros((1, len(string)),dtype=torch.int)))


        labels[-1].append(label)

        for i, char in enumerate(string):
            strings[-1][-1][i] = int(char)

    labels = [torch.LongTensor(label) for label in labels]
    return strings, labels


def abadingo_dataset_from_dict(input_file, output_file, alphabet):
    with open(input_file, "rb") as f:
        ds_dict = pickle.load(f)

    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))


    if sorted_ds_dict[0][0] == '':
        len_ds = len(sorted_ds_dict) -1
    else:
        len_ds = len(sorted_ds_dict)

    n_symbols = len(alphabet)

    f = open(output_file, "w")

    f.write("{} {}\n".format(len_ds, n_symbols))

    for string,label in sorted_ds_dict:
        if string=='':
            continue
        f.write("{} {}".format(int(label), len(string)))
        for char in string:
            f.write(" {}".format(char))
        f.write("\n")

def abadingo_dataset_with_errors_from_dict(input_file, output_file, alphabet, error_rate):
    with open(input_file, "rb") as f:
        ds_dict = pickle.load(f)

    sorted_ds_dict = sorted(list(ds_dict.items()), key=lambda x: len(x[0]))

    if sorted_ds_dict[0][0] == '':
        sorted_ds_dict = sorted_ds_dict[1:]

    len_ds = len(sorted_ds_dict)

    n_errors = round(error_rate*len_ds)
    errors = random.sample(list(range(len_ds)), n_errors)

    n_symbols = len(alphabet)

    f = open(output_file, "w")

    f.write("{} {}\n".format(len_ds, n_symbols))

    for i in range(len_ds):
        string, label = sorted_ds_dict[i]
        if string=='':
            continue
        if i in errors:
            label = not label
        f.write("{} {}".format(int(label), len(string)))
        for char in string:
            f.write(" {}".format(char))
        f.write("\n")

def eval_acceptance(classifier, automa, alphabet, dataset, automa_implementation='dfa', temperature = 1.0, discretize_labels= True, mutually_exc_sym=True):
    #automa implementation =
    #   - 'dfa' use the perfect dfa given
    #   - 'lstm' use the lstm model
    #   - 'logic_circuit' use the fuzzy automaton
    total = 0
    correct = 0
    test_loss = 0
    classifier.eval()
    numb_of_symbols = len(alphabet)
    with torch.no_grad():
        for i in range(len(dataset[0])):
            image_sequences = dataset[0][i].to(device)
            

            labels = dataset[1][i].to(device)

            batch_size = image_sequences.size()[0]

            length_seq = image_sequences.size()[1]

            num_channels = image_sequences.size()[2]
            pixels_v = image_sequences.size()[3]
            pixels_h = image_sequences.size()[4]
        

            symbols = classifier(image_sequences.view(-1, num_channels, pixels_v, pixels_h))
            if discretize_labels:
                symbols[:,0] = torch.where(symbols[:,0] > 0.5, 1., 0.)
                symbols = sftmx_with_temp(symbols, temp=0.00001)
            sym_sequences = symbols.view(batch_size, length_seq, numb_of_symbols)


            if automa_implementation == 'lstm':
                accepted = automa(sym_sequences)
                accepted = accepted[-1]

                output = torch.argmax(accepted).item()
            elif automa_implementation == 'logic_circuit':

                pred_states, pred_rew = automa(sym_sequences, temperature)
                num_out = pred_rew.size()[-1]
                pred_rew = pred_rew.view(-1, num_out)
                labels = labels.view(-1)
               
                output = torch.argmax(pred_rew, dim=-1).to(device)
              
            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)
        
            total += labels.size()[0]

          
            correct += (output==labels).sum().item()
        test_accuracy = 100. * correct/(float)(total)
    return test_accuracy

def eval_image_classification_from_traces(traces_images, traces_labels, classifier, mutually_exclusive, return_errors=False):
    total = 0
    correct = 0
    classifier.eval()
    errors = torch.zeros((0,2)).to(device)

    LEN = min(len(traces_images),len(traces_labels))
 
    with torch.no_grad():
        for i in range(LEN) :
            batch_t_sym = traces_labels[i].to(device)
            batch_t_img = traces_images[i].to(device)
            batch_size, length_seq, num_channels , pixels_v, pixels_h = list(batch_t_img.size())
            pred_symbols = classifier(batch_t_img.view(-1, num_channels, pixels_v, pixels_h))
            gt_symbols = batch_t_sym.view(-1, batch_t_sym.size()[-1])
            if  not mutually_exclusive:

                y1 = torch.ones(batch_t_sym.size()).to(device)
                y2 = torch.zeros(batch_t_sym.size()).to(device)

                output_sym = pred_symbols.where(pred_symbols <= 0.5, y1)
                output_sym = output_sym.where(pred_symbols > 0.5, y2)

                correct += torch.sum(output_sym == batch_t_sym).item()
                total += torch.numel(pred_symbols)

            else:
                output_sym = torch.argmax(pred_symbols, dim=1)
                gt_sym = torch.argmax(gt_symbols, dim = 1)
                equality = output_sym == gt_sym
                correct += torch.sum(equality).item()
                if return_errors:
                    eq_list = list(equality)
                    for eq_i,eq in enumerate(eq_list):
                        if not eq:
                            errors = torch.cat((errors, pred_symbols[eq_i,:].unsqueeze(0)), dim=0)
                total += torch.numel(output_sym)


    accuracy = 100. * correct / (float)(total)
    if return_errors:
        return accuracy, errors
    return accuracy

