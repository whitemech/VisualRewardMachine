import random
import os
import torchvision
from PIL import Image
import torch
import pickle
from DeepAutoma import LSTMAutoma, ProbabilisticAutoma
from Classifier import CNN, Decoder
import itertools
import math
from statistics import mean
from sklearn.model_selection import train_test_split

from utils import eval_acceptance, eval_learnt_DFA_acceptance, eval_image_classification_from_traces
# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
device = 'cpu'
print("device = ", device)
import time
import matplotlib.pyplot as plt
from VRM_loss_functions import sat_current_output, sat_next_transition_batch, conjunction

def create_batches_same_length(dataset, labels, size):
    new_dataset = []
    new_labels = []
    num_batches = int(len(dataset)/size)
    for i in range(num_batches):
        batch_trace = []
        batch_label = []
        for j in range(size):
            batch_trace.append(dataset[i*size+j])
            batch_label.append(labels[i*size+j])
        batch_trace = torch.stack(batch_trace)
        batch_label = torch.stack(batch_label)
        new_dataset.append(batch_trace)
        new_labels.append(batch_label)
    return new_dataset, new_labels 

class Visual_DFA_induction:
    def __init__(self, numb_states, numb_symbols, numb_rewards, automa_implementation = 'logic_circuit', lstm_output= "acceptance", num_exp=0,log_dir="Results/", dataset="minecraft"):
        self.ltl_formula_string = "goal"
        self.log_dir = log_dir
        self.exp_num=num_exp

        self.numb_of_symbols = numb_symbols
        self.numb_of_states = numb_states
        self.numb_of_rewards = numb_rewards

        self.alphabet = ["c"+str(i) for i in range(self.numb_of_symbols) ]

        #################### networks
        self.hidden_dim =numb_states
        self.automa_implementation = automa_implementation

        ##### DeepDFA
        if self.automa_implementation == 'lstm':
            if lstm_output== "states":
                self.deepAutoma = LSTMAutoma(self.hidden_dim, self.numb_of_symbols, self.numb_of_states)
            elif lstm_output == "acceptance":
                self.deepAutoma = LSTMAutoma(self.hidden_dim, self.numb_of_symbols, 2)
            else:
                print("INVALID LSTM OUTPUT. Choose between 'states' and 'acceptance'")
        elif self.automa_implementation == 'logic_circuit':
            self.deepAutoma = ProbabilisticAutoma(self.numb_of_symbols, self.numb_of_states, self.numb_of_rewards)
        else:
            print("INVALID AUTOMA IMPLEMENTATION. Choose between 'lstm' and 'logic_circuit'")
        ##### Classifier
        # if dataset == 'MNIST':
        #     self.num_classes = 2
        #     self.num_channels = 1
        #     nodes_linear = 54

        #     self.pixels_h = 28
        #     self.pixels_v = 28
        #     self.num_features = 4
        if dataset == 'minecraft':
            self.num_classes = 5
            self.num_channels = 3
            nodes_linear = 4704

            self.pixels_h = 128
            self.pixels_v = 128

            self.num_features = 4


        self.classifier = CNN(self.num_channels, self.num_classes, nodes_linear, True)
        self.decoder = Decoder(self.numb_of_symbols)



        self.temperature = 1.0

        resize = torchvision.transforms.Resize((128,128))
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            resize,
        ])
     
        trace = []
        dir = os.listdir('custom_trace_whole')
        for i in range(len(dir)):
            img = Image.open('custom_trace_whole/img'+str(i)+'.jpg')
            img = transforms(img)
            trace.append(img)
        self.a = [torch.stack(trace).unsqueeze(0)]

        self.b = [torch.tensor([[0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,1,0,0],
                                [0,0,0,0,1], [1,0,0,0,0], [0,0,0,0,1], [0,0,0,0,1], 
                                [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1],
                                [0,1,0,0,0], [0,0,0,0,1], [0,0,0,0,1], [0,0,0,1,0]])]

        #[0,0,0,0,1] white cell
        #[0,0,0,1,0] lava
        #[0,0,1,0,0] door
        #[0,1,0,0,0] gem
        #[1,0,0,0,0] pick

    def set_dataset(self, image_traj, rew_traj):

        rew_traj = torch.FloatTensor(rew_traj)

        dataset_traces = []
        dataset_acceptances = rew_traj
        for i in range(len(image_traj)):
            trace = []
            for img in image_traj[i]:
                trace.append(img)
            trace_tensor = torch.stack(trace)
            trace_tensor = torch.squeeze(trace_tensor)
            dataset_traces.append(trace_tensor)
        
        train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = train_test_split(dataset_traces, dataset_acceptances, train_size=0.8, shuffle=True)

        train_img_seq, train_acceptance_img = create_batches_same_length(train_traces, train_acceptance_tr, 4)


        test_img_seq_hard, test_acceptance_img_hard = create_batches_same_length(test_traces, test_acceptance_tr, 4)

        image_seq_dataset = (train_img_seq, [], train_acceptance_img, test_img_seq_hard, [], test_acceptance_img_hard)
        self.train_img_seq, self.train_traces, self.train_acceptance_img, self.test_img_seq_hard, self.test_traces, self.test_acceptance_img_hard = image_seq_dataset
        return 

    def reduce_dfa(self):
        dfa = self.dfa

        admissible_transitions = []
        for true_sym in self.alphabet:
            trans = {}
            for i,sym in enumerate(self.alphabet):
                trans[sym] = False
            trans[true_sym] = True
            admissible_transitions.append(trans)
        red_trans_funct = {}
        for s0 in self.dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = self.dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                for sym, at in enumerate(admissible_transitions):
                    if label.subs(at):
                        red_trans_funct[s0][sym] = key

        self.reduced_dfa = red_trans_funct


    def eval_learnt_DFA(self, automa_implementation, temp, mode="dev"):
        if mode=="dev":
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.dev_traces, self.dev_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr), automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.dev_traces, self.dev_acceptance_tr), automa_implementation, temp)
        else:
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp)
        return train_acc, test_acc

    def train_autoencoder(self):
        ######## TRAIN OF THE AUTOENCODER (self.classifier + self.decoder)
        print("_____________training the Auntoencoder_____________")

        self.classifier.to(device)
        self.decoder.to(device)

        params = list(self.classifier.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=0.001)
        mse_loss = torch.nn.MSELoss()

        mean_loss =0.0
        epoch = 0
        while True:
            print("(Train classifier) epoch: " + str(epoch))
            epoch += 1
            losses = []

            for b in range(len(self.train_img_seq)):
                batch_img_seq = self.train_img_seq[b].to(device)

                optimizer.zero_grad()

                sym_sequence = self.classifier(batch_img_seq.view(-1, self.num_channels, self.pixels_v, self.pixels_h))
                reconstructed_img = self.decoder(sym_sequence)

                loss = mse_loss(reconstructed_img,
                                batch_img_seq.view(-1, self.num_channels, self.pixels_v, self.pixels_h))

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            print("__________________________")
            mean_loss_new = mean(losses)
            print("MEAN LOSS: ", mean_loss_new)

            if abs(mean_loss_new - mean_loss) < 0.1:
                break
            mean_loss =mean_loss_new

        #print histogram for symbol 0
        pred_sym0 = torch.zeros((0)).to(device)
        pred_sym1 = torch.zeros((0)).to(device)
        for b in range(len(self.train_img_seq)):
            batch_img_seq = self.train_img_seq[b].to(device)

            optimizer.zero_grad()

            sym_sequence = self.classifier(batch_img_seq.view(-1, self.num_channels, self.pixels_v, self.pixels_h))
            pred_sym0 = torch.cat((pred_sym0, sym_sequence[:,0]), dim=0)
            pred_sym1 = torch.cat((pred_sym1, sym_sequence[:,1]), dim=0)

        y, x = torch.histogram(pred_sym0.to('cpu'), 100)
        plt.plot(x[:100].detach(), y[:100].detach())
        plt.savefig("pred_sym_0_{}_exp{}.png".format(self.ltl_formula_string, self.exp_num))
        plt.clf()
        y, x = torch.histogram(pred_sym1.to('cpu'), 100)
        plt.plot(x[:100].detach(), y[:100].detach())
        plt.savefig("pred_sym_1_{}_exp{}.png".format(self.ltl_formula_string, self.exp_num))
        plt.clf()
        acc, errors = eval_image_classification_from_traces(self.train_img_seq, self.train_traces, self.classifier, True, True)

        wrong_max = torch.max(errors,1)[0]
        y, x = torch.histogram(wrong_max.to('cpu'), 100)
        plt.plot(x[:100].detach(), y[:100].detach())
        plt.savefig("wrong_max_{}_exp{}.png".format(self.ltl_formula_string, self.exp_num))
        plt.clf()


    def train_all(self, batch_size, num_of_epochs, decay=1.0, freezed=False, use_rec_err=False):
        tot_size = len(self.train_img_seq)

        self.deepAutoma.initFromDfa({0:{0:2, 1:5, 2:0, 3:1, 4:0}, 1:{0:1, 1:1, 2:1, 3:1, 4:1}, 2:{0:2, 1:3, 2:2, 3:1, 4:2}, 3:{0:3, 1:3, 2:4, 3:1, 4:3}, 4:{0:4, 1:4, 2:4, 3:1, 4:4}, 5:{0:3, 1:5, 2:5, 3:1, 4:5}}, [3,4,2,1,0,2])
        self.deepAutoma.to(device)
        self.temperature =1.0

        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_NS_exp"+str(self.exp_num), 'w')
        test_hard_file = open(self.log_dir+self.ltl_formula_string+"_test_hard_acc_NS_exp"+str(self.exp_num), 'w')
        image_classification_train_file = open(self.log_dir+self.ltl_formula_string+"_image_classification_train_acc_NS_exp"+str(self.exp_num), 'w')
        image_classification_test_file = open(self.log_dir+self.ltl_formula_string+"_image_classification_test_acc_NS_exp"+str(self.exp_num), 'w')

        self.classifier.to(device)
        self.decoder.to(device)

        cross_entr = torch.nn.CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss()
        print("_____________training the DFA_____________")
        print("training on {} sequences using {} automaton states".format(tot_size, self.numb_of_states))

        params = self.classifier.parameters()

        optimizer = torch.optim.Adam(params, lr=0.001)#, weight_decay=1e-3)
        sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-04)

        min_temp = 0.00001
        mean_loss = 1000
        if freezed:
            self.temperature = min_temp
        epoch = 0

        ######################## net2dfa
        self.dfa = self.deepAutoma.net2dfa( min_temp)
        try:
            self.dfa.to_graphviz().render("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_initial.dot")
        except:
            print("Not able to render automa")



        for _ in range(num_of_epochs):
            print("epoch: ", epoch)
            epoch+=1
            losses = []

            for b in range(len(self.train_img_seq)):


                batch_img_seq = self.train_img_seq[b].to(device)

                batch_size = batch_img_seq.size()[0]
                length_seq = batch_img_seq.size()[1]
                target_rew_seq = self.train_acceptance_img[b].type(torch.int64).to(device)

                optimizer.zero_grad()

                sym_sequences = self.classifier(batch_img_seq.view(-1, self.num_channels, self.pixels_v , self.pixels_h))
                if use_rec_err:
                    reconstructed_img = self.decoder(sym_sequences)

                sym_sequences = sym_sequences.view(batch_size, length_seq, self.numb_of_symbols)

                pred_states, pred_rew = self.deepAutoma(sym_sequences, self.temperature)
                pred_states = pred_states.view(-1,self.numb_of_states).to(device)
                pred_rew = pred_rew.view(-1, self.numb_of_rewards).to(device)
                target_rew_seq = target_rew_seq.view(-1)
                sat_co = sat_current_output(pred_rew, target_rew_seq)
                sat = sat_co
                loss = - torch.log(sat).mean()

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            
            print("__________________________")
            mean_loss_new = mean(losses)
            print("MEAN LOSS: ", mean_loss_new)

            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_all(automa_implementation='logic_circuit', temperature=min_temp, discretize_labels=True)

            print("SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy : {}".format(train_accuracy, test_accuracy_hard))

            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_all(automa_implementation='logic_circuit', temperature=self.temperature)
            print("SEQUENCE CLASSIFICATION (LOGIC CIRCUIT): train accuracy : {}\ttest accuracy(hard) : {}".format(train_accuracy, test_accuracy_hard))

            train_image_classification_accuracy, test_image_classification_accuracy = self.eval_image_classification()
            print("IMAGE CLASSIFICATION: train accuracy : {}\ttest accuracy : {}".format(train_image_classification_accuracy,test_image_classification_accuracy))

            train_file.write("{}\n".format(train_accuracy))
            test_hard_file.write("{}\n".format(test_accuracy_hard))
            image_classification_train_file.write("{}\n".format(train_image_classification_accuracy))
            image_classification_test_file.write("{}\n".format(test_image_classification_accuracy))

            if freezed:
                self.temperature = min_temp
            else:
                self.temperature = max(self.temperature*decay, min_temp)
            print("temp: ", self.temperature)
            print("optim: ", optimizer.param_groups[0]['lr'])
            


            ######################## net2dfa
            # save the minimized dfa
            self.dfa = self.deepAutoma.net2dfa(min_temp)
            min_num_states = len(self.dfa._states)
            print("min num states:", min_num_states)

            with open("DFA_predicted_nesy/" + self.ltl_formula_string + "_exp" + str(self.exp_num) + "_min_num_states",
                      "a") as f:
                f.write("{}\n".format(min_num_states))
            if abs(mean_loss_new - mean_loss) < 0.0001:
                break

            mean_loss = mean_loss_new
            sheduler.step(mean_loss)
        #write the accuracies of the last epoch
        f = open("image_class_accuracy.txt", "a")
        f.write(str(train_image_classification_accuracy) + "\n")
        f.close()
        f = open("dfa_accuracy.txt", "a")
        f.write(str(train_accuracy) + "\n")
        f.close()
        ######################## net2dfa
        #save the minimized dfa
        self.dfa = self.deepAutoma.net2dfa( min_temp)

        #print it
        try:
            self.dfa.to_graphviz().render("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_minimized.dot")
        except:
            print("Not able to render automa")
        with open("DFA_predicted_nesy/"+self.ltl_formula_string, 'wb') as outp:
            pickle.dump(self.dfa, outp, pickle.HIGHEST_PROTOCOL)


    def train_DFA(self, batch_size, num_of_epochs, decay=0.999, freezed=False):
        def get_lr(optim):
            for param_group in optim.param_groups:
                return param_group['lr']

        tot_size = len(self.train_traces)
        mean_loss = 1000000

        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_NS_exp"+str(self.exp_num), 'w')
        dev_file = open(self.log_dir+self.ltl_formula_string+"_dev_acc_NS_exp"+str(self.exp_num), 'w')

        train_file_dfa = open(self.log_dir+self.ltl_formula_string+"_train_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        dev_file_dfa = open(self.log_dir+self.ltl_formula_string+"_dev_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        test_file_dfa = open(self.log_dir+self.ltl_formula_string+"_test_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        loss_file = open(self.log_dir+self.ltl_formula_string+"_loss_dfa_NS_exp"+str(self.exp_num), 'w')

        cross_entr = torch.nn.CrossEntropyLoss()
        print("_____________training the DFA_____________")
        print("training on {} sequences using {} automaton states".format(tot_size, self.numb_of_states))

        params = [self.deepAutoma.trans_prob] + [self.deepAutoma.rew_matrix]
        optimizer = torch.optim.Adam(params, lr=0.01)
        sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-04)


        min_temp = 0.00001
        self.temperature =1.0

        if freezed:
            self.temperature = min_temp

        start_time = time.time()
        epoch= -1
        while True:
            epoch+=1
            print("epoch: ", epoch)
            losses = []
            for i in range(len(self.train_traces)):

                batch_trace_dataset = self.train_traces[i].to(device)
                batch_acceptance = self.train_acceptance_tr[i].to(device)
                optimizer.zero_grad()

                predictions= self.deepAutoma(batch_trace_dataset, self.temperature)

                loss = cross_entr(predictions, batch_acceptance)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=self.temperature)
            mean_loss_new = mean(losses)
            print("SEQUENCE CLASSIFICATION (LOGIC CIRCUIT): train accuracy : {}\ttest accuracy : {}\tloss : {}".format(train_accuracy, test_accuracy, mean_loss_new))

            train_file.write("{}\n".format(train_accuracy))
            dev_file.write("{}\n".format(test_accuracy))
            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=min_temp)
            print("SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy : {}".format(train_accuracy, test_accuracy))

            train_file_dfa.write("{}\n".format(train_accuracy))
            dev_file_dfa.write("{}\n".format(test_accuracy))
            loss_file.write("{}\n".format(mean(losses)))
            if freezed:
                self.temperature = min_temp
            else:
                self.temperature = max(self.temperature*decay, min_temp)
            print("temp: ", self.temperature)

            sheduler.step(mean_loss_new)
            print("lr: ", get_lr(optimizer))
            if mean_loss_new < 0.318 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            if epoch > 200 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            mean_loss = mean_loss_new


        ######################## net2dfa
        #save the minimized dfa
        self.dfa = self.deepAutoma.net2dfa( min_temp)
        ex_time =  time.time() - start_time

        with open("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+".ex_time", "w") as f:
            f.write("{}\n".format(ex_time))

        #print it
        try:
            self.dfa.to_graphviz().render("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_minimized.dot")
        except:
            print("Not able to render automa")
        with open("DFA_predicted_nesy/"+self.ltl_formula_string, 'wb') as outp:
            pickle.dump(self.dfa, outp, pickle.HIGHEST_PROTOCOL)

        with open("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_min_num_states", "w") as f:
            f.write(str(len(self.dfa._states)))

        #ULTIMO TEST usando il DFA sul TEST set
        train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='dfa', temp=min_temp, mode="test")
        print("FINAL SEQUENCE CLASSIFICATION ON TEST SET: {}".format(test_accuracy))

        test_file_dfa.write("{}\n".format(test_accuracy))


    def train_classifier_and_lstm(self, num_of_epochs):
        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_DL_exp"+str(self.exp_num), 'w')
        test_clss_file = open(self.log_dir+self.ltl_formula_string+"_test_clss_acc_DL_exp"+str(self.exp_num), 'w')
        test_aut_file = open(self.log_dir+self.ltl_formula_string+"_test_aut_acc_DL_exp"+str(self.exp_num), 'w')
        test_hard_file = open(self.log_dir+self.ltl_formula_string+"_test_hard_acc_DL_exp"+str(self.exp_num), 'w')
        print("_____________training classifier+lstm_____________")
        loss_crit = torch.nn.CrossEntropyLoss()
        params = [self.classifier.parameters(), self.deepAutoma.parameters()]
        params = itertools.chain(*params)

        optimizer = torch.optim.Adam(params=params, lr=0.001)
        batch_size = 64
        tot_size = len(self.train_img_seq)
        self.deepAutoma.to(device)
        epoch = 0
        mean_loss_new = 1000
        while True:
            print("epoch: ", epoch)
            epoch += 1
            train_losses = []
            for b in range(math.floor(tot_size/batch_size)):
                start = batch_size*b
                end = min(batch_size*(b+1), tot_size)
                batch_image_dataset = self.train_img_seq[start:end]
                batch_acceptance = self.train_acceptance_img[start:end]
                optimizer.zero_grad()
                losses = torch.zeros(0 ).to(device)


                for i in range(len(batch_image_dataset)):
                    img_sequence =batch_image_dataset[i].to(device)
                    target = batch_acceptance[i]
                    target = torch.LongTensor([target]).to(device)
                    sym_sequence = self.classifier(img_sequence)
                    acceptance = self.deepAutoma.predict(sym_sequence)
                    # Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    loss = loss_crit(acceptance.unsqueeze(0), target)
                    losses = torch.cat((losses, loss.unsqueeze(dim=0)), 0)

                loss = losses.mean()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_all(automa_implementation='lstm', temperature=1.0)
            print("__________________________train accuracy : {}\ttest accuracy(clss) : {}\ttest accuracy(aut) : {}\ttest accuracy(hard) : {}".format(train_accuracy,
                                                                                                 test_accuracy_clss, test_accuracy_aut, test_accuracy_hard))
            mean_loss_new = mean(train_losses)
            print("Mean loss: ", mean_loss_new)


            train_file.write("{}\n".format(train_accuracy))
            test_clss_file.write("{}\n".format(test_accuracy_clss))
            test_aut_file.write("{}\n".format(test_accuracy_aut))
            test_hard_file.write("{}\n".format(test_accuracy_hard))
            if mean_loss_new < 0.318 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            if epoch > 200 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            mean_loss = mean_loss_new


    def eval_all(self, automa_implementation, temperature, discretize_labels=False):
        train_accuracy = eval_acceptance(self.classifier, self.deepAutoma, self.alphabet, (self.train_img_seq, self.train_acceptance_img), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True)

        test_accuracy_hard= eval_acceptance( self.classifier, self.deepAutoma, self.alphabet,(self.test_img_seq_hard, self.test_acceptance_img_hard), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True)

        return train_accuracy, 0,0, test_accuracy_hard

    def eval_image_classification(self):
        train_acc = eval_image_classification_from_traces(self.a, self.b, self.classifier, True)
        test_acc = eval_image_classification_from_traces(self.a, self.b, self.classifier, True)
        return train_acc, test_acc
