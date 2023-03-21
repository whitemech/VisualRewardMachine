import torch


def calculate_VRM_loss_on_sequence(deepDFA, sym_seq, rew_seq):
    return 0
def calculate_VRM_loss_on():
    return 0

def sat_current_output(r_pred, r_target):
    r_target = r_target.unsqueeze(1)
    sat = torch.gather(r_pred, 1, r_target)

    return sat.squeeze(1)

def sat_next_transition_batch(s_pred_batch, r_target_batch, deep_dfa):
    batch_size = s_pred_batch.size()[0]
    sat_batch = torch.zeros((batch_size))
    for i in range(batch_size):
        sat = sat_next_transition(s_pred_batch[i], r_target_batch[i], deep_dfa)
        sat_batch[i] = sat

    return sat_batch

def sat_next_transition(s_pred, r_target, deep_dfa):
    if r_target == 0:
        return torch.ones((1))
    next_action = torch.eye(deep_dfa.numb_of_actions)
    s_pred = s_pred.repeat(deep_dfa.numb_of_actions,1)

    _, next_rew = deep_dfa.step(s_pred, next_action, 1.0)
    
    if r_target == deep_dfa.numb_of_rewards -1:
        sat = forall(next_rew[:,deep_dfa.numb_of_rewards - 1], 3)
    else:
        sat = exists(next_rew[:, r_target - 1], 3)
    return sat

def exists(tensor, p):
    tensor = torch.pow(tensor, p)

    sat= tensor.mean()
    sat=torch.pow(sat, 1/p)
    return sat

def forall(tensor, p):
   
    tensor = 1 - tensor
    tensor = torch.pow(tensor, p)
    sat = tensor.mean()
    sat = torch.pow(sat, 1/p)
    sat = 1 - sat
    return sat

def conjunction(a,b):
    return a*b

