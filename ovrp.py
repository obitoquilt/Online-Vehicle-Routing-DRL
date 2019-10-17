# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:24
# @Author  : obitolyz
# @FileName: OVRP.py
# @Software: PyCharm

import torch
import os
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from PtrNet import NeuralCombOptRL
from reward import reward_fn, OVRPDataset
from tqdm import tqdm

# parameters
batch_size = 10
train_size = 128
val_size = 1000
embedding_dim = 128  # p dim
hidden_dim = 128
n_process_blocks = 3
n_glimpses = 1
use_tanh = True
C = 10  # tanh exploration
n_epochs = 1
use_cuda = False
is_train = True
critic_beta = 0.9
beam_size = 1  # if set B=1 then the technique is same as greedy search
actor_net_lr = 1e-4
critic_net_lr = 1e-4
actor_lr_decay_step = 5000
actor_lr_decay_rate = 0.96
critic_lr_decay_step = 5000
critic_lr_decay_rate = 0.96

origin_node_num = 10  # the number of nodes in the original graph
lower_bound = 1
high_bound = 100
request_num = 1
depot_num = 1

load_path = ''

training_dataset = OVRPDataset(num_samples=train_size,
                               node_num=origin_node_num,
                               request_num=request_num,
                               depot_num=depot_num,
                               lower_bound=lower_bound,
                               high_bound=high_bound)
tour_graph_set = training_dataset.get_tour_graph()
request_set = training_dataset.get_request()
car_set = training_dataset.get_car()

# keep the original order
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

seq_len = len(tour_graph_set[0])  # the number of nodes after the connecting points are removed
# instantiate the Neural Combinatorial Opt with RL module
model = NeuralCombOptRL(embedding_dim,
                        hidden_dim,
                        seq_len,
                        n_glimpses,
                        n_process_blocks,
                        C,
                        use_tanh,
                        beam_size,
                        reward_fn,
                        is_train,
                        use_cuda)

# Load the model parameters from a saved state
if load_path != '':
    print('[*] Loading model from {}'.format(load_path))
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), load_path)))  # load parameters
    model.actor_net.decoder.seq_len = seq_len
    model.is_train = is_train

critic_mse = torch.nn.MSELoss()
critic_optim = optim.Adam(model.critic_net.parameters(), lr=critic_net_lr)
actor_optim = optim.Adam(model.actor_net.parameters(), lr=actor_net_lr)

actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
                                           list(range(actor_lr_decay_step,
                                                      actor_lr_decay_step * 1000,
                                                      actor_lr_decay_step)),
                                           gamma=actor_lr_decay_rate)

critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
                                            list(range(critic_lr_decay_step,
                                                       critic_lr_decay_step * 1000,
                                                       critic_lr_decay_step)),
                                            gamma=critic_lr_decay_rate)

if use_cuda:
    model = model.cuda()
    critic_mse = critic_mse.cuda()

step = 0
log_step = 50
epochs = 100

for epoch in range(epochs):
    # sample_batch is [batch_size x sourceL x input_dim]
    for batch_id, sample_batch in enumerate(tqdm(training_dataloader, disable=False)):
        graphs = tour_graph_set[batch_id * batch_size: (batch_id + 1) * batch_size]
        requests = request_set[batch_id * batch_size:(batch_id + 1) * batch_size]
        car = car_set[batch_id * batch_size:(batch_id + 1) * batch_size]

        if use_cuda:
            sample_batch = sample_batch.cuda()

        R, v, probs, actions, actions_idxs = model(sample_batch, car, graphs, requests)
        advantage = R - v  # means L(π|s)-b(s)
        advantage = -advantage

        # compute the sum of the log probs for each tour in the batch
        logprobs = sum([torch.log(prob) for prob in probs])
        # clamp any -inf's to 0 to throw away this tour
        logprobs[(logprobs < -1000).detach()] = 0.  # means log pθ(π|s)

        # multiply each time step by the advanrate
        reinforce = advantage * logprobs
        actor_loss = reinforce.mean()

        # actor net processing
        actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        # clip gradient norms
        torch.nn.utils.clip_grad_norm_(model.actor_net.parameters(), max_norm=2.0, norm_type=2)
        actor_optim.step()
        actor_scheduler.step()

        # critic net processing
        R = R.detach()
        critic_loss = critic_mse(v.squeeze(1), R)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic_net.parameters(), max_norm=2.0, norm_type=2)
        critic_optim.step()
        critic_scheduler.step()

        step += 1

        if step % log_step == 0:
            print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(epoch, batch_id, R.mean().item()))
