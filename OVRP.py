# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:24
# @Author  : obitolyz
# @FileName: OVRP.py
# @Software: PyCharm

import torch
import torch.optim as optim
import numpy as np
from GenetateBigGraph import generate_big_graph
from TourGraphCreation import single_car_tour_graph
from Struct2Vec import Struct2Vec
from PtrNet import NeuralCombOptRL

graph, requests = generate_big_graph(node_num=10, lower_bound=1, high_bound=100, request_num=3, depot_num=1)
graph = single_car_tour_graph(graph, requests)
x_all, mu_all, ser_num_list = Struct2Vec(graph)

print(ser_num_list)
