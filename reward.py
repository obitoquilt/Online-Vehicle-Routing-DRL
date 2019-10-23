# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 20:12
# @Author  : obitolyz
# @FileName: reward.py
# @Software: PyCharm

import torch
import random
import copy
from tqdm import tqdm
import torch.nn as nn
from GenerateBigGraph import generate_common_graph, generate_big_graph
from TourGraphCreation import single_car_tour_graph
from Struct2Vec import Struct2Vec


# randomly generate the data of graph of vehicle
class OVRPDataset(nn.Module):
    """
    data_set: [batch_size x seq_len x input_dim]
    """

    def __init__(self, num_samples, node_num, request_num, depot_num, lower_bound, high_bound, random_seed=111):
        # request_num, depot_num: variable vars
        super(OVRPDataset, self).__init__()
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        common_graph = generate_common_graph(node_num=node_num, lower_bound=lower_bound, high_bound=high_bound)
        # [num_samples x seq_len x input_dim]
        self.mu_set = []
        self.tour_graph_set = []
        self.request_set = []
        self.car_set = []
        for _ in tqdm(range(num_samples)):
            big_graph, requests = generate_big_graph(copy.deepcopy(common_graph), node_num=node_num,
                                                     request_num=request_num, depot_num=depot_num)
            tour_graph, car = single_car_tour_graph(big_graph, requests)
            x_all, mu_all, ser_num_list = Struct2Vec(copy.deepcopy(tour_graph))

            self.mu_set.append(mu_all)
            self.tour_graph_set.append(tour_graph)
            self.request_set.append(requests)
            self.car_set.append(car)

        self.size = len(self.mu_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.mu_set[idx]

    def get_tour_graph(self):
        return self.tour_graph_set

    def get_request(self):
        return self.request_set

    def get_car(self):
        return self.car_set


def reward_fn(Cars, Tours, Graphs, Requests, C1, C2, C4, time_penalty):
    """
    :param time_penalty: the time penalty when car is out of energy
    :param requests: a dict of request
    :param cars: a list of car object
    :param tours: the set of solution of m vehicles
    :param graphs: a list of graph for each car
    :return:
    """
    zipp = zip(Cars, Tours, Graphs, Requests)
    O = 0
    P = 0
    rr = 0
    for z in zipp:
        car = z[0]
        tour = z[1]
        graph_temp = z[2]
        requests = z[3]
        graph = {}
        cur_time = 0
        for node in graph_temp:
            graph[node.serial_number] = node
        for i in range(len(tour) - 1):
            node_number = tour[i]
            node = graph[node_number]
            if node.type.name == "Start":
                car.tour_time.append(cur_time)
            elif node.type.name == "Pick":
                car.tour_time.append(cur_time)
                request_num = node.type.request_num
                if cur_time <= node.type.pickup_deadline and requests[request_num].isload is False:
                    request = requests[request_num]
                    if car.capacity - car.used_capacity >= request.capacity_required:
                        car.load_request.append(request_num)
                        requests[request_num].isload = True
                        car.used_capacity += request.capacity_required
            elif node.type.name == "Delivery":
                car.tour_time.append(cur_time)
                request_num = node.type.request_number
                if request_num in car.load_request:
                    car.timeout = max(cur_time - node.type.delivery_deadline, 0)
                    car.finished_request.append(request_num)
                    car.used_capacity -= requests[request_num].capacity_required
            elif node.type.name == "Depot":
                car.tour_time.append(cur_time)
                cur_time += ((car.battery_size - car.cur_energy) / node.type.R)
            elif node.type.name == "Destination":
                break
            next_node_num = tour[i + 1]
            for e in node.edges:
                if e.to == next_node_num:
                    road = e
                    break
            if car.cur_energy < road.energy:
                car.cur_energy = car.battery_size
                cur_time += time_penalty
                car.outofenergy = max(road.energy - car.cur_energy, 0)
            else:
                car.cur_energy -= road.energy
            car.tour_len += road.length
            cur_time += road.time
        O += len(car.finished_request)*1000 - C1 * car.tour_len
        P += C2 * car.timeout + C4 * car.used_capacity
        rr += len(car.finished_request)

    print('finished_request:{}'.format(rr))

    return torch.FloatTensor([O - P])


def reward_fn_test(Cars, Tours, Graphs, Requests, C1, C2, C4, time_penalty):
    """
    :param time_penalty: the time penalty when car is out of energy
    :param requests: a dict of request
    :param cars: a list of car object
    :param tours: the set of solution of m vehicles
    :param graphs: a list of graph for each car
    :return:
    """
    zipp = zip(Cars, Tours, Graphs, Requests)
    O = 0
    P = 0
    rr = 0
    for z in zipp:
        car = z[0]
        tour = z[1]
        graph_temp = z[2]
        requests = z[3]
        graph = {}
        cur_time = 0
        for node in graph_temp:
            graph[node.serial_number] = node
        for i in range(len(tour) - 1):
            node_number = tour[i]
            node = graph[node_number]
            if node.type.name == "Start":
                car.tour_time.append(cur_time)
            elif node.type.name == "Pick":
                car.tour_time.append(cur_time)
                request_num = node.type.request_num
                if requests[request_num].isload is False:
                    request = requests[request_num]
                    car.load_request.append(request_num)
                    requests[request_num].isload = True
            elif node.type.name == "Delivery":
                car.tour_time.append(cur_time)
                request_num = node.type.request_number
                if request_num in car.load_request:
                    car.finished_request.append(request_num)
            elif node.type.name == "Depot":
                car.tour_time.append(cur_time)
            elif node.type.name == "Destination":
                break
            next_node_num = tour[i + 1]
            for e in node.edges:
                if e.to == next_node_num:
                    road = e
                    break
            car.tour_len += road.length
            cur_time += road.time
        O += len(car.finished_request)*1000 - C1 * car.tour_len
        P += C2 * car.timeout + C4 * car.used_capacity
        rr += len(car.finished_request)

    print('finished_request:{}'.format(rr))

    return torch.FloatTensor([O - P])
