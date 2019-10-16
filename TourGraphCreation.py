# -*- coding: utf-8 -*-
# @Time    : 2019/10/5 13:54
# @Author  : obitolyz
# @FileName: TourGraphCreation.py
# @Software: PyCharm

from math import inf
from itertools import product
from NodeAndEdge import Road, Car
import random


def single_car_tour_graph(graph, requests):
    """
    :param graph: a list of Node{serial_number, coordinate, type, edges{a list of Road{to, length, time, energy}}}
    :param requests: a dict of Request objects{number, pick, delivery, deadline, capacity_required, isload}
    :return:
    """
    node_num = len(graph)

    reqs = dict()  # key: pick, value: delivery
    for key in requests:
        req = requests[key]
        reqs[req.pick] = req.delivery
    # generate the matrix of distance
    dist = [[inf] * node_num for _ in range(node_num)]
    for node in graph:
        for road in node.edges:
            dist[node.serial_number][road.to] = road.length

    # utilize floyd_warshall algorithm to obtain the shortest distance between any two points
    for k, i, j in product(range(node_num), repeat=3):
        sum_ik_kj = dist[i][k] + dist[k][j]
        if sum_ik_kj < dist[i][j]:
            dist[i][j] = sum_ik_kj

    # re-create the graph for single car
    tour_graph = []
    D = {'Start': ['Pick', 'Depot'],
         'Pick': ['Pick', 'Delivery', 'Depot', 'Destination'],
         'Delivery': ['Pick', 'Delivery', 'Depot', 'Destination'],
         'Depot': ['Pick', 'Delivery', 'Depot', 'Destination'],
         'Destination': []}
    for i, node in enumerate(graph):
        if node.type.name in D.keys():
            if node.type.name == 'Pick':  # obtain information between pickup to delivery location
                # for r in reqs:
                #     if node.serial_number == r[0]:
                #         node.type.distance = node.type.time = node.type.energy = dist[node.serial_number][r[-1]]
                #         break
                node.type.distance = node.type.time = node.type.energy = dist[node.serial_number][reqs[node.serial_number]]
            node.edges = []
            for j, node_c in enumerate(graph):
                if node_c.type.name in D[node.type.name] and (i != j):
                    length = time = energy = dist[i][j]
                    node.edges.append(Road(j, length, time, energy))
            tour_graph.append(node)

    # L_k_0 point(start) is in the first place, then those of other stops are in a random order
    car = None
    random.shuffle(tour_graph)
    for i, node in enumerate(tour_graph):
        if node.type.name == 'Start':
            car = Car(number=0, cur_location=node.serial_number, battery_size=node.type.battery_size, capacity=node.type.capacity)
            tour_graph[0], tour_graph[i] = tour_graph[i], tour_graph[0]
            break

    return tour_graph, car


if __name__ == '__main__':
    from GenetateBigGraph import generate_big_graph, generate_common_graph

    common_graph = generate_common_graph(node_num=10, lower_bound=1, high_bound=100)
    graph, requests = generate_big_graph(common_graph, node_num=10, request_num=3, depot_num=1)
    single_car_tour_graph(graph, requests)
