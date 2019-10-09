# -*- coding: utf-8 -*-
# @Time    : 2019/10/5 13:54
# @Author  : obitolyz
# @FileName: TourGraphCreation.py
# @Software: PyCharm

from math import inf
from itertools import product
from NodeAndEdge import Road
import random


def single_car_tour_graph(graph, requests):
    """
    :param graph: a list of Node{serial_number, coordinate, type, edges{a list of Road{to, length, time, energy}}}
    :param requests: a list of (tuple(p, d), deadline, required_capacity), p, d denote pickup and delivery location respectively
    :return:
    """
    reqs = [r[0] for r in requests]  # extract (i, j) from requests
    node_num = len(graph)

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
         'Pick': ['Pick', 'Delivery', 'Depot'],
         'Delivery': ['Pick', 'Delivery', 'Depot', 'Destination'],
         'Depot': ['Pick', 'Delivery', 'Depot', 'Destination'],
         'Destination': []}
    for i, node in enumerate(graph):
        if node.type.name in D.keys():
            if node.type.name == 'Pick':  # obtain information between pickup to delivery location
                for r in reqs:
                    if node.serial_number == r[0]:
                        node.type.distance = node.type.time = node.type.energy = dist[node.serial_number][r[-1]]
                        break
            node.edges = []
            for j, node_c in enumerate(graph):
                if node_c.type.name in D[node.type.name] and (i != j):
                    length = time = energy = dist[i][j]
                    node.edges.append(Road(j, length, time, energy))
            tour_graph.append(node)

    # L_k_0 point(start) is in the first place, then those of other stops are in a random order
    random.shuffle(tour_graph)
    for i, node in enumerate(tour_graph):
        if node.type.name == 'Start':
            tour_graph[0], tour_graph[i] = tour_graph[i], tour_graph[0]
            break

    return tour_graph


if __name__ == '__main__':
    from GenetateBigGraph import generate_big_graph
    graph, requests = generate_big_graph(node_num=10, lower_bound=1, high_bound=100, request_num=3, depot_num=1)
    single_car_tour_graph(graph, requests)
