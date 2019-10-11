# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 20:12
# @Author  : obitolyz
# @FileName: reward.py
# @Software: PyCharm

from NodeAndEdge import Car


def find_something(node, d, graph, something=None):
    """
    :param node: corresponding node of pickup location
    :param d: delivery location
    :param graph: tour graph creation for vehicle k
    :param something: length, time or energy
    :return:
    """
    for e in node.edges:
        if e.to == d:
            if something == 'length':
                return e.length
            elif something == 'time':
                return e.time
            elif something == 'energy':
                return e.energy


def find_delivery(p, requests):
    """
    :param p: pickup location
    :param requests: a list of (tuple(p, d), deadline, required_capacity)
    :return:
    """
    for r in requests:
        if r[0][0] == p:
            return r[0][1], r[-1]


def reward_fn2(tour, graph, mapping_table, requests):
    """
    :param tour: a solution of vehicle k
    :param graph: tour graph creation for vehicle k (i.e. small graph)
    :param mapping_table: mapping table
    :param requests: a list of (tuple(p, d), deadline, required_capacity) and
    p, d denote pickup and delivery location respectively
    :return:
    """
    dest_id = None
    for node in graph:
        # node.serial_number = mapping_table[node.serial_number]
        if node.type.name == 'Destination':
            dest_id = node.serial_number

    # clip the tour
    for k, t in enumerate(tour):
        if t == dest_id:
            break
    tour = tour[:k]

    # objective reward
    sum_x_q = 0
    complete_p_d = []  # p->d
    for (p, d), T_q, rc in requests:
        if d in tour[tour.index(p) + 1:]:
            sum_x_q += 1
            complete_p_d.append((d, T_q, rc))

    W = 0
    for i in range(len(tour) - 1):
        W += find_something(graph[mapping_table.index(tour[i])], tour[i + 1], graph,
                            'length')  # can also use mapping table

    # constraint penalty
    T = 0
    for d, T_q, rc in complete_p_d:
        t = 0
        for i in range(tour.index(d)):
            t += find_something(graph[mapping_table.index(tour[i])], tour[i + 1], graph, 'time')

        T += max(t - T_q, 0)

    C = 0
    cur_capacity = 0
    C_bar = 0  # initial energy of vehicle k
    save_d = {}
    for i in range(len(tour) - 1):
        node = graph[mapping_table.index(tour[i])]
        if node.type.name == 'Start':
            cur_capacity += node.type.used_capacity

        elif node.type.name == 'Pick':
            d, rc = find_delivery(node.serial_number, requests)
            cur_capacity += rc
            save_d[d] = rc

        elif node.typ.name == 'Delivery':
            rc = save_d.get(node.serial_number, None)
            if rc:
                cur_capacity -= rc
                del save_d[node.serial_number]  # delete

        C += max(cur_capacity - C_bar, 0)

    C_L_star = cur_capacity


def reward_fn(cars, tours, graphs):
    """
    :param cars: a list of car object
    :param tours: the set of solution of m vehicles
    :param graphs:
    :return:
    """
