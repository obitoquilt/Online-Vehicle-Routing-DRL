# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 20:12
# @Author  : obitolyz
# @FileName: reward.py
# @Software: PyCharm


def reward_fn(cars, tours, graphs, requests, C1, C2, C4, time_penalty):
    """
    :param time_penalty: the time penalty when car is out of energy
    :param requests: a dic of request
    :param cars: a list of car object
    :param tours: the set of solution of m vehicles
    :param graphs:a list of graph for each car
    :return:
    """
    zipp = zip(cars, tours, graphs)
    O = 0
    P = 0
    for z in zipp:
        car = z[0]
        tour = z[1]
        graph_temp = z[2]
        graph = {}
        cur_time = 0
        for node in graph_temp:
            graph[node.serial_number] = node
        for i in range(len(tour)):
            node_number = tour[i]
            node = graph[node_number]
            if node.type.name == "Start":
                car.tour_time.append(cur_time)
            elif node.type.name == "Pick":
                car.tour_time.append(cur_time)
                request_num = node.type.request_num
                if cur_time < node.type.pickup_deadline and requests[request_num].isload is False:
                    request = requests[request_num]
                    if car.capacity - car.used_capacity >= request.capacity_required:
                        car.load_request.append(request_num)
                        requests[request_num].isload = True
                        car.used_capacity += request.capacity_required
            elif node.type.name == "Delivery":
                car.tour_time.append(cur_time)
                request_num = node.type.request_num
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
        O += len(car.finished_request) - C1 * car.tour_len
        P += C2 * car.timeout + C4 * car.used_capacity

    return O - P
