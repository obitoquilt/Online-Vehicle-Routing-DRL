from NodeAndEdge import *
import random


def generate_big_graph(node_num, lower_bound, high_bound, request_num, depot_num):
    """

    :param node_num: the number of node
    :param lower_bound: the lower bound of coordinate of node
    :param high_bound: the high bound of coordinate of node
    :param request_num: the number of request
    :param depot_num: the number of depot
    :return:
    """
    common_graph = generate_common_graph(node_num, lower_bound, high_bound)
    uncommon_node_num = 2 * request_num + depot_num + 2
    assert uncommon_node_num <= node_num, "uncommon_node_num must be less than node_num!"
    node_serials = random.sample(range(0, node_num), uncommon_node_num)
    pick = node_serials[:request_num]
    delivery = node_serials[request_num:2 * request_num]
    depots = node_serials[2 * request_num:2 * request_num + depot_num]
    start = node_serials[-2]  # starting point of car
    destination = node_serials[-1]  # destination of car
    deadline = random.sample(range(200, 300), request_num)
    capacity_required = random.sample(range(5, 20), request_num)
    requests_temp = list(zip(pick, delivery, deadline, capacity_required))
    requests = {}
    for i, re in enumerate(requests_temp):
        request = Request(i, re[0], re[1], re[2], re[3])
        common_graph[re[0]].type = Pick(re[2], re[3], i)
        common_graph[re[1]].type = Delivery(re[2], -re[3], i)
        requests[i] = request
    for depot in depots:
        R = random.randint(100, 200)
        common_graph[depot].type = Depot(R)
    battery_size = random.randint(200, 300)
    initial_energy = random.uniform(0.2, 0.9) * battery_size
    capacity = random.randint(50, 100)
    common_graph[start].type = Start(battery_size, initial_energy, capacity)
    common_graph[destination].type = Destination()
    return common_graph, requests


def generate_common_graph(node_num, lower_bound, high_bound):
    # 生成一个普通的大图，图中各个点并没有属性
    # 参数: 节点数量 坐标范围下界 坐标范围上界
    result = []
    coordinate_list = []
    x = []
    y = []
    for i in range(node_num):
        x.append(random.uniform(lower_bound, high_bound))
        y.append(random.uniform(lower_bound, high_bound))
    coordinates = list(zip(x, y))
    for i, coordinate in enumerate(coordinates):
        type = Commom()
        node = Node(i, coordinate, type)
        result.append(node)

    roads = []
    for i in range(node_num - 1):
        to = random.sample(range(i + 1, node_num), int((node_num - i - 1) / 2) + 1)
        for j in to:
            roads.append((i, j))

    for road in roads:
        t1 = result[road[0]]
        t2 = result[road[1]]
        t1_co = t1.coordinate
        t2_co = t2.coordinate
        length = ((t1_co[0] - t2_co[0]) ** 2 + (t1_co[1] - t2_co[1]) ** 2) ** 0.5
        time = length
        energy = length
        result[road[0]].add_road(road[1], length, time, energy)
        result[road[1]].add_road(road[0], length, time, energy)

    return result
