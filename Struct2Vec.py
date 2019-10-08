import torch
import torch.nn as nn


class struct2vec_model(nn.Module):
    def __init__(self, p_dim):
        super(struct2vec_model, self).__init__()
        self.theta1_linear = nn.Linear(p_dim, p_dim, bias=False)
        self.theta2_linear = nn.Linear(p_dim, p_dim, bias=False)
        self.theta3_linear = nn.Linear(1, p_dim, bias=False)
        self.theta4_linear = nn.Linear(p_dim, p_dim, bias=False)
        self.theta5_linear = nn.Linear(1, p_dim, bias=False)
        self.theta6_linear = nn.Linear(p_dim, p_dim, bias=False)
        self.theta7_linear = nn.Linear(1, p_dim, bias=False)
        self.theta8_Start_linear = nn.Linear(4, p_dim, bias=False)
        self.theta8_Pick_linear = nn.Linear(7, p_dim, bias=False)
        self.theta8_Delivery_linear = nn.Linear(2, p_dim, bias=False)
        self.theta8_Depot_linear = nn.Linear(1, p_dim, bias=False)
        self.theta8_Destination_linear = nn.Linear(1, p_dim, bias=False)

    def forward(self, name, xi, mu_N, wi, ui, ti):
        # xi:(xi_dim); mu_N:(|N(i)|, p_dim); wi:(|N(i)|, 1); ui:(|N(i)|, 1); ti:(|N(i)|, 1)
        tmp = self.theta1_linear(torch.sum(mu_N, 0)) + self.theta2_linear(
            torch.sum(torch.relu(self.theta3_linear(wi)), 0)) \
              + self.theta4_linear(torch.sum(torch.relu(self.theta5_linear(ui)), 0)) \
              + self.theta6_linear(torch.sum(torch.relu(self.theta7_linear(ti)), 0))
        if name == "Start":
            mu = torch.relu(tmp + self.theta8_Start_linear(xi))
        elif name == "Pick":
            mu = torch.relu(tmp + self.theta8_Pick_linear(xi))
        elif name == "Delivery":
            mu = torch.relu(tmp + self.theta8_Delivery_linear(xi))
        elif name == "Depot":
            mu = torch.relu(tmp + self.theta8_Depot_linear(xi))
        else:
            mu = torch.relu(tmp)

        return mu  # (p_dim)


def Struct2Vec(graph, p_dim=128, R=4):
    # R denotes the iterations of var mu
    node_list = graph  # 小图的节点列表
    ser_num_list = []  # mapping table

    for node in node_list:
        ser_num_list.append(node.serial_number)

    for node in node_list:
        node.serial_number = ser_num_list.index(node.serial_number)
        for edge in node.edges:
            edge.to = ser_num_list.index(edge.to)

    # print the type of node
    for node in node_list:
        print(node.type.name, node.serial_number)

    node_num = len(node_list)  # 小图的节点总数

    mu_all = torch.zeros(node_num, p_dim)
    struct2vec = struct2vec_model(p_dim)
    x_all = []  # 存所有xi
    for _ in range(R):
        for node in node_list:
            mu_N = []
            wi = []
            ui = []
            ti = []
            for edge in node.edges:
                mu_N.append(mu_all[edge.to].unsqueeze(0))
                wi.append(edge.length)
                ui.append(edge.energy)
                ti.append(edge.time)
            if len(mu_N) > 0:
                mu_N = torch.cat(mu_N)
            else:
                mu_all[node.serial_number] = torch.zeros(p_dim)
                continue
            wi = torch.Tensor(wi).unsqueeze(1)
            ui = torch.Tensor(ui).unsqueeze(1)
            ti = torch.Tensor(ti).unsqueeze(1)
            xi = []

            if node.type.name == "Start":
                xi.append(node.type.battery_size)
                xi.append(node.type.initial_energy)
                xi.append(node.type.capacity)
                xi.append(node.type.used_capacity)
                xi = torch.Tensor(xi)

            elif node.type.name == "Pick":
                xi.append(node.type.pickup_deadline)
                xi.append(node.type.capacity_required)
                xi.append(node.type.time)
                xi.append(node.type.distance)
                xi.append(node.type.energy)
                xi.append(node.type.Hq)
                xi.append(node.type.constant)
                xi = torch.Tensor(xi)

            elif node.type.name == "Delivery":
                xi.append(node.type.delivery_deadline)
                xi.append(node.type.capacity_required)
                xi = torch.Tensor(xi)

            elif node.type.name == "Depot":
                xi.append(node.type.R)
                xi = torch.Tensor(xi)

            else:
                xi = torch.Tensor([0])

            mu_all[node.serial_number] = struct2vec(node.type.name, xi, mu_N, wi, ui, ti)
            x_all.append(xi)

        # print(list(struct2vec.named_parameters()))

        return x_all, mu_all, ser_num_list


if __name__ == '__main__':
    import numpy as np
    from GenetateBigGraph import generate_big_graph
    from TourGraphCreation import single_car_tour_graph

    torch.set_printoptions(threshold=np.nan)  # show all data
    graph, requests = generate_big_graph(node_num=10, lower_bound=1, high_bound=100, request_num=3, depot_num=1)
    graph = single_car_tour_graph(graph, requests)
    x_all, mu_all, ser_num_list = Struct2Vec(graph)
    print(x_all)
    print(mu_all)
    print(ser_num_list)
