class Start:
    def __init__(self, battery_size=0, initial_energy=0, capacity=0, used_capacity=0):
        # 电池大小，初始电量，车辆总容量，已用容量
        self.battery_size = battery_size
        self.initial_energy = initial_energy
        self.capacity = capacity
        self.used_capacity = used_capacity
        self.name = "Start"


class Pick:
    def __init__(self, pickup_deadline, capacity_required, time=0, distance=0, energy=0, Hq=0, constant=0):
        # 取包裹最晚时间，货物大小，货物起点到终点的时间，货物起点到终点的距离，货物起点到终点消耗的能量，是否有其他车将在下一站取得该包裹，常量0
        self.pickup_deadline = pickup_deadline
        self.capacity_required = capacity_required
        self.time = time
        self.distance = distance
        self.energy = energy
        self.Hq = Hq
        self.constant = constant
        self.name = "Pick"


class Delivery:
    def __init__(self, delivery_deadline, capacity_required):
        # 送货期限，货物大小的相反数
        self.delivery_deadline = delivery_deadline
        self.capacity_required = capacity_required
        self.name = "Delivery"


class Depot:
    def __init__(self, R):
        # 充电速率
        self.R = R
        self.name = "Depot"


class Destination:
    def __init__(self):
        self.a = None
        self.name = "Destination"


class Commom:
    def __init__(self):
        self.a = None
        self.name = "Common"


class Road:
    def __init__(self, to, length, time, energy):
        # 连接的另一个端点 道路长度 车辆走过这段路所用的时间 车辆走过这段路所消耗的电量
        self.to = to  # serial_number
        self.length = length
        self.time = time
        self.energy = energy


class Node:
    def __init__(self, serial_number, coordinate, type):
        # 节点序号 坐标 节点类型
        self.serial_number = serial_number  # node number
        self.coordinate = coordinate
        self.type = type  # 类的对象
        self.edges = []   # a list of Road object

    def add_road(self, to, length, time, energy):
        road = Road(to, length, time, energy)
        self.edges.append(road)
