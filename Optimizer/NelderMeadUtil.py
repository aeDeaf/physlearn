import sys

import numpy


def create_points(dim, min_element, max_element):
    points = []
    for i in range(dim + 1):  # Создаем массив точек размера dim + 1 (так требуется по методу)
        points.append(numpy.random.uniform(min_element, max_element, dim))
    return numpy.array(points)


def find_points(y_points):
    # В данном методе мы находим три точки: h - точка с наибользим значением оптимизируемой функции,
    # g - следующая за точкой с наибольишм значение, l - с наименьшим значением.
    # Далее мы задаем начальные парамтры...
    h_point = -sys.maxsize
    g_point = -sys.maxsize
    l_point = sys.maxsize
    h_index = 0
    g_index = 0
    l_index = 0
    # ...и проводим стандарнтый поиск.
    for index, item in enumerate(y_points):
        if item > h_point:
            g_point = h_point
            h_point = item
            g_index = h_index
            h_index = index
        elif (item > g_point) and (item != h_point):
            g_point = item
            g_index = index
        if item < l_point:
            l_point = item
            l_index = index

    return h_index, g_index, l_index


def calculate_center(x_points, h_index):
    # Вычисляем "центр масс" всех точек, за исключением h
    sum_data = 0
    n = len(x_points) - 1
    for index, item in enumerate(x_points):
        if index != h_index:
            sum_data += item

    return sum_data / n


def calculate_reflected_point(x_h, x_center, alpha):
    # В данной функции выполняется отражение точки h относительно центра масс
    x_reflected = ((1 + alpha) * x_center) - (alpha * x_h)
    return x_reflected


def calculate_stretched_point(x_reflected, x_center, gamma):
    # В данной функции выполняется растяжение в направлении, соединяющим h, center и reflected
    x_stretch = (gamma * x_reflected) + ((1 - gamma) * x_center)
    return x_stretch


def calculate_compressed_point(x_h, x_center, beta):
    # В данной функции выполняется сжатие к center
    x_compress = (beta * x_h) + ((1 - beta) * x_center)
    return x_compress


def compress_simplex(x, x_l):
    # В данной функции происходит схатие всего симплекса к l
    x_points = x
    for index, x_i in enumerate(x_points):
        x_points[index] = 0.5 * (x_i + x_l)
    return x_points


def average(data):
    # Вычисление среднего значения
    return sum(data) / len(data)


def variance(data):
    # Вычисление дисперсии
    mean_data = average(data)
    sum_var = 0
    for item in data:
        sum_var += (item - mean_data) ** 2
    return sum_var / len(data)
