import copy
import random

import numpy
from tqdm import tqdm


def create_population(amount_of_individuals, dim, min_element, max_element):
    # Создаем популяцию
    population = []
    for _ in range(amount_of_individuals):
        population.append(numpy.random.uniform(min_element, max_element, dim))
    return numpy.array(population)


def choose_partner(cur_index, population):
    parent_index = cur_index
    while parent_index == cur_index:
        parent_index = random.randint(0, len(population) - 1)
    return population[parent_index], parent_index


def choose_two_random_vectors(cur_index, parent_index, population):
    a_index = cur_index
    b_index = cur_index
    while (a_index == cur_index) or (b_index == cur_index) or (a_index == parent_index) or (b_index == parent_index):
        a_index = random.randint(0, len(population) - 1)
        b_index = random.randint(0, len(population) - 1)
    return population[a_index], population[b_index]


def mutation(partner, a_vector, b_vector, f):
    # Мутация
    return partner + f * (a_vector - b_vector)


def cross(first_partner, second_partner, p):
    # Создаем заполненный нулями массив соотвествующего размера
    child = numpy.zeros_like(first_partner)
    for index, _ in enumerate(first_partner):
        random_value = random.random()
        if random_value <= p:  # Если случайная величина меньше p - то берем элемент из второго парнтера...
            child[index] = second_partner[index]
        else:  # ...иначе берем от первого
            child[index] = first_partner[index]
    return child


def iteration(func, population, func_population, partner_array, a_array, b_array, f, p):
    # Перемешиваем массивы partner_array, a_array, b_array
    numpy.random.shuffle(partner_array)
    numpy.random.shuffle(a_array)
    numpy.random.shuffle(b_array)
    for index, first_partner in enumerate(population):  # Проходим в цикле по каждой особи в популяции
        first_partner_func = func_population[index]  # Находим соответсвующее значение функции
        #partner, partner_index = choose_partner(index, population)
        #a_vector, b_vector = choose_two_random_vectors(index, partner_index, population)
        partner = population[partner_array[index]]  # Находим партнера
        a_vector = population[partner_array[index]]  # Находим вектор a...
        b_vector = population[partner_array[index]]  # ...и b.
        mutation_partner = mutation(partner, a_vector, b_vector, f)  # Проводим мутацию
        child = cross(first_partner, mutation_partner, p)  # Делаем потомка
        child_func = func(child)  # Вычисляем значение функции потомка
        #print(first_partner, mutation_partner)
        #print(child)
        #print('---------------------')
        if child_func <= first_partner_func:  # Если она ниже, чем у исходной особи - то заменяем ее
            population[index] = child
            func_population[index] = child_func
    return population, func_population


def choose_best_individual(population, func_population):
    # Данная функция находит лучшую особь в популяции
    func_list = list(func_population)
    best_index = func_list.index(min(func_list))
    return population[best_index]


def create_indexes_arrays(amount_of_individuals):
    partner_array = numpy.array([i for i in range(amount_of_individuals)])
    a_array = copy.deepcopy(partner_array)
    b_array = copy.deepcopy(partner_array)
    return partner_array, a_array, b_array


def optimize(func, amount_of_individuals, dim, end_cond, end_method='max_iter', f=0.5, p=0.9, min_element=-1,
             max_element=1):
    # func - оптимизиуемая функция, должна принмать в качетсве параметра массив numpy.array размерности dim
    # dim - размерность
    # amount_of_individuals - количество особей
    # f - сила мутации
    # p - вероятность того, что в потомке элемент будет взят из второго партнера
    population = create_population(amount_of_individuals, dim, min_element, max_element)  # Создаем популяцию
    partner_array, a_array, b_array = create_indexes_arrays(amount_of_individuals)  # Создаем три массива
    # Каждый массив: numpy.array([1, 2, ..., amount_of_individuals])
    # Каждый элемент массива - это индекс особи в поппуляции, взятой в нужной роли (как второй партнер или как один из
    # двух векторов A и B, которые учавствуют в дальнейшем в мутации).
    # Далее на каждой итерации эти массивы перемещиваются случайным образом.
    func_population = numpy.array(list(map(lambda i: func(i), population)))  # Вычисляем для каждой особи в популяции
    # значении функции
    if end_method == 'max_iter':
        for _ in tqdm(range(end_cond)):
            population, func_population = iteration(func, population, func_population, partner_array, a_array,
                                                    b_array, f, p)
    return choose_best_individual(population, func_population)
