import numpy
from tqdm import tqdm


def create_population(amount_of_individuals, dim, min_element, max_element):
    # Создаем популяцию
    population = []
    for _ in range(amount_of_individuals):
        population.append(numpy.random.uniform(min_element, max_element, dim))
    return numpy.array(population)


def choose_best_individual(population, func_population):
    # Данная функция находит лучшую особь в популяции
    func_list = list(func_population)
    best_index = func_list.index(min(func_list))
    return population[best_index]


def iteration(func, population, func_population, f, p):
    # Создаем необходимые матрицы, перемешиванием матрицы популяции
    partners_matrix = numpy.random.permutation(population)
    a_matrix = numpy.random.permutation(population)
    b_matrix = numpy.random.permutation(population)
    # Мутировавший партнер вычисляется по соотвествующей формуле
    mutation_matrix = partners_matrix + f * (a_matrix - b_matrix)
    # Далее мы создаем "маску". Если на месте с инедксами i, j  в маске стоит единица, то соотвествующий элемент потомка
    # берется из мутировавшего партнера. Если 0 - то из исходного.
    # Для начала создаем случайную матрицу, заполненную числами от 0 до 1 с равномерным распределением
    random_matrix = numpy.random.random(population.shape)
    # Затем сравниваем эту матрицу с нужной вероятноостью выпадения единицы. После сравнения у нас получится матрица,
    # каждый элемент которой есть булевская переменная, причем значения True будут в ней находится с вероятностью p,
    # а False - 1-p. Затем, после домножения на 1 True превратится в единиуц, а False в ноль.
    mask = (random_matrix < p) * 1
    # Затем мы получаем матрицу потомков
    child_matrix = mask * mutation_matrix - (mask - 1) * population
    # Вычисляем значения оптимизируемой функции на потомках
    child_funcs = numpy.array(list(map(func, child_matrix)))
    # Аналогично, получаем маску для выбора лучшей особей
    func_mask = (child_funcs < func_population) * 1
    reshaped_func_mask = func_mask.reshape(func_mask.size, 1)
    # Получаем новую популяцию
    new_population = reshaped_func_mask * child_matrix - (reshaped_func_mask - 1) * population
    # И новый список значений функции особей
    new_func_population = func_mask * child_funcs - (func_mask - 1) * func_population

    return new_population, new_func_population


def optimize(func, amount_of_individuals, dim, end_cond, end_method='max_iter', f=0.5, p=0.9, min_element=-1,
             max_element=1, debug_pop_print=-1):
    # func - оптимизиуемая функция, должна принмать в качетсве параметра массив numpy.array размерности dim
    # dim - размерность
    # amount_of_individuals - количество особей
    # f - сила мутации
    # p - вероятность того, что в потомке элемент будет взят из второго партнера
    population = create_population(amount_of_individuals, dim, min_element, max_element)  # Создаем популяцию
    # Каждый массив: numpy.array([1, 2, ..., amount_of_individuals])
    func_population = numpy.array(list(map(lambda item: func(item), population)))  # Вычисляем для каждой особи в
    # популяции значении функции
    if end_method == 'max_iter':
        if debug_pop_print == -1:
            for _ in tqdm(range(end_cond)):
                population, func_population = iteration(func, population, func_population, f, p)
        else:
            for i in tqdm(range(end_cond)):
                if i % debug_pop_print == 0:
                    print(population)
                    print('-------------------------------')
                population, func_population = iteration(func, population, func_population, f, p)
    return choose_best_individual(population, func_population)
