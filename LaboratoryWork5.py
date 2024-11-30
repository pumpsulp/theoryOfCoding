import math
import random as rnd
from itertools import combinations

import numpy as np


def get_basis_order(n: int, m: int) -> list[str]:
    """
    Генерация базисного порядка для кода Рида-Маллера.

    :param n: Длина кода.
    :param m: Количество битов в кодовом слове.
    :return: Список строк, представляющих двоичные числа от 0 до n-1 в обратном порядке.
    """
    ans = []
    for i in range(n):
        binary = f'{i:b}'[::-1]  # Перевод в двоичное представление и разворот
        if len(binary) < m:
            binary += '0' * (m - len(binary))  # Добавление недостающих нулей
        ans.append(binary)
    return ans


def get_vectors_order(r: int, m: int) -> list[list[int]]:
    """
    Генерация порядка векторов для кода Рида-Маллера.

    :param r: Параметр кода Рида-Маллера.
    :param m: Количество битов в кодовом слове.
    :return: Список векторов, представляющих комбинации индексов.
    """
    elements = list(range(m))
    ans = []
    for i in range(r + 1):
        combinations_list = sorted(list(combinations(elements, i)), reverse=True)
        for combination in combinations_list:
            ans.append(list(combination))
    return ans


def get_rm_G_matr(r: int, m: int) -> tuple[np.ndarray, list[str], list[list[int]]]:
    """
    Формирование порождающей матрицы кода Рида-Маллера.

    :param r: Параметр кода Рида-Маллера.
    :param m: Количество битов в кодовом слове.
    :return: Порождающая матрица, базисный порядок и порядок векторов.
    """
    n = 2 ** m  # Длина кода
    basis_order = get_basis_order(n, m)  # Базисный порядок
    vectors_order = get_vectors_order(r, m)  # Порядок векторов
    g_matr = np.zeros((len(vectors_order), n), dtype=int)
    
    for i in range(g_matr.shape[0]):
        for j in range(g_matr.shape[1]):
            flag = True
            for indx in vectors_order[i]:
                if basis_order[j][indx] == '1':
                    g_matr[i][j] = 0
                    flag = False
                    break
            if flag:
                g_matr[i][j] = 1
    
    print("Промежуточный вывод: порождающая матрица кода Рида-Маллера")
    print(g_matr)
    print("Промежуточный вывод: базисный порядок")
    print(basis_order)
    print("Промежуточный вывод: порядок векторов")
    print(vectors_order)
    
    return g_matr, basis_order, vectors_order


def get_complement(m: int, I: list[int]) -> list[int]:
    """
    Нахождение дополнения множества I в пределах множества от 0 до m-1.

    :param m: Количество битов в кодовом слове.
    :param I: Множество индексов.
    :return: Дополнение множества I в пределах от 0 до m-1.
    """
    Zm = list(range(m))
    return [i for i in Zm if i not in I]


def get_Hj(g_matr: np.ndarray, basis_order: list[str], vectors_order: list[list[int]],
           Jc: list[int], m: int) -> list[str]:
    """
    Вычисление множества Hj для заданного множества Jc.

    :param g_matr: Порождающая матрица кода Рида-Маллера.
    :param basis_order: Базисный порядок.
    :param vectors_order: Порядок векторов.
    :param Jc: Множество индексов для вычисления Hj.
    :param m: Количество битов в кодовом слове.
    :return: Список строк, представляющих элементы множества Hj.
    """
    Hj = []
    J = list(Jc)
    if J == list(range(m)):
        str_var = vectors_order.index([])  # Для случая, когда J = {0, ..., m-1}
    else:
        str_var = vectors_order.index(J)
    
    for i in range(len(g_matr[str_var])):
        if g_matr[str_var][i] == 1:
            Hj.append(basis_order[i])
    return Hj


def get_V(Jc: list[int], basis_order: list[str], hj: str) -> list[int]:
    """
    Вычисление вектора V для заданных Jc и hj.

    :param Jc: Множество индексов для вычисления V.
    :param basis_order: Базисный порядок.
    :param hj: Строка, представляющая элемент множества Hj.
    :return: Вектор, представляющий вычисленное значение V.
    """
    v = []
    for pos in basis_order:
        flag = True
        for j in Jc:
            if pos[j] != hj[j]:
                v.append(0)
                flag = False
                break
        if flag:
            v.append(1)
    return v


def get_Mj(W: list[int], m: int, basis_order: list[str], r: int,
           g_matr: np.ndarray, vectors_order: list[list[int]]) -> dict[tuple[int], int]:
    """
    Мажоритарное декодирование для кода Рида-Маллера.

    :param W: Принятое сообщение с ошибками.
    :param m: Количество битов в кодовом слове.
    :param basis_order: Базисный порядок.
    :param r: Параметр кода Рида-Маллера.
    :param g_matr: Порождающая матрица кода Рида-Маллера.
    :param vectors_order: Порядок векторов.
    :return: Маппинг множества индексов к значениям для декодированного сообщения.
    """
    M = {}
    for I in range(r, -1, -1):
        if I == r:
            w = W
        else:
            for key in sorted(M):
                if len(key) == I + 1 and M[key] == 1:
                    _w = w
                    w = []
                    v = g_matr[vectors_order.index(list(key))]
                    for e in range(len(_w)):
                        w.append((_w[e] + v[e]) % 2)
                    break
        J = sorted(list(combinations(range(m), I)))
        for j in J:
            Jc = get_complement(m, j)
            Hj = get_Hj(g_matr, basis_order, vectors_order, j, m)
            count1 = 0
            count0 = 0
            for hj in Hj:
                V = get_V(Jc, basis_order, hj)
                
                temp = []
                s = 0
                for k in range(len(V)):
                    temp.append((V[k] or w[k]))
                    s += temp[-1] if temp[-1] == 1 else 0
                if Jc == list(range(m)):
                    M[j] = 0
                    break
                if ((s + 1) % 2) == 1:
                    count1 += 1
                else:
                    count0 += 1
                
                if count1 > 2 ** (m - I - 1):
                    M[j] = 1
                    break
                elif count0 > 2 ** (m - I - 1):
                    M[j] = 0
                    break
    return M


def get_err_word(g_matr: np.ndarray, r: int, basis_order: list[str], vectors_order: list[list[int]],
                 t: int) -> None:
    """
    Моделирование ошибок и декодирования кода Рида-Маллера.

    :param g_matr: Порождающая матрица кода Рида-Маллера.
    :param r: Параметр кода Рида-Маллера.
    :param basis_order: Базисный порядок.
    :param vectors_order: Порядок векторов.
    :param t: Количество ошибок, которые нужно ввести.
    """
    m = int(math.log2(g_matr.shape[1]))
    row = g_matr.shape[0]
    
    # Генерация случайного кодового слова
    idx = rnd.randint(0, row - 1)
    word = np.array(g_matr[idx][:row])
    w = np.dot(word, g_matr) % 2
    print(f"Исходное сообщение: {word}")
    print(f"Отправленное сообщение: {w}")
    
    # Введение ошибок
    for i in range(t):
        w[i] += 1
        w[i] %= 2
    print(f"Принятое сообщение с ошибкой: {w}")
    
    # Декодирование с мажоритарным методом
    M = get_Mj(w, m, basis_order, r, g_matr, vectors_order)
    u = []
    for i, j in M.items():
        u.append(j)
    u = u[::-1]
    print(f"Изменённое сообщение после преобразования: {u}")
    
    # Декодирование и вывод результата
    try:
        decoded_message = np.dot(u, g_matr) % 2
        print(f"Декодированное сообщение: {decoded_message}")
    except:
        print("Произошла ошибка, необходимо повторно отправить сообщение")


if __name__ == '__main__':
    # Тестирование кода Рида-Маллера
    r, m = 2, 4
    rm_g_matr, basis_order, vectors_order = get_rm_G_matr(r, m)
    t_list = [1, 2]
    
    for t in t_list:
        print(f"\nЭкспериментальная проверка декодирования кода Рида-Маллера RM({r}, {m}) при t = {t}")
        get_err_word(rm_g_matr, r, basis_order, vectors_order, t)
        if t != t_list[-1]:
            print("")  # Пустая строка для разделения экспериментов
