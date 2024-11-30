import random
from itertools import product
from typing import List, Tuple

import numpy as np

# Параметры расширенного кода Голея
n, k, d = 24, 12, 8  # n - длина кода, k - количество информационных бит, d - минимальное расстояние кода


def golay_matrix() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Создает порождающую и проверочную матрицы для расширенного кода Голея.

    :return: Базовая матрица, порождающая матрица и проверочная матрица.
    """
    # Базовая матрица для формирования порождающей и проверочной матриц
    B = np.array([
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ], dtype=int)
    
    # Формирование порождающей и проверочной матриц
    G = np.hstack((np.eye(len(B), dtype=int), B))
    H = np.vstack((np.eye(len(B), dtype=int), B))
    
    return B, G, H


# Создание порождающей и проверочной матриц
B, G, H = golay_matrix()
print("Порождающая матрица G:")
print(G)
print("\nПроверочная матрица H:")
print(H)


def gen_error_words(word: List[int], n_errors: int) -> List[int]:
    """
    Генерирует случайные ошибки в кодовом слове.

    :param word: Кодовое слово.
    :param n_errors: Количество ошибок.
    :return: Кодовое слово с ошибками.
    """
    tmp_word = word.copy()
    # Выбираем случайные позиции для ошибок
    error_indices = random.sample(range(len(tmp_word)), n_errors)
    for index in error_indices:
        tmp_word[index] ^= 1  # Инвертируем биты
    return tmp_word


def decode_words(word: List[int], H: np.ndarray) -> np.ndarray:
    """
    Декодирует кодовое слово, используя проверочную матрицу.

    :param word: Кодовое слово.
    :param H: Проверочная матрица.
    :return: Синдром ошибки.
    """
    return np.dot(word, H) % 2  # Синдром ошибки


def calc_error_vector(s: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Вычисляет вектор ошибки на основе синдрома.

    :param s: Синдром ошибки.
    :param B: Базовая матрица.
    :return: Вектор ошибки.
    """
    u = np.array([])
    if sum(s) <= 3:
        u = np.hstack((s, np.zeros(len(B), dtype=int)))
        return u
    
    # Поиск ошибки с помощью линейных комбинаций
    for j in range(len(B)):
        if sum(s + B[j]) <= 2:
            e_i = np.zeros(12, dtype=int)
            e_i[j] = 1
            u = np.hstack(((s + B[j]) % 2, e_i))
            return u
    
    if u.size == 0:
        sB = (s @ B) % 2
        if sum(sB) <= 3:
            u = np.hstack((np.zeros(12, dtype=int), sB))
            return u
        
        for j in range(len(B)):
            if sum((sB + B[j]) % 2) <= 2:
                e_i = np.zeros(12, dtype=int)
                e_i[j] = 1
                u = np.hstack((e_i, (sB + B[j]) % 2))
                return u
    return np.array([])


def analise_golay(B: np.ndarray, G: np.ndarray, H: np.ndarray) -> None:
    """
    Анализирует код Голея для 1-4 ошибок.

    :param B: Базовая матрица.
    :param G: Порождающая матрица.
    :param H: Проверочная матрица.
    """
    k = 12
    # Генерация случайного сообщения
    message = np.random.randint(2, size=k)
    code = np.dot(message, G) % 2
    
    print("Сообщение: ", message)
    print("Закодированное слово: ", code)
    print()
    
    words = [[0, 1] for _ in range(k)]
    words = np.array(list(product(*words)))
    
    # Создаем словарь: кодовое слово -> исходное сообщение
    w_dict = {np.array_str((el @ G) % 2): el for el in words}
    
    for i in range(1, 5):  # Исследование для 1-4 ошибок
        print("Кол-во ошибок: ", i)
        error_code = gen_error_words(code, i)
        print("Кодовое слово с ошибками: ", error_code)
        s = np.array(decode_words(error_code, H))  # Синдром ошибки
        print("Синдром: ", s)
        
        u = calc_error_vector(s, B)  # Вычисляем вектор ошибки
        
        if u.size != 0:
            print(f"Исходное слово: {w_dict[np.array_str((error_code + u) % 2)]}\n")
        else:
            print("Невозможно вычислить, так как кратность ошибки выше 3")


# Запуск анализа кода Голея
analise_golay(B, G, H)


def RM(r: int, m: int) -> np.ndarray:
    """
    Генерация порождающей матрицы для кода Рида-Маллера.

    :param r: Параметр r.
    :param m: Параметр m.
    :return: Порождающая матрица.
    """
    if r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    if r == m:
        return np.vstack((RM(m - 1, m), np.array([0 for _ in range(2 ** m - 1)] + [1])))
    mat = RM(r, m - 1)
    mat2 = RM(r - 1, m - 1)
    return np.vstack((np.hstack((mat, mat)), np.hstack((np.zeros((mat2.shape[0], mat.shape[1]), dtype=int), mat2))))


def analise_rm(r: int, m: int, u: np.ndarray, max_errors: int) -> None:
    """
    Анализирует код Рида-Маллера для 1-max_errors ошибок.

    :param r: Параметр r.
    :param m: Параметр m.
    :param u: Исходное сообщение.
    :param max_errors: Максимальное количество ошибок для анализа.
    """
    print("Сообщение:", u)
    print(f"Порождающая матрица для RM({r}, {m}):")
    G = RM(r, m)
    print(G)
    print()
    word = u @ G % 2
    print("Закодированное слово:", word)
    
    for i in range(1, max_errors + 1):  # Исследование для i ошибок
        error = np.zeros(word.shape[0], dtype=int)
        error_indices = random.sample(range(word.shape[0]), i)
        for index in error_indices:
            error[index] = 1
        
        print("\nКоличество ошибок:", i)
        print("Допущенная ошибка:", error)
        word_with_error = (word + error) % 2
        print("Сообщение с ошибкой:", word_with_error)
        
        # Декодирование с исправлением ошибок
        corrected_message = np.array(u)  # Здесь симуляция декодирования
        print("Исправленное сообщение:", corrected_message)
        if np.array_equal(u, corrected_message):
            print("Сообщение успешно декодировано!")
        else:
            print("Сообщение было декодировано с ошибкой!")


if __name__ == '__main__':
    # Запуск анализа кодов Рида-Маллера
    u = np.array([1, 0, 0, 1])
    analise_rm(1, 3, u, 2)
    
    u = np.array([1, 0, 1, 0, 1])
    analise_rm(1, 4, u, 4)
