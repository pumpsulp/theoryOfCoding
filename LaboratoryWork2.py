import numpy as np
from typing import Optional, List, Dict, Tuple


# Функция для создания порождающей матрицы
def generate_G(k: int, n: int, custom_X: Optional[List[List[int]]] = None) -> np.ndarray:
    """
    Создает порождающую матрицу для кодирования.

    :param k: Количество информационных бит.
    :param n: Длина кодового слова (общее количество бит).
    :param custom_X: Опциональная матрица X для специфической генерации.
    :return: Порождающая матрица G.
    """
    I = np.eye(k, dtype=int)
    if custom_X is None:
        X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    else:
        X = np.array(custom_X)
    G = np.hstack((I, X))
    return G


# Функция для создания проверочной матрицы
def generate_H(X: np.ndarray) -> np.ndarray:
    """
    Создает проверочную матрицу для кодирования.

    :param X: Матрица X, которая является частью порождающей матрицы.
    :return: Проверочная матрица H.
    """
    I = np.eye(X.shape[1], dtype=int)
    H = np.hstack((X.T, I))
    return H


# Генерация синдромов
def generate_syndromes(H: np.ndarray) -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Генерирует синдромы для возможных ошибок.

    :param H: Проверочная матрица.
    :return: Словарь синдромов, где ключ - это синдром, а значение - вектор ошибки.
    """
    syndromes = {}
    for i in range(H.shape[1]):
        error_vector = np.zeros(H.shape[1], dtype=int)
        error_vector[i] = 1
        syndrome = np.dot(H, error_vector) % 2
        syndromes[tuple(syndrome)] = error_vector
    return syndromes


# Генерация кодового слова
def generate_codeword(data: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Генерирует кодовое слово из информационного слова.

    :param data: Информационное слово (вектор).
    :param G: Порождающая матрица.
    :return: Кодовое слово.
    """
    return np.dot(data, G) % 2


# Внесение ошибки
def introduce_error(codeword: np.ndarray, positions: List[int]) -> np.ndarray:
    """
    Вносит ошибку в кодовое слово на указанных позициях.

    :param codeword: Кодовое слово.
    :param positions: Список позиций для внесения ошибок.
    :return: Кодовое слово с внесенными ошибками.
    """
    for pos in positions:
        codeword[pos] ^= 1
    return codeword


# Вычисление синдрома
def calculate_syndrome(received_word: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Вычисляет синдром для принятого слова с ошибкой.

    :param received_word: Принятое слово (вектор с ошибкой).
    :param H: Проверочная матрица.
    :return: Синдром для принятого слова.
    """
    return np.dot(H, received_word) % 2


# Исправление ошибки
def correct_error(received_word: np.ndarray, syndrome: np.ndarray,
                  syndromes: Dict[Tuple[int, ...], np.ndarray]) -> np.ndarray:
    """
    Исправляет ошибку в принятом слове на основе синдрома.

    :param received_word: Принятое слово с ошибкой.
    :param syndrome: Синдром для принятого слова.
    :param syndromes: Словарь синдромов и соответствующих ошибок.
    :return: Исправленное слово.
    """
    if tuple(syndrome) in syndromes:
        error_vector = syndromes[tuple(syndrome)]
        corrected_word = (received_word + error_vector) % 2
        return corrected_word
    return received_word


# Основная функция для выполнения всех заданий
def main() -> None:
    """
    Основная функция для выполнения лабораторной работы по теории кодирования.
    Включает генерацию матриц, кодирование, внесение ошибок и исправление ошибок.
    """
    print("=== Лабораторная работа по теории кодирования ===\n")
    
    # Часть 1: (7, 4, 3)
    k, n = 4, 7
    print("Часть 1: Порождающая и проверочная матрицы для кода (7, 4, 3)")
    G = generate_G(k, n)
    print("Порождающая матрица G:\n", G)
    
    H = generate_H(G[:, k:])
    print("\nПроверочная матрица H:\n", H)
    
    syndromes = generate_syndromes(H)
    print("\nСиндромы для однократных ошибок:")
    for syndrome, error in syndromes.items():
        print(f"Синдром {syndrome}: Ошибка {error}")
    
    # 2.4: Пример кодирования и внесения однократной ошибки
    data_word = np.array([1, 0, 1, 1])
    codeword = generate_codeword(data_word, G)
    print("\nКодовое слово:", codeword)
    
    received_word = introduce_error(codeword.copy(), [2])
    print("Кодовое слово с однократной ошибкой:", received_word)
    
    syndrome = calculate_syndrome(received_word, H)
    print("Синдром:", syndrome)
    
    corrected_word = correct_error(received_word, syndrome, syndromes)
    print("Исправленное слово:", corrected_word)
    print("Проверка совпадения с исходным:", np.array_equal(corrected_word, codeword))
    
    # 2.5: Пример с двукратной ошибкой
    received_word_double = introduce_error(codeword.copy(), [1, 5])
    print("\nКодовое слово с двукратной ошибкой:", received_word_double)
    
    syndrome_double = calculate_syndrome(received_word_double, H)
    print("Синдром для двукратной ошибки:", syndrome_double)
    
    corrected_word_double = correct_error(received_word_double, syndrome_double, syndromes)
    print("Попытка исправления двукратной ошибки:", corrected_word_double)
    print("Полученное слово отличается от исходного:", not np.array_equal(corrected_word_double, codeword))
    
    # Часть 2: (10, 5, 5)
    k2, n2 = 5, 10
    print("\nЧасть 2: Порождающая и проверочная матрицы для кода (10, 5, 5)")
    custom_X = [[1, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 1, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 1]]
    G2 = generate_G(k2, n2, custom_X)
    print("Порождающая матрица G2:\n", G2)
    
    H2 = generate_H(G2[:, k2:])
    print("\nПроверочная матрица H2:\n", H2)
    
    double_error_syndromes = generate_syndromes(H2)
    print("\nСиндромы для двукратных ошибок:")
    for syndrome, error in double_error_syndromes.items():
        print(f"Синдром {syndrome}: Ошибка {error}")
    
    # 2.9: Кодирование и внесение однократной ошибки
    data_word_2 = np.array([1, 0, 1, 1, 0])
    codeword_2 = generate_codeword(data_word_2, G2)
    print("\nКодовое слово (d=5):", codeword_2)
    
    received_word_2 = introduce_error(codeword_2.copy(), [3])
    print("Кодовое слово с однократной ошибкой:", received_word_2)
    
    syndrome_2 = calculate_syndrome(received_word_2, H2)
    print("Синдром:", syndrome_2)
    
    corrected_word_2 = correct_error(received_word_2, syndrome_2, double_error_syndromes)
    print("Исправленное слово:", corrected_word_2)
    print("Проверка совпадения с исходным:", np.array_equal(corrected_word_2, codeword_2))
    
    # 2.10: Внесение двукратной ошибки
    received_word_2_double = introduce_error(codeword_2.copy(), [2, 8])
    print("\nКодовое слово с двукратной ошибкой:", received_word_2_double)
    
    syndrome_2_double = calculate_syndrome(received_word_2_double, H2)
    print("Синдром для двукратной ошибки:", syndrome_2_double)
    
    corrected_word_2_double = correct_error(received_word_2_double, syndrome_2_double, double_error_syndromes)
    print("Попытка исправления двукратной ошибки:", corrected_word_2_double)
    print("Полученное слово отличается от исходного:", not np.array_equal(corrected_word_2_double, codeword_2))
    
    # 2.11: Внесение трёхкратной ошибки
    received_word_2_triple = introduce_error(codeword_2.copy(), [1, 4, 9])
    print("\nКодовое слово с трёхкратной ошибкой:", received_word_2_triple)
    
    syndrome_2_triple = calculate_syndrome(received_word_2_triple, H2)
    print("Синдром для трёхкратной ошибки:", syndrome_2_triple)
    
    corrected_word_2_triple = correct_error(received_word_2_triple, syndrome_2_triple, double_error_syndromes)
    print("Попытка исправления трёхкратной ошибки:", corrected_word_2_triple)
    print("Полученное слово отличается от исходного:", not np.array_equal(corrected_word_2_triple, codeword_2))


if __name__ == '__main__':
    main()
