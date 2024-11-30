import numpy as np
import random


def polynomial_division_remainder(dividend: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """
    Выполняет деление многочленов с остатком.

    :param dividend: Многочлен-делимое (вектор коэффициентов).
    :param divisor: Многочлен-делитель (вектор коэффициентов).
    :return: Остаток от деления многочленов.
    """
    remainder = list(dividend)  # Копия делимого
    len_divisor = len(divisor)
    
    while len(remainder) >= len_divisor:
        shift = len(remainder) - len_divisor
        for i in range(len_divisor):
            remainder[shift + i] ^= divisor[i]  # XOR между делителем и остатком
        while len(remainder) > 0 and remainder[-1] == 0:
            remainder.pop()  # Удаляем нули из конца
    
    return np.array(remainder, dtype=int)


def polynomial_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Умножает два многочлена.

    :param A: Многочлен A (вектор коэффициентов).
    :param B: Многочлен B (вектор коэффициентов).
    :return: Результат умножения многочленов.
    """
    degree_A = len(A)
    degree_B = len(B)
    result = np.zeros(degree_A + degree_B - 1, dtype=int)
    
    for i in range(degree_A):
        for j in range(degree_B):
            result[i + j] ^= A[i] & B[j]  # Побитовая операция AND с XOR для сложения коэффициентов
    
    return result


def generate_random_message(length: int) -> np.ndarray:
    """
    Генерирует случайное сообщение длины `length` для кодирования.

    :param length: Длина генерируемого сообщения.
    :return: Случайное бинарное сообщение.
    """
    return np.random.randint(0, 2, size=length, dtype=int)


def encode(message: np.ndarray, generator: np.ndarray) -> np.ndarray:
    """
    Кодирует сообщение с использованием порождающего многочлена.

    :param message: Сообщение для кодирования (вектор коэффициентов).
    :param generator: Порождающий многочлен (вектор коэффициентов).
    :return: Закодированное сообщение.
    """
    codeword = polynomial_multiply(message, generator)  # Умножаем на порождающий многочлен
    codeword %= 2  # Приводим к двоичной арифметике
    return codeword


def add_errors(codeword: np.ndarray, error_count: int) -> np.ndarray:
    """
    Добавляет одиночные или многократные ошибки в закодированное сообщение.

    :param codeword: Закодированное сообщение.
    :param error_count: Количество ошибок, которое нужно добавить.
    :return: Сообщение с добавленными ошибками.
    """
    codeword_with_errors = codeword.copy()
    error_positions = random.sample(range(len(codeword)), error_count)
    
    for pos in error_positions:
        codeword_with_errors[pos] ^= 1  # Инвертируем бит
    
    print(f"Ошибки добавлены в позиции: {error_positions}")
    return codeword_with_errors


def add_packet_errors(codeword: np.ndarray, packet_length: int) -> np.ndarray:
    """
    Добавляет пакетные ошибки в закодированное сообщение.

    :param codeword: Закодированное сообщение.
    :param packet_length: Длина пакета ошибок.
    :return: Сообщение с добавленными пакетными ошибками.
    """
    packet_error = np.zeros(len(codeword), dtype=int)
    start_index = random.randint(0, len(codeword) - packet_length)
    
    for i in range(packet_length):
        if random.random() > 0.5:  # Пакетная ошибка может быть частичной
            packet_error[start_index + i] = 1
    
    print(f"Пакетная ошибка: {packet_error}")
    return (codeword + packet_error) % 2


def decode(received: np.ndarray, generator: np.ndarray, error_correction: int,
           is_packet_error: bool = False) -> np.ndarray | None:
    """
    Декодирует принятое сообщение с использованием порождающего многочлена и корректирует ошибки.

    :param received: Принятое сообщение (с ошибками).
    :param generator: Порождающий многочлен (вектор коэффициентов).
    :param error_correction: Количество исправляемых ошибок.
    :param is_packet_error: Флаг, указывающий, является ли ошибка пакетной.
    :return: Декодированное сообщение или None, если не удалось исправить ошибку.
    """
    syndrome = polynomial_division_remainder(received, generator)  # Синдром
    if not np.any(syndrome):  # Если синдром равен 0, ошибок нет
        return received
    
    n = len(received)
    for i in range(n):
        error_pattern = np.zeros(n, dtype=int)
        error_pattern[i] = 1  # Проверяем одиночную ошибку
        shifted_syndrome = polynomial_multiply(syndrome, error_pattern) % 2
        
        # Приводим размеры к длине received
        if len(shifted_syndrome) < n:
            shifted_syndrome = np.pad(shifted_syndrome, (n - len(shifted_syndrome), 0))
        elif len(shifted_syndrome) > n:
            shifted_syndrome = shifted_syndrome[-n:]
        
        remainder = polynomial_division_remainder(shifted_syndrome, generator)
        
        if is_packet_error:
            if len(remainder) <= error_correction and np.any(remainder):
                corrected = (received + shifted_syndrome) % 2
                return corrected
        else:
            if np.sum(remainder) <= error_correction:
                corrected = (received + shifted_syndrome) % 2
                return corrected
    
    print("Ошибка не может быть исправлена.")
    return None


def run_experiment(generator: np.ndarray, message_length: int, error_correction: int,
                   is_packet_error: bool = False, max_errors: int = 3) -> None:
    """
    Запускает эксперимент по кодированию, добавлению ошибок и декодированию.

    :param generator: Порождающий многочлен (вектор коэффициентов).
    :param message_length: Длина исходного сообщения.
    :param error_correction: Максимальное количество ошибок для исправления.
    :param is_packet_error: Флаг, указывающий, используется ли пакетная ошибка.
    :param max_errors: Максимальное количество ошибок для добавления.
    """
    message = generate_random_message(message_length)
    print(f"Исходное сообщение: {message}")
    codeword = encode(message, generator)
    print(f"Закодированное сообщение: {codeword}")
    
    for errors in range(1, max_errors + 1):
        print(f"\n--- Ошибок: {errors} ---")
        if is_packet_error:
            received = add_packet_errors(codeword, errors)
        else:
            received = add_errors(codeword, errors)
        
        print(f"Принятое сообщение: {received}")
        decoded = decode(received, generator, error_correction, is_packet_error)
        print(f"Декодированное сообщение: {decoded}")


# Основной блок
if __name__ == "__main__":
    print("6.1: Циклический код g(x) = 1 + x^2 + x^3, исправляющий однократные ошибки")
    generator_6_1 = np.array([1, 0, 1, 1])  # 1 + x^2 + x^3
    run_experiment(generator_6_1, message_length=4, error_correction=1, max_errors=3)
    
    print("\n6.2: Циклический код (15,9) g(x) = 1 + x^3 + x^4 + x^5 + x^6, исправляющий пакеты ошибок кратности 3")
    generator_6_2 = np.array([1, 0, 0, 1, 1, 1, 1])  # 1 + x^3 + x^4 + x^5 + x^6
    run_experiment(generator_6_2, message_length=9, error_correction=3, is_packet_error=True, max_errors=4)
