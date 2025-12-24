"""
Матричная реализация алгоритма Гровера через NumPy
"""

import numpy as np
import math
from typing import List, Tuple
import time


class MatrixOracle:
    """Эмулятор квантового оракула через матричные преобразования"""

    def __init__(self, **kwargs):
        self.shots = kwargs.get("shots", 1024)
        self.quantum_time = 0.0
        self.calls = 0

    @staticmethod
    def create_oracle_matrix(vertex_f_values: List[float], threshold: float, n_qubits: int) -> np.ndarray:
        """
        Создает матрицу оракула для помечания вершин с f < threshold
        """
        size = 2 ** n_qubits
        oracle_matrix = np.eye(size, dtype=complex)

        for i, f_value in enumerate(vertex_f_values):
            if i < size and f_value < threshold:
                oracle_matrix[i, i] = -1

        return oracle_matrix

    @staticmethod
    def create_diffusion_matrix(n_qubits: int) -> np.ndarray:
        """Создает матрицу диффузионного оператора Гровера"""
        size = 2 ** n_qubits
        H = MatrixOracle.hadamard_matrix(n_qubits)
        reflection = np.eye(size, dtype=complex)
        reflection[0, 0] = -1
        diffusion = -H @ reflection @ H
        return diffusion

    @staticmethod
    def hadamard_matrix(n_qubits: int) -> np.ndarray:
        """Матрица преобразования Адамара на n кубитах"""
        H_single = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_total = H_single
        for _ in range(n_qubits - 1):
            H_total = np.kron(H_total, H_single)
        return H_total

    def grover_search(self, vertex_f_values: List[float], threshold: float, iterations: int = None) -> Tuple[
        bool, List[float]]:
        """
        Алгоритм Гровера через матричные операции для поиска вершин с f < threshold
        """
        start_time = time.time()
        self.calls += 1

        try:
            if not vertex_f_values:
                probabilities = []
                found = False
                return found, probabilities

            n_vertices = len(vertex_f_values)
            n_qubits = math.ceil(math.log2(n_vertices))
            actual_size = 2 ** n_qubits

            # Создаем все необходимые матрицы
            oracle = self.create_oracle_matrix(vertex_f_values, threshold, n_qubits)
            diffusion = self.create_diffusion_matrix(n_qubits)
            H = self.hadamard_matrix(n_qubits)

            # Начальное состояние: равномерная суперпозиция
            initial_state = np.zeros(actual_size, dtype=complex)
            initial_state[0] = 1.0  # |0...0⟩ состояние

            # Применяем преобразование адмара для создания равномерной суперпозиции
            state = H @ initial_state

            # Оптимальное число итераций Гровера
            if iterations is None:
                marked_count = sum(1 for f in vertex_f_values if f < threshold)
                if marked_count == 0:
                    # Если нет хороших вершин, возвращаем равномерное распределение
                    probabilities = [1.0 / n_vertices] * n_vertices if n_vertices > 0 else []
                    found = False
                    return found, probabilities
                if marked_count == n_vertices:
                    # Если все вершины хорошие, возвращаем равномерное распределение
                    probabilities = [1.0 / n_vertices] * n_vertices if n_vertices > 0 else []
                    found = True
                    return found, probabilities

                theta = math.asin(math.sqrt(marked_count / n_vertices))
                iterations = int(math.pi / (4 * theta))
                iterations = max(1, min(iterations, 5))

                print(f"Отладка: n={n_vertices}, хороших={marked_count}, theta={theta:.4f}, итераций={iterations}")

            # Применяем итерации Гровера
            for iteration in range(iterations):
                state = oracle @ state  # Применяем оракул (помечаем хорошие состояния)
                state = diffusion @ state  # Применяем диффузию (усиливаем помеченные)

                # Отладочная информация для каждой итерации
                if False:  # Включить для отладки
                    probs_temp = np.abs(state) ** 2
                    probs_temp = probs_temp[:n_vertices]
                    if np.sum(probs_temp) > 0:
                        probs_temp = probs_temp / np.sum(probs_temp)
                    max_prob = np.max(probs_temp)
                    max_idx = np.argmax(probs_temp)
                    print(f"  Итерация {iteration + 1}: макс p={max_prob:.4f} на вершине {max_idx}")

            # Вычисляем вероятности
            probabilities = np.abs(state) ** 2
            probabilities = probabilities[:n_vertices]

            # Нормализуем вероятности
            prob_sum = np.sum(probabilities)
            if prob_sum > 0:
                probabilities = probabilities / prob_sum
            else:
                probabilities = np.zeros_like(probabilities)

            # Определяем результат
            if len(probabilities) > 0:
                max_prob_idx = np.argmax(probabilities)
                found = vertex_f_values[max_prob_idx] < threshold if max_prob_idx < n_vertices else False
            else:
                found = False

            return found, probabilities.tolist()

        finally:
            elapsed = time.time() - start_time
            self.quantum_time += elapsed