"""
Реализация алгоритма Гровера на Qiskit (исправлено для Qiskit 2.2+)
"""

import numpy as np
import math
import time
from typing import List, Tuple

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import Aer
    from qiskit.circuit.library import DiagonalGate
    import qiskit

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class QiskitOracle:
    """Реализация алгоритма Гровера на Qiskit"""

    def __init__(self, **kwargs):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not available. Install with: pip install qiskit-aer")

        self.shots = kwargs.get("shots", 1024)
        self.optimization_level = kwargs.get("optimization_level", 1)
        self.backend = Aer.get_backend('aer_simulator')
        self.quantum_time = 0.0
        self.calls = 0

    def create_oracle_circuit(self, f_values: List[float], threshold: float):
        """Создает квантовую схему оракула"""
        n = len(f_values)
        n_qubits = max(1, math.ceil(math.log2(n)))

        # Создаем диагональную матрицу для оракула
        oracle_diag = []
        for i in range(2 ** n_qubits):
            if i < n and f_values[i] < threshold:
                oracle_diag.append(-1)
            else:
                oracle_diag.append(1)

        # Создаем схему
        qr = QuantumRegister(n_qubits, 'q')
        oracle_circuit = QuantumCircuit(qr, name="Oracle")
        oracle_gate = DiagonalGate(oracle_diag)
        oracle_circuit.append(oracle_gate, qr[:])

        return oracle_circuit

    def create_diffusion_circuit(self, n_qubits: int):
        """Создает диффузионный оператор"""
        qr = QuantumRegister(n_qubits, 'q')
        diffusion_circuit = QuantumCircuit(qr, name="Diffusion")

        diffusion_circuit.h(range(n_qubits))
        diffusion_circuit.x(range(n_qubits))

        # Многокубитная операция Z
        if n_qubits == 1:
            diffusion_circuit.z(0)
        else:
            diffusion_circuit.h(n_qubits - 1)
            diffusion_circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            diffusion_circuit.h(n_qubits - 1)

        diffusion_circuit.x(range(n_qubits))
        diffusion_circuit.h(range(n_qubits))

        return diffusion_circuit

    def grover_search(self, f_values: List[float], threshold: float, iterations: int = None):
        """Алгоритм Гровера через Qiskit"""
        start_time = time.time()
        self.calls += 1

        if not f_values:
            elapsed = time.time() - start_time
            self.quantum_time += elapsed
            return False, []

        n = len(f_values)

        # Оптимальное число итераций
        marked_count = sum(1 for f in f_values if f < threshold)
        if marked_count == 0:
            elapsed = time.time() - start_time
            self.quantum_time += elapsed
            return False, [1.0 / n] * n if n > 0 else []
        if marked_count == n:
            elapsed = time.time() - start_time
            self.quantum_time += elapsed
            return True, [1.0 / n] * n

        # Вычисляем оптимальное число итераций
        if iterations is None:
            theta = math.asin(math.sqrt(marked_count / n))
            iterations = int(math.pi / (4 * theta))
            iterations = max(1, min(iterations, 5))

        n_qubits = max(1, math.ceil(math.log2(n)))

        # Создаем полную схему
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)

        # Инициализация в равномерную суперпозицию
        qc.h(qr)

        # Создаем оракул и диффузию
        oracle = self.create_oracle_circuit(f_values, threshold)
        diffusion = self.create_diffusion_circuit(n_qubits)

        # Применяем итерации
        for _ in range(iterations):
            qc.append(oracle, qr)
            qc.append(diffusion, qr)

        # Измерение
        qc.measure(qr, cr)

        # Выполнение на симуляторе
        transpiled_qc = transpile(qc, self.backend, optimization_level=self.optimization_level)
        job = self.backend.run(transpiled_qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()

        # Анализируем результаты
        total_shots = sum(counts.values())
        probabilities_dict = {state: count / total_shots for state, count in counts.items()}

        # Преобразуем в список вероятностей
        probabilities = [0.0] * n
        for state, prob in probabilities_dict.items():
            idx = int(state, 2)
            if idx < n:
                probabilities[idx] = prob

        # Нормализуем
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]

        # Определяем результат
        if probabilities:
            max_prob_idx = np.argmax(probabilities)
            found = f_values[max_prob_idx] < threshold
        else:
            found = False

        elapsed = time.time() - start_time
        self.quantum_time += elapsed

        return found, probabilities