"""
Версия для работы с реальными квантовыми компьютерами IBM
"""

import numpy as np
import math
import time
import logging
from typing import List
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Конфигурация для реального железа"""
    backend_name: str = "ibmq_qasm_simulator"
    shots: int = 1024
    optimization_level: int = 3
    use_error_mitigation: bool = False
    max_retries: int = 3
    save_jobs: bool = True
    jobs_dir: str = "quantum_jobs"
    test_mode: bool = False
    instance: str = "ibm-q/open/main"  # Добавляем instance


# Проверяем наличие Qiskit 1.0+
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit.library import DiagonalGate
    from qiskit.transpiler import CouplingMap

    # Для симулятора
    from qiskit_aer import Aer

    # Для реального железа (Qiskit 1.0+)
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

        QISKIT_IBM_AVAILABLE = True
    except ImportError:
        QISKIT_IBM_AVAILABLE = False
        logger.warning("qiskit_ibm_runtime не установлен. Реальное железо недоступно.")

    QISKIT_AVAILABLE = True

except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit не установлен. Установите: pip install qiskit qiskit-aer qiskit-ibm-runtime")


class QiskitHardwareOracle:
    """Реализация для работы с реальными квантовыми компьютерами IBM (Qiskit 1.0+)"""

    def __init__(self, **kwargs):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit не установлен. Установите: pip install qiskit qiskit-aer")

        self.shots = kwargs.get("shots", 1024)
        self.optimization_level = kwargs.get("optimization_level", 1)
        self.backend_name = kwargs.get("backend", "aer_simulator")
        self.hardware_config = kwargs.get("hardware_config", HardwareConfig())
        self.test_mode = kwargs.get("test_mode", False)
        self.quantum_time = 0.0
        self.calls = 0
        self.service = None
        self.backend = None
        self.test_results = []
        self.job_history = []
        self.monitoring = False

        # Создаем директорию для сохранения результатов
        self.jobs_dir = Path(self.hardware_config.jobs_dir)
        self.jobs_dir.mkdir(exist_ok=True)

        # Инициализация бэкенда
        self._init_backend()

    def _init_backend(self):
        """Инициализация бэкенда для Qiskit 1.0+"""
        try:
            # Определяем тип бэкенда
            if self.backend_name in ["aer_simulator", "simulator"] or "simulator" in self.backend_name.lower():
                # Локальный симулятор
                self.backend = Aer.get_backend('aer_simulator')
                logger.info("Используется локальный симулятор Aer")

            elif "ibm" in self.backend_name.lower():
                # IBM Quantum бэкенд
                if not QISKIT_IBM_AVAILABLE:
                    logger.warning("qiskit_ibm_runtime не установлен. Используется локальный симулятор.")
                    self.backend = Aer.get_backend('aer_simulator')
                    return

                try:
                    # Инициализация IBM Quantum Runtime
                    self.service = QiskitRuntimeService()

                    # Получаем бэкенд
                    try:
                        self.backend = self.service.get_backend(self.backend_name)
                    except Exception as e:
                        logger.warning(f"Не удалось получить бэкенд {self.backend_name}: {e}")
                        # Пробуем получить доступные бэкенды
                        backends = self.service.backends()
                        if backends:
                            # Берем первое доступное реальное устройство
                            for backend in backends:
                                if not backend.configuration().simulator:
                                    self.backend = backend
                                    logger.info(f"Используем доступное устройство: {self.backend.name}")
                                    break

                        if not self.backend:
                            # Используем симулятор
                            self.backend = self.service.get_backend('ibmq_qasm_simulator')
                            logger.info("Используем симулятор IBM")

                    logger.info(f"Используется IBM Quantum бэкенд: {self.backend.name}")
                    self._log_backend_info()

                except Exception as e:
                    logger.warning(f"Не удалось подключиться к IBM Quantum: {e}")
                    logger.info("Используется локальный симулятор")
                    self.backend = Aer.get_backend('aer_simulator')

            else:
                # По умолчанию локальный симулятор
                self.backend = Aer.get_backend('aer_simulator')
                logger.info("Используется локальный симулятор Aer по умолчанию")

        except Exception as e:
            logger.error(f"Ошибка инициализации бэкенда: {e}")
            # Fallback на локальный симулятор
            try:
                self.backend = Aer.get_backend('aer_simulator')
            except:
                raise

    def _log_backend_info(self):
        """Логирование информации о бэкенде"""
        if not self.backend:
            return

        try:
            if hasattr(self.backend, 'configuration'):
                config = self.backend.configuration()
                status = self.backend.status()

                logger.info("Информация о бэкенде:")
                logger.info(f"  • Имя: {self.backend.name}")
                logger.info(f"  • Кубитов: {config.n_qubits}")
                logger.info(f"  • Симулятор: {config.simulator}")
                if hasattr(status, 'operational'):
                    logger.info(f"  • Операционный: {status.operational}")
                if hasattr(status, 'pending_jobs'):
                    logger.info(f"  • Очередь: {status.pending_jobs} заданий")

                if hasattr(config, 'basis_gates'):
                    logger.info(f"  • Базовые гейты: {config.basis_gates[:5]}...")

        except Exception as e:
            logger.warning(f"Не удалось получить информацию о бэкенде: {e}")

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
        """Алгоритм Гровера на Qiskit 1.0+"""
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
            iterations = max(1, min(iterations, 3))  # Меньше итераций для железа

        n_qubits = max(1, math.ceil(math.log2(n)))

        # Создаем схему
        qc = self._create_grover_circuit(f_values, threshold, iterations)

        try:
            # Проверяем, реальное ли это железо
            is_real_hardware = (
                    hasattr(self.backend, 'configuration') and
                    hasattr(self.backend.configuration(), 'simulator') and
                    not self.backend.configuration().simulator
            )

            # Для реального железа упрощаем схему
            if is_real_hardware and n_qubits > 3:
                logger.warning(f"Слишком много кубитов ({n_qubits}) для реального железа.")
                # Ограничиваем максимальное количество кубитов
                n_qubits = min(n_qubits, 3)
                # Берем только первые 2^n_qubits значений
                f_values = f_values[:min(len(f_values), 2 ** n_qubits)]
                n = len(f_values)
                # Пересоздаем схему с меньшим числом кубитов
                qc = self._create_grover_circuit(f_values, threshold, iterations)

            # Транспиляция с учетом железа
            if hasattr(self.backend, 'configuration'):
                try:
                    config = self.backend.configuration()
                    if hasattr(config, 'coupling_map') and config.coupling_map:
                        coupling_map = CouplingMap(config.coupling_map)
                        transpiled_qc = transpile(
                            qc,
                            self.backend,
                            optimization_level=self.optimization_level,
                            coupling_map=coupling_map
                        )
                    else:
                        transpiled_qc = transpile(
                            qc,
                            self.backend,
                            optimization_level=self.optimization_level
                        )
                except:
                    transpiled_qc = transpile(qc, self.backend)
            else:
                transpiled_qc = transpile(qc, self.backend)

            # Запуск
            if is_real_hardware and QISKIT_IBM_AVAILABLE:
                # Для реального IBM Quantum используем Runtime
                logger.info(f"Отправка задания на реальное устройство {self.backend.name}...")

                try:
                    with Session(backend=self.backend) as session:
                        sampler = Sampler(session=session)
                        job = sampler.run(transpiled_qc, shots=self.shots)

                        logger.info(f"Задание отправлено. ID: {job.job_id()}")
                        logger.info("Ожидание результатов... (может занять несколько минут)")

                        # Ждем результат с таймаутом
                        result = job.result(timeout=600)  # 10 минут таймаут
                        counts = result.quasi_dists[0]

                        logger.info(f"Результаты получены с устройства {self.backend.name}")

                except Exception as e:
                    logger.error(f"Ошибка выполнения на реальном железе: {e}")
                    raise

            else:
                # Для локального симулятора
                job = self.backend.run(transpiled_qc, shots=self.shots)
                result = job.result()
                counts = result.get_counts()

            # Обработка результатов
            probabilities = self._process_counts(counts, n)

            # Определяем результат
            if probabilities:
                max_prob_idx = np.argmax(probabilities)
                found = f_values[max_prob_idx] < threshold
            else:
                found = False

            logger.info(f"Квантовое вычисление завершено на {self.backend.name}")

        except Exception as e:
            logger.error(f"Ошибка выполнения на железе: {e}")
            logger.info("Используется локальный симулятор")

            # Fallback на локальный симулятор
            try:
                simulator = Aer.get_backend('aer_simulator')
                transpiled_qc = transpile(qc, simulator)
                job = simulator.run(transpiled_qc, shots=self.shots)
                result = job.result()
                counts = result.get_counts()

                probabilities = self._process_counts(counts, n)

                if probabilities:
                    max_prob_idx = np.argmax(probabilities)
                    found = f_values[max_prob_idx] < threshold
                else:
                    found = False

            except Exception as fallback_error:
                logger.error(f"Fallback тоже не сработал: {fallback_error}")
                found = False
                probabilities = []

        elapsed = time.time() - start_time
        self.quantum_time += elapsed

        return found, probabilities

    def _create_grover_circuit(self, f_values: List[float], threshold: float, iterations: int):
        """Создание схемы Гровера"""
        n = len(f_values)
        n_qubits = max(1, math.ceil(math.log2(n)))

        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)

        # Инициализация
        qc.h(qr)

        # Оракул и диффузия
        oracle = self.create_oracle_circuit(f_values, threshold)
        diffusion = self.create_diffusion_circuit(n_qubits)

        # Итерации
        for _ in range(iterations):
            qc.append(oracle, qr)
            qc.append(diffusion, qr)

        # Измерение
        qc.measure(qr, cr)

        return qc

    def _process_counts(self, counts, n):
        """Обработка результатов измерений"""
        if isinstance(counts, dict):
            # Старый формат (get_counts)
            total_shots = sum(counts.values())
            probabilities_dict = {state: count / total_shots for state, count in counts.items()}
        else:
            # Новый формат (quasi_dists)
            probabilities_dict = {int(state): prob for state, prob in counts.items()}
            total_shots = sum(probabilities_dict.values()) if probabilities_dict else 1

        # Преобразуем в список
        probabilities = [0.0] * n
        for state, prob in probabilities_dict.items():
            idx = int(state)
            if idx < n:
                probabilities[idx] = prob

        # Нормализуем
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]

        return probabilities