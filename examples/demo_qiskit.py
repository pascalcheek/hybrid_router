"""
Демонстрация работы гибридного алгоритма с Qiskit (исправленная версия)
"""

import time
import sys
import os

# Добавляем src в путь Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    # Пробуем импортировать Qiskit отдельно
    import qiskit
    from qiskit_aer import Aer
    QISKIT_IMPORT_OK = True
except ImportError:
    QISKIT_IMPORT_OK = False
    print("Qiskit не установлен. Установите: pip install qiskit-aer")

# Пробуем импортировать компоненты проекта
try:
    from src.architecture import (
        HybridRoutingEngine,
        RouterConfig,
        QuantumBackend,
        SearchStrategy
    )
    from src.classical.graph_utils import (
        create_dead_end_graph,
        create_complex_test_graph
    )
    PROJECT_IMPORT_OK = True
except ImportError as e:
    PROJECT_IMPORT_OK = False
    print(f"Ошибка импорта компонентов проекта: {e}")


def demo_qiskit_implementation():
    """Демонстрация реализации на Qiskit"""
    if not QISKIT_IMPORT_OK:
        print("Пропущено: Qiskit не установлен")
        return

    if not PROJECT_IMPORT_OK:
        print("Пропущено: Ошибка импорта компонентов проекта")
        return

    print("=" * 80)
    print("ДЕМОНСТРАЦИЯ ГИБРИДНОГО АЛГОРИТМА С QISKIT")
    print("=" * 80)

    # 1. Базовая демонстрация
    print("\n1. Базовая демонстрация с Qiskit симулятором:")
    print("-" * 40)

    graph = create_dead_end_graph()

    config = RouterConfig(
        backend=QuantumBackend.QISKIT_SIMULATOR,
        strategy=SearchStrategy.HYBRID_CONSERVATIVE,
        quantum_threshold=0.9,
        qiskit_shots=512,
        qiskit_optimization_level=1
    )

    engine = HybridRoutingEngine(config)
    engine.set_graph(graph)

    start_time = time.time()
    path, cost, metrics = engine.find_path(0, 3)
    exec_time = time.time() - start_time

    print(f"   Путь: {path}")
    print(f"   Стоимость: {cost:.3f}")
    print(f"   Время: {exec_time:.4f} сек")
    print(f"   Посещено вершин: {metrics.total_nodes_visited}")
    print(f"   Квантовых запросов: {metrics.quantum_calls}")
    print(f"   Время на квант: {metrics.quantum_time:.4f} сек")

    # 2. Сравнение с матричной эмуляцией
    print("\n2. Сравнение Qiskit и матричной эмуляцией:")
    print("-" * 40)

    backends = [
        ("Матричная эмуляция", QuantumBackend.MATRIX_EMULATION),
        ("Qiskit симулятор", QuantumBackend.QISKIT_SIMULATOR)
    ]

    graph = create_complex_test_graph()
    results = []

    for backend_name, backend in backends:
        config = RouterConfig(
            backend=backend,
            strategy=SearchStrategy.HYBRID_CONSERVATIVE,
            quantum_threshold=0.9
        )

        if backend == QuantumBackend.QISKIT_SIMULATOR:
            config.qiskit_shots = 256  # Меньше shots для скорости

        engine = HybridRoutingEngine(config)
        engine.set_graph(graph)

        start_time = time.time()
        path, cost, metrics = engine.find_path(0, 14)
        exec_time = time.time() - start_time

        results.append({
            "backend": backend_name,
            "path_length": len(path),
            "cost": cost,
            "time": exec_time,
            "visited": metrics.total_nodes_visited,
            "quantum_calls": metrics.quantum_calls,
            "quantum_time": metrics.quantum_time
        })

        print(f"   {backend_name}:")
        print(f"     Время: {exec_time:.4f} сек")
        print(f"     Квантовых запросов: {metrics.quantum_calls}")
        print(f"     Время на квант: {metrics.quantum_time:.4f} сек")

    # 3. Анализ производительности
    print("\n3. Анализ производительности:")
    print("-" * 40)

    matrix_result = next(r for r in results if r["backend"] == "Матричная эмуляция")
    qiskit_result = next(r for r in results if r["backend"] == "Qiskit симулятор")

    time_ratio = qiskit_result["time"] / max(0.001, matrix_result["time"])
    quantum_time_ratio = qiskit_result["quantum_time"] / max(0.001, matrix_result["quantum_time"])

    print(f"   Отношение времени выполнения (Qiskit/Матрица): {time_ratio:.2f}x")
    print(f"   Отношение времени на квант (Qiskit/Матрица): {quantum_time_ratio:.2f}x")

    # 4. Демонстрация работы Гровера на Qiskit
    print("\n4. Демонстрация алгоритма Гровера на Qiskit:")
    print("-" * 40)

    try:
        from src.quantum.qiskit_oracle import QiskitOracle
        oracle = QiskitOracle(shots=256)

        # Пример поиска
        f_values = [10.0, 9.5, 4.2, 10.5, 9.8]
        threshold = 5.0

        print(f"   f-значения: {f_values}")
        print(f"   Порог: {threshold}")

        start_time = time.time()
        found, probabilities = oracle.grover_search(f_values, threshold)
        exec_time = time.time() - start_time

        print(f"   Найдена хорошая вершина: {'Да' if found else 'Нет'}")
        print(f"   Вероятности: {[f'{p:.3f}' for p in probabilities]}")
        print(f"   Время выполнения: {exec_time:.4f} сек")
        print(f"   Количество shots: {oracle.shots}")

    except Exception as e:
        print(f"   Ошибка: {e}")
        import traceback
        traceback.print_exc()

    # 5. Информация о доступных бэкендах
    print("\n5. Информация о доступных квантовых бэкендах:")
    print("-" * 40)

    try:
        from qiskit_aer import Aer

        print("   Доступные симуляторы Aer:")

        # Универсальный способ
        backends = Aer.backends()

        if isinstance(backends, dict):
            # Новые версии Qiskit
            for backend_name in list(backends.keys())[:5]:
                print(f"  - {backend_name}")
        elif isinstance(backends, list):
            # Старые версии Qiskit
            for backend in backends[:5]:
                try:
                    if hasattr(backend, 'name'):
                        if callable(backend.name):
                            print(f"  - {backend.name()}")
                        else:
                            print(f"  - {backend.name}")
                    else:
                        print(f"  - {str(backend)}")
                except:
                    print(f"  - Неизвестный бэкенд")
        else:
            print(f"  Неизвестный формат backends: {type(backends)}")

    except ImportError:
        print("   Qiskit Aer не установлен")



def check_qiskit_installation():
    """Проверка установки Qiskit"""
    print("Проверка установки Qiskit...")

    try:
        import qiskit
        from qiskit_aer import Aer

        print(f"Qiskit версия: {qiskit.__version__}")
        print("Qiskit Aer доступен")

        # Универсальная проверка backends
        try:
            backends = Aer.backends()

            if isinstance(backends, dict):
                print(f"Доступно симуляторов: {len(backends)}")
                print("  Первые 3 симулятора:")
                for backend_name in list(backends.keys())[:3]:
                    print(f"  - {backend_name}")

            elif isinstance(backends, list):
                print(f"Доступно симуляторов: {len(backends)}")
                print("  Первые 3 симулятора:")
                for backend in backends[:3]:
                    try:
                        if hasattr(backend, 'name'):
                            name = backend.name() if callable(backend.name) else str(backend.name)
                        else:
                            name = str(backend)
                        print(f"  - {name}")
                    except:
                        print(f"  - Симулятор")

            else:
                print(f"Backends в формате: {type(backends)}")

        except Exception as e:
            print(f"Не удалось получить список симуляторов: {e}")

        return True

    except ImportError as e:
        print("✗ Qiskit не установлен или установлен неправильно")
        print(f"  Ошибка: {e}")
        print("\nУстановите Qiskit:")
        print("  pip install qiskit-aer")
        return False


if __name__ == "__main__":
    # Проверка установки Qiskit
    qiskit_ok = check_qiskit_installation()

    if qiskit_ok:
        print("\n" + "="*60)
        print("ЗАПУСК ДЕМОНСТРАЦИИ QISKIT")
        print("="*60)
        demo_qiskit_implementation()
    else:
        print("\nДля демонстрации Qiskit необходимо установить библиотеку.")