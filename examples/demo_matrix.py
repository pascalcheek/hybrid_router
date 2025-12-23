"""
Демонстрация работы гибридного алгоритма с матричной эмуляцией
"""

import time
from src.architecture import (
    HybridRoutingEngine,
    RouterConfig,
    QuantumBackend,
    SearchStrategy
)
from src.classical.graph_utils import (
    create_dead_end_graph,
    create_complex_test_graph,
    create_maze_graph
)


def demo_matrix_emulation():
    """Демонстрация матричной эмуляции"""
    print("=" * 80)
    print("ДЕМОНСТРАЦИЯ ГИБРИДНОГО АЛГОРИТМА С МАТРИЧНОЙ ЭМУЛЯЦИЕЙ")
    print("=" * 80)

    # 1. Граф с тупиковой ветвью
    print("\n1. Граф с тупиковой ветвью:")
    print("-" * 40)

    graph1 = create_dead_end_graph()

    # Конфигурация с матричной эмуляцией
    config = RouterConfig(
        backend=QuantumBackend.MATRIX_EMULATION,
        strategy=SearchStrategy.HYBRID_CONSERVATIVE,
        quantum_threshold=0.9,
        confidence_threshold=0.15
    )

    engine = HybridRoutingEngine(config)
    engine.set_graph(graph1)

    start_time = time.time()
    path, cost, metrics = engine.find_path(0, 3)
    exec_time = time.time() - start_time

    print(f"   Путь: {path}")
    print(f"   Стоимость: {cost:.3f}")
    print(f"   Время: {exec_time:.4f} сек")
    print(f"   Посещено вершин: {metrics.total_nodes_visited}")
    print(f"   Отсечено вершин: {metrics.pruned_nodes}")
    print(f"   Квантовых запросов: {metrics.quantum_calls}")
    print(f"   Время на квант: {metrics.quantum_time:.4f} сек")

    # 2. Сложный граф
    print("\n2. Сложный граф с несколькими тупиками:")
    print("-" * 40)

    graph2 = create_complex_test_graph()
    engine.set_graph(graph2)

    start_time = time.time()
    path, cost, metrics = engine.find_path(0, 14)
    exec_time = time.time() - start_time

    print(f"   Путь: {path}")
    print(f"   Стоимость: {cost:.3f}")
    print(f"   Время: {exec_time:.4f} сек")
    print(f"   Посещено вершин: {metrics.total_nodes_visited}")
    print(f"   Отсечено вершин: {metrics.pruned_nodes}")
    print(f"   Квантовых запросов: {metrics.quantum_calls}")

    # 3. Граф-лабиринт
    print("\n3. Граф-лабиринт:")
    print("-" * 40)

    graph3 = create_maze_graph()
    engine.set_graph(graph3)

    start_time = time.time()
    path, cost, metrics = engine.find_path(0, 15)
    exec_time = time.time() - start_time

    print(f"   Путь: {path}")
    print(f"   Стоимость: {cost:.3f}")
    print(f"   Время: {exec_time:.4f} сек")
    print(f"   Посещено вершин: {metrics.total_nodes_visited}")
    print(f"   Отсечено вершин: {metrics.pruned_nodes}")

    # 4. Сравнение стратегий
    print("\n4. Сравнение различных стратегий:")
    print("-" * 40)

    strategies = [
        ("Чисто классическая", SearchStrategy.PURE_CLASSICAL),
        ("Консервативная", SearchStrategy.HYBRID_CONSERVATIVE),
        ("Агрессивная", SearchStrategy.HYBRID_AGGRESSIVE),
        ("Адаптивная", SearchStrategy.ADAPTIVE)
    ]

    results = []
    for strategy_name, strategy in strategies:
        config = RouterConfig(
            backend=QuantumBackend.MATRIX_EMULATION,
            strategy=strategy,
            quantum_threshold=0.9
        )

        engine = HybridRoutingEngine(config)
        engine.set_graph(graph2)

        start_time = time.time()
        path, cost, metrics = engine.find_path(0, 14)
        exec_time = time.time() - start_time

        results.append({
            "strategy": strategy_name,
            "path_length": len(path),
            "cost": cost,
            "time": exec_time,
            "visited": metrics.total_nodes_visited,
            "pruned": metrics.pruned_nodes,
            "quantum_calls": metrics.quantum_calls
        })

        print(f"   {strategy_name}:")
        print(f"     Посещено вершин: {metrics.total_nodes_visited}")
        print(f"     Отсечено вершин: {metrics.pruned_nodes}")
        print(f"     Время: {exec_time:.4f} сек")

    # 5. Анализ эффективности
    print("\n5. Анализ эффективности отсечения:")
    print("-" * 40)

    best_result = min(results, key=lambda x: x["visited"])
    worst_result = max(results, key=lambda x: x["visited"])

    print(f"   Лучшая стратегия: {best_result['strategy']}")
    print(f"     Посещено вершин: {best_result['visited']}")
    print(f"     Отсечено вершин: {best_result['pruned']}")
    print(f"     Эффективность отсечения: {best_result['pruned'] / max(1, best_result['visited']):.1%}")

    print(f"   Худшая стратегия: {worst_result['strategy']}")
    print(f"     Посещено вершин: {worst_result['visited']}")

    # 6. Демонстрация работы Гровера
    print("\n6. Демонстрация работы алгоритма Гровера:")
    print("-" * 40)

    from src.quantum.matrix_oracle import MatrixOracle
    oracle = MatrixOracle()

    # Пример 1: Одна хорошая вершина
    f_values = [10.0, 9.5, 4.2, 10.5, 9.8]
    threshold = 5.0

    print(f"   Пример 1: Одна хорошая вершина")
    print(f"   f-значения: {f_values}")
    print(f"   Порог: {threshold}")

    found, probabilities = oracle.grover_search(f_values, threshold)

    print(f"   Найдена хорошая вершина: {'Да' if found else 'Нет'}")
    print(f"   Вероятности: {[f'{p:.3f}' for p in probabilities]}")

    # Пример 2: Несколько хороших вершин
    f_values = [3.0, 6.0, 4.5, 7.0, 3.8]
    threshold = 5.0

    print(f"\n   Пример 2: Несколько хороших вершин")
    print(f"   f-значения: {f_values}")
    print(f"   Порог: {threshold}")

    found, probabilities = oracle.grover_search(f_values, threshold)

    print(f"   Найдены хорошие вершины: {'Да' if found else 'Нет'}")
    print(f"   Вероятности: {[f'{p:.3f}' for p in probabilities]}")


if __name__ == "__main__":
    demo_matrix_emulation()