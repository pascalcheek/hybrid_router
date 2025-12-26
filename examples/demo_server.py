"""
демонстрация гибридного квантово-классического алгоритма маршрутизации
Показывает реальную разницу в производительности
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.architecture import (
    HybridRoutingEngine,
    RouterConfig,
    QuantumBackend,
    SearchStrategy,
    Graph,
    Vertex,
    Edge
)

def create_challenging_graph() -> Graph:
    """Создает сложный граф с множеством тупиков для демонстрации"""
    vertices = [Vertex(i, i % 5, i // 5) for i in range(20)]

    edges = []

    # Оптимальный путь (стоимость ~8.0)
    optimal_path = [0, 1, 6, 11, 16, 17, 18, 19]
    for i in range(len(optimal_path) - 1):
        edges.append(Edge(optimal_path[i], optimal_path[i + 1], 1.0))

    # Множество тупиковых ветвей (уводят от цели)
    dead_ends = [
        (0, 5, 1.0), (5, 10, 1.0), (10, 15, 1.0),  # Тупик 15
        (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),      # Тупик 4
        (6, 7, 1.0), (7, 8, 1.0), (8, 9, 1.0),      # Тупик 9
        (11, 12, 1.0), (12, 13, 1.0), (13, 14, 1.0),# Тупик 14
    ]

    for from_v, to_v, weight in dead_ends:
        edges.append(Edge(from_v, to_v, weight))

    # Дорогие обходные пути (хуже оптимального)
    expensive_paths = [
        (0, 1, 1.0), (1, 7, 2.0), (7, 13, 2.0), (13, 19, 3.0),  # Стоимость 8.0
        (0, 5, 1.0), (5, 11, 2.0), (11, 17, 2.0), (17, 19, 3.0),# Стоимость 8.0
    ]

    for from_v, to_v, weight in expensive_paths:
        if (from_v, to_v, weight) not in edges:
            edges.append(Edge(from_v, to_v, weight))

    # Случайные связи для сложности
    random_connections = [
        (2, 7, 1.5), (3, 8, 1.5), (4, 9, 1.5),
        (10, 11, 1.2), (12, 13, 1.2), (14, 15, 1.2),
        (16, 17, 1.0), (17, 18, 1.0), (18, 19, 1.0),
    ]

    for from_v, to_v, weight in random_connections:
        edges.append(Edge(from_v, to_v, weight))

    return Graph(vertices, edges)

def benchmark_algorithms() -> Dict:
    """Тестирование производительности разных алгоритмов"""
    print("\n" + "="*80)
    print("БЕНЧМАРК-ТЕСТИРОВАНИЕ АЛГОРИТМОВ")
    print("="*80)

    graph = create_challenging_graph()

    print(f"\n Тестовый граф:")
    print(f"   • Вершин: {len(graph.vertices)}")
    print(f"   • Рёбер: {len(graph.edges)}")
    print(f"   • Оптимальный путь: 0 → 19 (стоимость ~8.0)")

    algorithms = [
        {
            "name": "Классический A*",
            "config": RouterConfig(strategy=SearchStrategy.PURE_CLASSICAL),
            "color": "red"
        },
        {
            "name": "Гибридный (консервативный)",
            "config": RouterConfig(
                backend=QuantumBackend.MATRIX_EMULATION,
                strategy=SearchStrategy.HYBRID_CONSERVATIVE,
                quantum_threshold=0.85,
                confidence_threshold=0.2
            ),
            "color": "blue"
        },
        {
            "name": "Гибридный (агрессивный)",
            "config": RouterConfig(
                backend=QuantumBackend.MATRIX_EMULATION,
                strategy=SearchStrategy.HYBRID_AGGRESSIVE,
                quantum_threshold=0.7,
                confidence_threshold=0.1
            ),
            "color": "green"
        },
        {
            "name": "Гибридный (адаптивный)",
            "config": RouterConfig(
                backend=QuantumBackend.MATRIX_EMULATION,
                strategy=SearchStrategy.ADAPTIVE,
                quantum_threshold=0.8,
                confidence_threshold=0.15
            ),
            "color": "orange"
        }
    ]

    results = []

    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Тестируем: {algo['name']}")
        print(f"{'='*60}")

        engine = HybridRoutingEngine(algo['config'])
        engine.set_graph(graph)

        # Запускаем 5 раз для статистики
        times = []
        visited_nodes = []
        pruned_nodes = []
        quantum_calls = []

        for run in range(5):
            start_time = time.perf_counter()
            path, cost, metrics = engine.find_path(0, 19)
            end_time = time.perf_counter()

            times.append(end_time - start_time)
            visited_nodes.append(metrics.total_nodes_visited)
            pruned_nodes.append(metrics.pruned_nodes)
            quantum_calls.append(metrics.quantum_calls)

            if run == 0:  # Показываем результаты первого запуска
                print(f"   Путь: {' → '.join(map(str, path[:5]))}... → {path[-1] if path else '?'}")
                print(f"   Стоимость: {cost:.2f}")
                print(f"   Посещено вершин: {metrics.total_nodes_visited}")
                print(f"   Отсечено вершин: {metrics.pruned_nodes}")
                print(f"   Квантовых запросов: {metrics.quantum_calls}")
                print(f"   Время: {times[-1]:.4f} сек")

                if metrics.fallback_used:
                    print("    Использован fallback на классический A*")

        # Средние значения
        avg_time = np.mean(times)
        avg_visited = np.mean(visited_nodes)
        avg_pruned = np.mean(pruned_nodes)
        avg_quantum = np.mean(quantum_calls)

        results.append({
            "name": algo['name'],
            "color": algo['color'],
            "avg_time": avg_time,
            "avg_visited": avg_visited,
            "avg_pruned": avg_pruned,
            "avg_quantum": avg_quantum,
            "path": path if 'path' in locals() else [],
            "cost": cost if 'cost' in locals() else float('inf')
        })

    return results

def visualize_results(results: List[Dict]):
    """Визуализация результатов тестирования"""
    print("\n" + "="*80)
    print(" ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("="*80)

    # Текстовая таблица
    print(f"\n{'Алгоритм':<30} {'Вершин':<10} {'Отсечено':<10} {'Время (с)':<12} {'Кв.запросов':<12} {'Ускорение':<12}")
    print("-" * 90)

    classical_result = next(r for r in results if "Классический" in r["name"])

    for result in results:
        speedup = classical_result["avg_time"] / max(0.0001, result["avg_time"])
        reduction = ((classical_result["avg_visited"] - result["avg_visited"]) /
                    classical_result["avg_visited"] * 100)

        print(f"{result['name']:<30} "
              f"{result['avg_visited']:<10.1f} "
              f"{result['avg_pruned']:<10.1f} "
              f"{result['avg_time']:<12.4f} "
              f"{result['avg_quantum']:<12.1f} "
              f"{speedup:<12.2f}x")

        if result != classical_result:
            print(f"  → Сокращение поиска: {reduction:>6.1f}%")

    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Посещенные вершины
    names = [r["name"] for r in results]
    visited = [r["avg_visited"] for r in results]
    colors = [r["color"] for r in results]

    axes[0, 0].bar(names, visited, color=colors)
    axes[0, 0].set_title('Среднее количество посещенных вершин')
    axes[0, 0].set_ylabel('Вершины')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Время выполнения
    times = [r["avg_time"] * 1000 for r in results]  # в миллисекундах

    axes[0, 1].bar(names, times, color=colors)
    axes[0, 1].set_title('Среднее время выполнения')
    axes[0, 1].set_ylabel('Время (мс)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Отсеченные вершины
    pruned = [r["avg_pruned"] for r in results]

    axes[1, 0].bar(names, pruned, color=colors)
    axes[1, 0].set_title('Среднее количество отсеченных вершин')
    axes[1, 0].set_ylabel('Вершины')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Ускорение относительно классического A*
    speedups = [classical_result["avg_time"] / max(0.0001, r["avg_time"]) for r in results]

    axes[1, 1].bar(names, speedups, color=colors)
    axes[1, 1].set_title('Ускорение относительно классического A*')
    axes[1, 1].set_ylabel('Коэффициент ускорения')
    axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Сохраняем график
    plot_filename = "benchmark_results.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\nГрафики сохранены в файл: {plot_filename}")

    # Показываем график
    plt.show()

def demo_quantum_advantage():
    """Демонстрация квантового преимущества"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ КВАНТОВОГО ПРЕИМУЩЕСТВА")
    print("="*80)

    from src.quantum.matrix_oracle import MatrixOracle

    oracle = MatrixOracle()

    print("\n Пример 1: Поиск иголки в стоге сена")

    # Создаем список из 16 значений, где только 1 "хорошая" вершина
    n = 16
    f_values = [10.0] * n
    good_index = 7
    f_values[good_index] = 3.0  # Единственная хорошая вершина
    threshold = 5.0

    print(f"   Размер поискового пространства: {n} вершин")
    print(f"   Хороших вершин: 1 (индекс {good_index})")
    print(f"   Соотношение: 1/{n} = {1/n*100:.2f}%")

    print("\n   Классический поиск (линейный):")
    print(f"   • В среднем нужно проверить {n/2:.1f} вершин")
    print(f"   • В худшем случае: {n} проверок")

    print("\n   Квантовый поиск (Гровер):")
    found, probabilities = oracle.grover_search(f_values, threshold)

    if found and probabilities:
        max_prob = max(probabilities)
        max_idx = probabilities.index(max_prob)
        uniform_prob = 1.0 / n

        print(f"   • Найдена вершина: индекс {max_idx}")
        print(f"   • Вероятность: {max_prob:.4f} (равномерная: {uniform_prob:.4f})")
        print(f"   • Усиление: {max_prob/uniform_prob:.1f}x")
        print(f"   • Квантовое ускорение: ~√{n} = {np.sqrt(n):.1f}x")

    # Пример 2: Множество хороших вершин
    print("\n Пример 2: Поиск среди множества хороших вариантов")

    f_values = [10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 5.0]
    threshold = 6.0

    good_count = sum(1 for f in f_values if f < threshold)
    print(f"   Всего вершин: {len(f_values)}")
    print(f"   Хороших вершин: {good_count} ({good_count/len(f_values)*100:.0f}%)")

    print("\n   Классический поиск:")
    print(f"   • В среднем: {len(f_values)/(good_count+1):.1f} проверок до первой хорошей")

    print("\n   Квантовый поиск:")
    found, probabilities = oracle.grover_search(f_values, threshold)

    if probabilities:
        # Суммируем вероятности хороших вершин
        good_indices = [i for i, f in enumerate(f_values) if f < threshold]
        good_prob = sum(probabilities[i] for i in good_indices)
        bad_prob = 1.0 - good_prob

        print(f"   • Вероятность найти хорошую вершину: {good_prob:.3f}")
        print(f"   • Отношение хорошие/плохие: {good_prob/bad_prob:.2f}:1")

        # Показываем усиление каждой хорошей вершины
        print(f"   • Усиление хороших вершин:")
        for i in good_indices:
            amplification = probabilities[i] / (1.0/len(f_values))
            print(f"     Вершина {i} (f={f_values[i]}): {amplification:.1f}x")

def demo_real_world_scenario():
    """Демонстрация реального сценария использования"""
    print("\n" + "="*80)
    print("  РЕАЛЬНЫЙ СЦЕНАРИЙ: МАРШРУТИЗАЦИЯ В ГОРОДЕ")
    print("="*80)

    # Создаем граф, имитирующий городскую сеть
    print("\n Моделируем городскую сеть с:")
    print("   • 25 перекрестками (вершины)")
    print("   • Дорогами разной пропускной способности (веса рёбер)")
    print("   • Пробками на некоторых участках")
    print("   • Альтернативными маршрутами")

    # Создаем граф города
    vertices = []
    for i in range(25):
        row = i // 5
        col = i % 5
        vertices.append(Vertex(i, col * 100, row * 100))  # Координаты в метрах

    edges = []

    # Горизонтальные дороги
    for row in range(5):
        for col in range(4):
            from_v = row * 5 + col
            to_v = row * 5 + col + 1
            # Некоторые дороги загружены (больший вес)
            weight = 1.0 if col % 2 == 0 else 2.0
            edges.append(Edge(from_v, to_v, weight))
            edges.append(Edge(to_v, from_v, weight))

    # Вертикальные дороги
    for col in range(5):
        for row in range(4):
            from_v = row * 5 + col
            to_v = (row + 1) * 5 + col
            weight = 1.0 if row % 2 == 0 else 1.5
            edges.append(Edge(from_v, to_v, weight))
            edges.append(Edge(to_v, from_v, weight))

    # Диагональные дороги (проспекты)
    diagonals = [(0, 6, 1.2), (1, 7, 1.2), (2, 8, 1.2), (3, 9, 1.2),
                 (5, 11, 1.2), (6, 12, 1.2), (7, 13, 1.2), (8, 14, 1.2)]

    for from_v, to_v, weight in diagonals:
        edges.append(Edge(from_v, to_v, weight))
        edges.append(Edge(to_v, from_v, weight))

    graph = Graph(vertices, edges)

    print(f"\n Параметры графа:")
    print(f"   • Перекрестков: {len(vertices)}")
    print(f"   • Дорог: {len(edges)}")
    print(f"   • Средняя степень вершины: {len(edges)/len(vertices):.1f}")

    # Тестируем маршрут из северо-западного угла в юго-восточный
    start, goal = 0, 24

    print(f"\n Задача: найти путь из точки {start} в точку {goal}")
    print("   (северо-западный угол → юго-восточный угол)")

    # Классический A*
    print("\n" + "-"*60)
    print("1. КЛАССИЧЕСКИЙ АЛГОРИТМ A*")
    print("-"*60)

    classical_config = RouterConfig(strategy=SearchStrategy.PURE_CLASSICAL)
    classical_engine = HybridRoutingEngine(classical_config)
    classical_engine.set_graph(graph)

    start_time = time.perf_counter()
    classical_path, classical_cost, classical_metrics = classical_engine.find_path(start, goal)
    classical_time = time.perf_counter() - start_time

    print(f"   Найденный путь: {' → '.join(map(str, classical_path))}")
    print(f"   Длина пути: {len(classical_path)} перекрестков")
    print(f"   Оценочная стоимость: {classical_cost:.1f} усл.ед.")
    print(f"   Посещено перекрестков: {classical_metrics.total_nodes_visited}")
    print(f"   Время расчета: {classical_time*1000:.1f} мс")

    # Гибридный алгоритм
    print("\n" + "-"*60)
    print("2. ГИБРИДНЫЙ АЛГОРИТМ (A* + Гровер)")
    print("-"*60)

    hybrid_config = RouterConfig(
        backend=QuantumBackend.MATRIX_EMULATION,
        strategy=SearchStrategy.HYBRID_CONSERVATIVE,
        quantum_threshold=0.8,
        confidence_threshold=0.15
    )

    hybrid_engine = HybridRoutingEngine(hybrid_config)
    hybrid_engine.set_graph(graph)

    start_time = time.perf_counter()
    hybrid_path, hybrid_cost, hybrid_metrics = hybrid_engine.find_path(start, goal)
    hybrid_time = time.perf_counter() - start_time

    print(f"   Найденный путь: {' → '.join(map(str, hybrid_path))}")
    print(f"   Длина пути: {len(hybrid_path)} перекрестков")
    print(f"   Оценочная стоимость: {hybrid_cost:.1f} усл.ед.")
    print(f"   Посещено перекрестков: {hybrid_metrics.total_nodes_visited}")
    print(f"   Отсечено перекрестков: {hybrid_metrics.pruned_nodes}")
    print(f"   Квантовых запросов: {hybrid_metrics.quantum_calls}")
    print(f"   Время расчета: {hybrid_time*1000:.1f} мс")

    # Сравнение
    print("\n" + "-"*60)
    print("3. СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-"*60)

    if classical_path and hybrid_path:
        # Проверяем, нашли ли оптимальный путь
        if abs(classical_cost - hybrid_cost) < 0.1:
            print("    Оба алгоритма нашли одинаково оптимальный путь!")
        else:
            print(f"     Разница в стоимости: {abs(classical_cost - hybrid_cost):.2f}")

        # Эффективность
        visited_reduction = ((classical_metrics.total_nodes_visited - hybrid_metrics.total_nodes_visited) /
                            classical_metrics.total_nodes_visited * 100)

        time_speedup = classical_time / hybrid_time if hybrid_time > 0 else 1

        print(f"\n    Эффективность гибридного алгоритма:")
        print(f"   • Сокращение поиска: {visited_reduction:+.1f}%")
        print(f"   • Отсечено вершин: {hybrid_metrics.pruned_nodes}")
        print(f"   • Ускорение расчета: {time_speedup:.2f}x")

        if hybrid_metrics.quantum_calls > 0:
            print(f"   • Квантовых решений: {hybrid_metrics.quantum_calls}")
            print(f"   • Время на квант.часть: {hybrid_metrics.quantum_time*1000:.1f} мс")



def main():
    """Главная функция демонстрации"""

    # Запускаем все демонстрации
    try:
        # 1. Бенчмарк-тестирование
        results = benchmark_algorithms()

        # 2. Визуализация результатов
        visualize_results(results)

        # 3. Демонстрация квантового преимущества
        demo_quantum_advantage()

        # 4. Реальный сценарий
        demo_real_world_scenario()

    except KeyboardInterrupt:
        print("\n\n  Демонстрация прервана пользователем")
    except Exception as e:
        print(f"\n\n Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()