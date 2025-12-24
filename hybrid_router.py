"""
Главный модуль для удобного импорта: import hybrid_router as hr
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.architecture import (
    HybridRoutingEngine,
    RouterConfig,
    QuantumBackend,
    SearchStrategy,
    PerformanceMetrics,
    Graph,
    Vertex,
    Edge
)

from src.classical.graph_utils import (
    create_dead_end_graph,
    create_complex_test_graph,
    create_maze_graph,
    create_test_graphs
)

from src.quantum.matrix_oracle import MatrixOracle
from src.quantum.qiskit_oracle import QiskitOracle
from src.quantum.qiskit_hardware_oracle import QiskitHardwareOracle

from src.server.quantum_service import QuantumService
from src.server.main_server import MainServer

from src.client.api_client import APIClient

__version__ = "1.0.1"
__author__ = "Khuri Pascal"


def find_path(graph, start, goal, config_dict=None, **kwargs):
    """
    Упрощенная функция поиска пути

    Args:
        graph: Объект Graph из src.architecture
        start: ID стартовой вершины
        goal: ID целевой вершины
        config_dict: Словарь с настройками
        **kwargs: Дополнительные параметры для конфигурации

    Returns:
        tuple: (path, cost, metrics)
    """
    if config_dict is None:
        config_dict = {}

    config_dict.update(kwargs)

    config = RouterConfig.from_dict(config_dict) if config_dict else RouterConfig()

    engine = HybridRoutingEngine(config)
    engine.set_graph(graph)
    return engine.find_path(start, goal)


def create_graph(vertices_data, edges_data):
    """
    Упрощенное создание графа

    Args:
        vertices_data: Список словарей с вершинами
            Пример: [{"id": 0, "x": 0, "y": 0}, ...]
        edges_data: Список словарей с ребрами
            Пример: [{"from_vertex": 0, "to_vertex": 1, "weight": 1.0}, ...]

    Returns:
        Graph: Объект графа
    """
    vertices = [Vertex(**v) for v in vertices_data]
    edges = [Edge(**e) for e in edges_data]
    return Graph(vertices, edges)


def quick_route(graph_data, start, goal, strategy="hybrid_conservative"):
    """
    Самый простой способ найти маршрут

    Args:
        graph_data: Словарь с данными графа
            {"vertices": [...], "edges": [...]}
        start: Стартовая вершина
        goal: Целевая вершина
        strategy: Стратегия поиска

    Returns:
        dict: Результат поиска
    """
    graph = create_graph(graph_data.get("vertices", []),
                         graph_data.get("edges", []))

    path, cost, metrics = find_path(
        graph, start, goal,
        config_dict={"strategy": strategy}
    )

    return {
        "success": len(path) > 0,
        "path": path,
        "cost": cost,
        "metrics": metrics.to_dict()
    }


def compare_strategies(graph, start, goal, strategies=None):
    """
    Сравнение разных стратегий поиска

    Args:
        graph: Граф
        start: Стартовая вершина
        goal: Целевая вершина
        strategies: Список стратегий для сравнения

    Returns:
        dict: Результаты сравнения
    """
    if strategies is None:
        strategies = [
            ("pure_classical", "Чисто классический"),
            ("hybrid_conservative", "Гибридный консервативный"),
            ("hybrid_aggressive", "Гибридный агрессивный"),
            ("adaptive", "Адаптивный")
        ]

    results = {}

    for strategy_key, strategy_name in strategies:
        config = RouterConfig(strategy=SearchStrategy(strategy_key))
        engine = HybridRoutingEngine(config)
        engine.set_graph(graph)

        import time
        start_time = time.time()
        path, cost, metrics = engine.find_path(start, goal)
        elapsed = time.time() - start_time

        results[strategy_key] = {
            "name": strategy_name,
            "path": path,
            "cost": cost,
            "time": elapsed,
            "visited": metrics.total_nodes_visited,
            "pruned": metrics.pruned_nodes,
            "quantum_calls": metrics.quantum_calls
        }

    return results


class SimpleNavigator:
    """Упрощенный навигатор для пользователя"""

    def __init__(self, config=None):
        """
        Args:
            config: Словарь с настройками или объект RouterConfig
        """
        if isinstance(config, dict):
            self.config = RouterConfig.from_dict(config)
        else:
            self.config = config or RouterConfig()

        self.engine = HybridRoutingEngine(self.config)
        self.graph = None

    def set_graph(self, vertices, edges):
        """Установить граф из списков вершин и ребер"""
        self.graph = create_graph(vertices, edges)
        self.engine.set_graph(self.graph)
        return self

    def set_graph_object(self, graph):
        """Установить готовый объект Graph"""
        self.graph = graph
        self.engine.set_graph(graph)
        return self

    def find_route(self, start, goal):
        """Найти маршрут"""
        if self.graph is None:
            raise ValueError("Граф не установлен. Используйте set_graph()")

        return self.engine.find_path(start, goal)

    def get_report(self):
        """Получить отчет о работе"""
        return self.engine.get_detailed_report()



__all__ = [
    # Архитектура
    "HybridRoutingEngine",
    "RouterConfig",
    "QuantumBackend",
    "SearchStrategy",
    "PerformanceMetrics",
    "Graph",
    "Vertex",
    "Edge",

    # Функции для графов
    "create_dead_end_graph",
    "create_complex_test_graph",
    "create_maze_graph",
    "create_test_graphs",

    # Квантовые оракулы
    "MatrixOracle",
    "QiskitOracle",
    "QiskitHardwareOracle",

    # Сервер
    "QuantumService",
    "MainServer",

    # Клиент
    "APIClient",


    "find_path",
    "create_graph",
    "quick_route",
    "compare_strategies",
    "SimpleNavigator",

    # Версия
    "__version__",
    "__author__"
]

