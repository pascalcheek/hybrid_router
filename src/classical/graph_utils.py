"""
Утилиты для работы с графами и создания тестовых графов
"""

from ..architecture import Graph, Vertex, Edge
from typing import List, Tuple

def create_dead_end_graph() -> Graph:
    """Граф с явной тупиковой ветвью для демонстрации эффекта отсечения"""
    vertices = [Vertex(i, i, 0) for i in range(5)]
    edges = [
        Edge(0, 1, 1.0),
        Edge(0, 2, 1.0),
        Edge(1, 3, 1.0),
        Edge(2, 4, 1.0),
        Edge(4, 3, 10.0)
    ]
    return Graph(vertices, edges)

def create_complex_test_graph() -> Graph:
    """Сложный граф с несколькими тупиками и альтернативными путями"""
    vertices = [Vertex(id=i, x=i % 5, y=i // 5) for i in range(15)]
    edges = [
        # Основной оптимальный путь: 0-1-7-8-13-14
        Edge(0, 1, 1.0), Edge(1, 7, 1.0), Edge(7, 8, 1.0),
        Edge(8, 13, 1.0), Edge(13, 14, 1.0),

        # Тупиковая ветвь: 0-5-10
        Edge(0, 5, 2.0), Edge(5, 10, 3.0), Edge(10, 14, 10.0),

        # Еще одна тупиковая ветвь: 0-6-11-12
        Edge(0, 6, 1.5), Edge(6, 11, 2.0), Edge(11, 12, 3.0),

        # Альтернативные пути
        Edge(1, 2, 1.0), Edge(2, 3, 1.0), Edge(3, 4, 1.0), Edge(4, 9, 1.0),
        Edge(9, 14, 1.0),

        # Дополнительные связи
        Edge(2, 7, 1.1), Edge(3, 8, 1.2)
    ]
    return Graph(vertices, edges)

def create_maze_graph() -> Graph:
    """Граф-лабиринт с множеством тупиков и одним оптимальным путем"""
    vertices = [Vertex(i, i % 4, i // 4) for i in range(16)]
    edges = [
        # Оптимальный путь: 0-1-2-3-7-11-15
        Edge(0, 1, 1.0), Edge(1, 2, 1.0), Edge(2, 3, 1.0),
        Edge(3, 7, 1.0), Edge(7, 11, 1.0), Edge(11, 15, 1.0),

        # Тупиковые ветви
        Edge(0, 4, 1.0), Edge(4, 5, 1.0), Edge(5, 9, 2.0),
        Edge(1, 5, 1.5),
        Edge(2, 6, 2.0), Edge(6, 10, 3.0),

        # Альтернативные пути
        Edge(0, 5, 2.0), Edge(5, 10, 2.0), Edge(10, 15, 3.0),
        Edge(1, 6, 2.5), Edge(6, 11, 2.5),

        # Дополнительные тупики
        Edge(8, 9, 1.0), Edge(9, 10, 1.0),
        Edge(12, 13, 1.0), Edge(13, 14, 1.0),
    ]
    return Graph(vertices, edges)

def create_test_graphs() -> List[Tuple[str, Graph]]:
    """Создание списка тестовых графов"""
    return [
        ("Граф с тупиковой ветвью", create_dead_end_graph()),
        ("Сложный граф с тупиками", create_complex_test_graph()),
        ("Граф-лабиринт", create_maze_graph())
    ]