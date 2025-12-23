"""
Тесты классических компонентов
"""

import pytest
from src.classical.a_star import ClassicalAStar
from src.classical.graph_utils import (
    create_dead_end_graph,
    create_complex_test_graph,
    create_maze_graph
)

def test_classical_astar_initialization():
    """Тест инициализации классического A*"""
    graph = create_dead_end_graph()
    astar = ClassicalAStar(graph)

    assert astar.graph == graph
    assert astar.heuristic_cache == {}

def test_heuristic_function():
    """Тест эвристической функции"""
    graph = create_dead_end_graph()
    astar = ClassicalAStar(graph)

    distance = astar.heuristic(0, 3)
    expected = 3.0
    assert abs(distance - expected) < 0.001

    assert (0, 3) in astar.heuristic_cache
    assert astar.heuristic(0, 3) == distance

def test_find_path_simple():
    """Тест поиска пути в простом графе"""
    graph = create_dead_end_graph()
    astar = ClassicalAStar(graph)

    path, cost, visited = astar.find_path(0, 3)

    assert path == [0, 1, 3]
    assert abs(cost - 2.0) < 0.001
    assert visited > 0

def test_find_path_no_path():
    """Тест поиска пути, когда пути не существует"""
    from src.architecture import Graph, Vertex, Edge

    vertices = [Vertex(i, i, 0) for i in range(3)]
    edges = [Edge(0, 1, 1.0)]
    graph = Graph(vertices, edges)

    astar = ClassicalAStar(graph)
    path, cost, visited = astar.find_path(0, 2)

    assert path == []
    assert cost == float('inf')
    assert visited > 0

def test_find_path_multiple_paths():
    """Тест поиска пути с несколькими вариантами"""
    from src.architecture import Graph, Vertex, Edge

    vertices = [Vertex(i, i, 0) for i in range(4)]
    edges = [
        Edge(0, 1, 1.5),
        Edge(0, 2, 1.0),
        Edge(1, 3, 1.5),
        Edge(2, 3, 1.5)
    ]
    graph = Graph(vertices, edges)

    astar = ClassicalAStar(graph)
    path, cost, visited = astar.find_path(0, 3)

    assert path == [0, 2, 3]
    assert abs(cost - 2.5) < 0.001

def test_graph_utils_dead_end():
    """Тест создания графа с тупиковой ветвью"""
    graph = create_dead_end_graph()

    assert len(graph.vertices) == 5
    assert len(graph.edges) == 5

    assert 0 in graph.adjacency_list
    assert len(graph.adjacency_list[0]) == 2

    neighbors_0 = dict(graph.adjacency_list[0])
    assert neighbors_0[1] == 1.0
    assert neighbors_0[2] == 1.0

def test_graph_utils_complex():
    """Тест создания сложного графа"""
    graph = create_complex_test_graph()

    assert len(graph.vertices) == 15
    assert len(graph.edges) == 18  # ← ИСПРАВЛЕНО: было 16, стало 18

    assert 0 in graph.adjacency_list
    assert 14 in graph.adjacency_list

    neighbors_0 = [n for n, _ in graph.adjacency_list[0]]
    assert set(neighbors_0) == {1, 5, 6}

def test_graph_utils_maze():
    """Тест создания графа-лабиринта"""
    graph = create_maze_graph()

    assert len(graph.vertices) == 16
    assert len(graph.edges) == 21  # ← ИСПРАВЛЕНО: было 18, стало 21

    # Проверка оптимального пути
    assert (0, 1) in [(e.from_vertex, e.to_vertex) for e in graph.edges]
    assert (1, 2) in [(e.from_vertex, e.to_vertex) for e in graph.edges]
    assert (2, 3) in [(e.from_vertex, e.to_vertex) for e in graph.edges]
    assert (3, 7) in [(e.from_vertex, e.to_vertex) for e in graph.edges]
    assert (7, 11) in [(e.from_vertex, e.to_vertex) for e in graph.edges]
    assert (11, 15) in [(e.from_vertex, e.to_vertex) for e in graph.edges]

def test_graph_adjacency_symmetry():
    """Тест симметричности списка смежности для неориентированного графа"""
    graph = create_dead_end_graph()

    for edge in graph.edges:
        a, b = edge.from_vertex, edge.to_vertex
        weight = edge.weight

        assert b in dict(graph.adjacency_list[a])
        assert dict(graph.adjacency_list[a])[b] == weight

        assert a in dict(graph.adjacency_list[b])
        assert dict(graph.adjacency_list[b])[a] == weight

if __name__ == "__main__":
    pytest.main([__file__, "-v"])