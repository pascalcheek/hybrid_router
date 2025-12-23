"""
Интеграционные тесты гибридного алгоритма
"""

import pytest
from src.architecture import (
    HybridRoutingEngine,
    RouterConfig,
    QuantumBackend,
    SearchStrategy
)
from src.classical.graph_utils import create_dead_end_graph



def qiskit_available():
    try:
        from qiskit import QuantumCircuit
        return True
    except ImportError:
        return False

def test_hybrid_engine_initialization():
    """Тест инициализации гибридного движка"""
    config = RouterConfig(
        backend=QuantumBackend.MATRIX_EMULATION,
        strategy=SearchStrategy.HYBRID_CONSERVATIVE
    )

    engine = HybridRoutingEngine(config)

    assert engine.config == config
    assert engine.graph is None
    assert engine.goal is None
    assert engine.heuristic_cache == {}
    assert engine.decision_cache == {}
    assert engine.metrics.total_nodes_visited == 0


def test_set_graph():
    """Тест установки графа"""
    config = RouterConfig()
    engine = HybridRoutingEngine(config)

    graph = create_dead_end_graph()
    engine.set_graph(graph)

    assert engine.graph == graph
    assert engine.heuristic_cache == {}
    assert engine.decision_cache == {}
    assert engine.metrics.total_nodes_visited == 0


def test_heuristic_calculation():
    """Тест расчета эвристики"""
    config = RouterConfig()
    engine = HybridRoutingEngine(config)

    graph = create_dead_end_graph()
    engine.set_graph(graph)

    # Расстояние между вершинами 0 и 3
    distance = engine.heuristic(0, 3)
    expected = 3.0  # sqrt((3-0)^2 + (0-0)^2)
    assert abs(distance - expected) < 0.001

    # Проверка кэширования
    assert (0, 3) in engine.heuristic_cache


def test_collect_subgraph():
    """Тест сбора подграфа"""
    config = RouterConfig()
    engine = HybridRoutingEngine(config)

    graph = create_dead_end_graph()
    engine.set_graph(graph)

    g_score = {0: 0, 1: 1.0, 2: 1.0}
    vertices, f_values = engine.collect_subgraph(0, g_score, 3, depth=1)

    # Из вершины 0 на глубине 1 должны быть вершины 0, 1, 2
    assert set(vertices) == {0, 1, 2}
    assert len(f_values) == 3

    # Проверяем расчет f-значений
    # f = g + h
    # Для вершины 0: g=0, h(0,3)=3 → f=3
    # Для вершины 1: g=1, h(1,3)=2 → f=3
    # Для вершины 2: g=1, h(2,3)=2 → f=3 (но на самом деле h(2,3)=1? проверяем)
    # Vertex(2, 2, 0) до Vertex(3, 3, 0) = sqrt(1) = 1
    # Так что f(2) = 1 + 1 = 2

    # Найдем индексы вершин в списке
    idx_0 = vertices.index(0)
    idx_1 = vertices.index(1)
    idx_2 = vertices.index(2)

    assert abs(f_values[idx_0] - 3.0) < 0.1
    assert abs(f_values[idx_1] - 3.0) < 0.1
    assert abs(f_values[idx_2] - 2.0) < 0.1


def test_find_path_pure_classical():
    """Тест поиска пути в чисто классическом режиме"""
    config = RouterConfig(
        strategy=SearchStrategy.PURE_CLASSICAL
    )
    engine = HybridRoutingEngine(config)

    graph = create_dead_end_graph()
    engine.set_graph(graph)

    path, cost, metrics = engine.find_path(0, 3)

    assert path == [0, 1, 3]
    assert abs(cost - 2.0) < 0.001
    assert metrics.total_nodes_visited > 0
    assert metrics.quantum_calls == 0
    assert not metrics.fallback_used


def test_find_path_hybrid_matrix():
    """Тест поиска пути в гибридном режиме с матричной эмуляцией"""
    config = RouterConfig(
        backend=QuantumBackend.MATRIX_EMULATION,
        strategy=SearchStrategy.HYBRID_CONSERVATIVE,
        quantum_threshold=0.9,
        confidence_threshold=0.15
    )
    engine = HybridRoutingEngine(config)

    graph = create_dead_end_graph()
    engine.set_graph(graph)

    path, cost, metrics = engine.find_path(0, 3)

    assert len(path) > 0
    assert path == [0, 1, 3]  # ← ДОБАВЬТЕ ЭТУ ПРОВЕРКУ
    assert abs(cost - 2.0) < 0.001  # ← ДОБАВЬТЕ ЭТУ ПРОВЕРКУ
    assert metrics.total_nodes_visited > 0

    # Замените проверку времени на более надежную
    assert metrics.execution_time >= 0.0


def test_config_serialization():
    """Тест сериализации конфигурации"""
    config = RouterConfig(
        backend=QuantumBackend.MATRIX_EMULATION,
        strategy=SearchStrategy.HYBRID_CONSERVATIVE,
        quantum_threshold=0.85,
        confidence_threshold=0.2,
        max_subgraph_size=10,
        use_fallback=True,
        cache_enabled=False
    )

    config_dict = config.to_dict()

    assert config_dict["backend"] == "matrix_emulation"
    assert config_dict["strategy"] == "hybrid_conservative"
    assert config_dict["quantum_threshold"] == 0.85
    assert config_dict["confidence_threshold"] == 0.2
    assert config_dict["max_subgraph_size"] == 10
    assert config_dict["use_fallback"] == True
    assert config_dict["cache_enabled"] == False

    # Обратная десериализация
    config2 = RouterConfig.from_dict(config_dict)

    assert config2.backend == config.backend
    assert config2.strategy == config.strategy
    assert config2.quantum_threshold == config.quantum_threshold
    assert config2.confidence_threshold == config.confidence_threshold
    assert config2.max_subgraph_size == config.max_subgraph_size
    assert config2.use_fallback == config.use_fallback
    assert config2.cache_enabled == config.cache_enabled


def test_performance_metrics():
    """Тест метрик производительности"""
    from src.architecture import PerformanceMetrics

    metrics = PerformanceMetrics(
        total_nodes_visited=100,
        quantum_calls=10,
        pruned_nodes=20,
        fallback_used=True,
        execution_time=1.5,
        quantum_time=0.3,
        classical_time=1.2,
        avg_decision_quality=0.8,
        memory_usage_mb=50.5
    )

    metrics_dict = metrics.to_dict()

    assert metrics_dict["total_nodes_visited"] == 100
    assert metrics_dict["quantum_calls"] == 10
    assert metrics_dict["pruned_nodes"] == 20
    assert metrics_dict["fallback_used"] == True
    assert metrics_dict["execution_time"] == 1.5
    assert metrics_dict["quantum_time"] == 0.3
    assert metrics_dict["classical_time"] == 1.2
    assert metrics_dict["avg_decision_quality"] == 0.8
    assert metrics_dict["memory_usage_mb"] == 50.5


def test_detailed_report():
    """Тест детального отчета"""
    config = RouterConfig()
    engine = HybridRoutingEngine(config)

    graph = create_dead_end_graph()
    engine.set_graph(graph)

    # Выполняем поиск
    engine.find_path(0, 3)

    report = engine.get_detailed_report()

    assert "config" in report
    assert "metrics" in report
    assert "decision_history_count" in report
    assert "cache_hits" in report
    assert "cache_misses" in report
    assert "pruning_efficiency" in report
    assert "quantum_acceleration" in report

    # Проверяем структуру отчета
    assert isinstance(report["config"], dict)
    assert isinstance(report["metrics"], dict)


@pytest.mark.skipif(not qiskit_available(),
                   reason="Требует установленного Qiskit")
def test_qiskit_backend():
    """Тест работы с Qiskit бэкендом"""
    try:
        from qiskit_aer import Aer

        config = RouterConfig(
            backend=QuantumBackend.QISKIT_SIMULATOR,
            strategy=SearchStrategy.HYBRID_CONSERVATIVE,
            qiskit_shots=512
        )

        engine = HybridRoutingEngine(config)

        graph = create_dead_end_graph()
        engine.set_graph(graph)

        path, cost, metrics = engine.find_path(0, 3)

        assert len(path) > 0
        assert cost < float('inf')

    except ImportError:
        pytest.skip("Qiskit не установлен")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--run-qiskit"])