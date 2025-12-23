"""
Главная архитектура гибридного квантово-классического алгоритма маршрутизации
"""

import numpy as np
import heapq
import math
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Конфигурация для реального квантового железа"""
    backend_name: str = "ibmq_qasm_simulator"  # или "ibm_brisbane" и т.д.
    shots: int = 1024
    optimization_level: int = 3
    use_error_mitigation: bool = True
    max_retries: int = 3
    retry_delay: float = 5.0
    queue_timeout: int = 3600  # 1 час максимум в очереди
    save_job_results: bool = True
    job_results_dir: str = "quantum_jobs"

    def to_dict(self) -> Dict:
        return {
            "backend_name": self.backend_name,
            "shots": self.shots,
            "optimization_level": self.optimization_level,
            "use_error_mitigation": self.use_error_mitigation,
            "max_retries": self.max_retries
        }


class QuantumBackend(Enum):
    """Типы квантовых бэкендов"""
    MATRIX_EMULATION = "matrix_emulation"
    QISKIT_SIMULATOR = "qiskit_simulator"
    QISKIT_HARDWARE = "qiskit_hardware"


class SearchStrategy(Enum):
    """Стратегии поиска"""
    PURE_CLASSICAL = "pure_classical"
    HYBRID_CONSERVATIVE = "hybrid_conservative"
    HYBRID_AGGRESSIVE = "hybrid_aggressive"
    ADAPTIVE = "adaptive"


@dataclass
class Vertex:
    """Вершина графа"""
    id: int
    x: float = 0
    y: float = 0


@dataclass
class Edge:
    """Ребро графа"""
    from_vertex: int
    to_vertex: int
    weight: float


@dataclass
class Graph:
    """Граф для маршрутизации"""
    vertices: List[Vertex]
    edges: List[Edge]
    adjacency_list: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)

    def __post_init__(self):
        """Инициализация списка смежности из рёбер графа"""
        self.adjacency_list = {}
        for edge in self.edges:
            if edge.from_vertex not in self.adjacency_list:
                self.adjacency_list[edge.from_vertex] = []
            self.adjacency_list[edge.from_vertex].append((edge.to_vertex, edge.weight))

            # Для неориентированного графа добавляем обратное ребро
            if edge.to_vertex not in self.adjacency_list:
                self.adjacency_list[edge.to_vertex] = []
            self.adjacency_list[edge.to_vertex].append((edge.from_vertex, edge.weight))


@dataclass
class RouterConfig:
    """Конфигурация маршрутизатора"""
    backend: QuantumBackend = QuantumBackend.MATRIX_EMULATION
    strategy: SearchStrategy = SearchStrategy.HYBRID_CONSERVATIVE
    quantum_threshold: float = 0.9
    confidence_threshold: float = 0.15
    max_subgraph_size: int = 8
    grover_iterations: Optional[int] = None
    use_fallback: bool = True
    cache_enabled: bool = True
    visualizations_enabled: bool = True
    log_decisions: bool = True

    # Параметры для Qiskit
    qiskit_shots: int = 1024
    qiskit_optimization_level: int = 1

    # НОВЫЙ ПАРАМЕТР для реального железа
    hardware_config: HardwareConfig = field(default_factory=HardwareConfig)

    # НОВЫЙ ПАРАМЕТР: тестовый режим
    test_mode: bool = False
    test_iterations: int = 3  # Количество тестовых запусков

    def to_dict(self) -> Dict:
        return {
            "backend": self.backend.value,
            "strategy": self.strategy.value,
            "quantum_threshold": self.quantum_threshold,
            "confidence_threshold": self.confidence_threshold,
            "max_subgraph_size": self.max_subgraph_size,
            "grover_iterations": self.grover_iterations,
            "use_fallback": self.use_fallback,
            "cache_enabled": self.cache_enabled,
            "test_mode": self.test_mode,
            "test_iterations": self.test_iterations,
            "hardware_config": self.hardware_config.to_dict()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RouterConfig':
        if "hardware_config" in config_dict:
            hardware_config = HardwareConfig(**config_dict["hardware_config"])
        else:
            hardware_config = HardwareConfig()

        return cls(
            backend=QuantumBackend(config_dict.get("backend", "matrix_emulation")),
            strategy=SearchStrategy(config_dict.get("strategy", "hybrid_conservative")),
            quantum_threshold=config_dict.get("quantum_threshold", 0.9),
            confidence_threshold=config_dict.get("confidence_threshold", 0.15),
            max_subgraph_size=config_dict.get("max_subgraph_size", 8),
            grover_iterations=config_dict.get("grover_iterations"),
            use_fallback=config_dict.get("use_fallback", True),
            cache_enabled=config_dict.get("cache_enabled", True),
            hardware_config=hardware_config,
            test_mode=config_dict.get("test_mode", False),
            test_iterations=config_dict.get("test_iterations", 3)
        )


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    total_nodes_visited: int = 0
    quantum_calls: int = 0
    pruned_nodes: int = 0
    fallback_used: bool = False
    execution_time: float = 0.0
    quantum_time: float = 0.0
    classical_time: float = 0.0
    avg_decision_quality: float = 0.0
    memory_usage_mb: float = 0.0

    hardware_backend: str = ""
    hardware_shots: int = 0
    hardware_queue_time: float = 0.0
    hardware_execution_time: float = 0.0
    hardware_errors: int = 0
    job_id: str = ""
    noise_level: float = 0.0
    success_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_nodes_visited": self.total_nodes_visited,
            "quantum_calls": self.quantum_calls,
            "pruned_nodes": self.pruned_nodes,
            "fallback_used": self.fallback_used,
            "execution_time": self.execution_time,
            "quantum_time": self.quantum_time,
            "classical_time": self.classical_time,
            "avg_decision_quality": self.avg_decision_quality,
            "memory_usage_mb": self.memory_usage_mb,
            "hardware_backend": self.hardware_backend,
            "hardware_shots": self.hardware_shots,
            "hardware_queue_time": self.hardware_queue_time,
            "hardware_execution_time": self.hardware_execution_time,
            "hardware_errors": self.hardware_errors,
            "job_id": self.job_id,
            "noise_level": self.noise_level,
            "success_rate": self.success_rate
        }

    def update(self, other_dict: Dict):
        """Обновление метрик из словаря"""
        for key, value in other_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class QuantumOracleFactory:
    """Фабрика для создания квантовых оракулов"""

    @staticmethod
    def create_oracle(backend: QuantumBackend, **kwargs):
        """Создание оракула для заданного бэкенда"""
        if backend == QuantumBackend.MATRIX_EMULATION:
            from .quantum.matrix_oracle import MatrixOracle
            return MatrixOracle(**kwargs)
        elif backend == QuantumBackend.QISKIT_SIMULATOR:
            from .quantum.qiskit_oracle import QiskitOracle
            return QiskitOracle(**kwargs)
        elif backend == QuantumBackend.QISKIT_HARDWARE:
            from .quantum.qiskit_hardware_oracle import QiskitHardwareOracle
            return QiskitHardwareOracle(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")


class HybridRoutingEngine:
    """
    Основной движок гибридной маршрутизации
    Объединяет классический A* с квантовым ускорением
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self.graph = None
        self.goal = None
        self.heuristic_cache = {}
        self.decision_cache = {}
        self.metrics = PerformanceMetrics()
        self.decision_history = []

        # Инициализация квантового оракула
        oracle_kwargs = {}
        if config.backend == QuantumBackend.QISKIT_SIMULATOR:
            oracle_kwargs["shots"] = config.qiskit_shots
            oracle_kwargs["optimization_level"] = config.qiskit_optimization_level

        self.quantum_oracle = QuantumOracleFactory.create_oracle(
            config.backend, **oracle_kwargs
        )

        logger.info(f"Initialized HybridRoutingEngine with {config.backend.value}")

    def set_graph(self, graph: Graph):
        """Установка графа для маршрутизации"""
        self.graph = graph
        self.heuristic_cache = {}
        self.decision_cache = {}
        self.metrics = PerformanceMetrics()
        self.decision_history = []

    def heuristic(self, a: int, b: int) -> float:
        """Эвристическая функция"""
        cache_key = (a, b)
        if cache_key not in self.heuristic_cache:
            v1 = self.graph.vertices[a]
            v2 = self.graph.vertices[b]
            # Евклидово расстояние как допустимая эвристика
            distance = math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)
            self.heuristic_cache[cache_key] = distance
        return self.heuristic_cache[cache_key]

    def collect_subgraph(self, current: int, g_score: Dict[int, float],
                         goal: int, depth: int = 2) -> Tuple[List[int], List[float]]:
        """Сбор подграфа для квантового анализа"""
        vertices_set = set([current])
        f_values_dict = {}

        queue = [(current, 0, g_score.get(current, 0))]
        visited = set([current])

        while queue and len(vertices_set) < self.config.max_subgraph_size:
            node, cur_depth, g_to_node = queue.pop(0)

            if node not in f_values_dict:
                if node in g_score:
                    g_exact = g_score[node]
                else:
                    g_exact = g_to_node
                f_value = g_exact + self.heuristic(node, goal)
                f_values_dict[node] = f_value

            if cur_depth >= depth:
                continue

            for neighbor, weight in self.graph.adjacency_list.get(node, []):
                if neighbor not in visited and len(vertices_set) < self.config.max_subgraph_size:
                    visited.add(neighbor)
                    vertices_set.add(neighbor)
                    new_g = g_to_node + weight
                    queue.append((neighbor, cur_depth + 1, new_g))

        vertices_list = list(vertices_set)
        f_list = [f_values_dict.get(v, float('inf')) for v in vertices_list]

        return vertices_list, f_list

    def should_explore_quantum(self, current: int, f_values: List[float],
                               subgraph_vertices: List[int], threshold: float) -> Tuple[bool, Dict]:
        """Квантовый анализ подграфа с помощью алгоритма Гровера"""
        start_time = time.time()

        # Проверка кэша
        cache_key = (tuple(subgraph_vertices), tuple(f_values), threshold)
        if self.config.cache_enabled and cache_key in self.decision_cache:
            result = self.decision_cache[cache_key]
            result["cached"] = True
            result["quantum_time"] = 0.0
            if "should_explore" not in result:
                result["should_explore"] = True  # Значение по умолчанию
            return result["should_explore"], result

        # Анализ особых случаев
        direct_neighbors = [n for n, _ in self.graph.adjacency_list.get(current, [])]

        # Исключения: всегда раскрываем в особых случаях
        special_cases = self._check_special_cases(current, subgraph_vertices, f_values,
                                                  threshold, direct_neighbors)
        if special_cases["force_explore"]:
            decision_info = {
                "vertex": current,
                "decision": "explore",
                "reason": special_cases["reason"],
                "special_case": True,
                "quantum_time": 0.0,
                "cached": False
            }
            if self.config.cache_enabled:
                self.decision_cache[cache_key] = decision_info
            return True, decision_info

        # Квантовый анализ с помощью Гровера
        found, probabilities = self.quantum_oracle.grover_search(
            f_values, threshold, self.config.grover_iterations
        )

        quantum_time = time.time() - start_time
        self.metrics.quantum_time += quantum_time
        self.metrics.quantum_calls += 1

        # Принятие решения на основе стратегии
        decision_info = self._make_decision(
            current, f_values, subgraph_vertices, threshold,
            found, probabilities, direct_neighbors, quantum_time
        )

        if self.config.cache_enabled:
            self.decision_cache[cache_key] = decision_info

        return decision_info["should_explore"], decision_info

    def _check_special_cases(self, current: int, subgraph_vertices: List[int],
                             f_values: List[float], threshold: float,
                             direct_neighbors: List[int]) -> Dict:
        """Проверка особых случаев, когда квантовый анализ не нужен"""
        result = {"force_explore": False, "reason": None}

        # Текущая вершина близка к цели
        if self.heuristic(current, self.goal) < 2.0:
            result["force_explore"] = True
            result["reason"] = "close_to_goal"
            return result

        # Маленький подграф
        if len(subgraph_vertices) <= 2:
            result["force_explore"] = True
            result["reason"] = "small_subgraph"
            return result

        # Все вершины хорошие (f < threshold)
        good_count = sum(1 for f in f_values if f < threshold)
        if good_count == len(f_values):
            result["force_explore"] = True
            result["reason"] = "all_vertices_good"
            return result

        # Цель в подграфе
        if self.goal in subgraph_vertices:
            result["force_explore"] = True
            result["reason"] = "goal_in_subgraph"
            return result

        return result

    def _make_decision(self, current: int, f_values: List[float],
                       subgraph_vertices: List[int], threshold: float,
                       found: bool, probabilities: List[float],
                       direct_neighbors: List[int], quantum_time: float) -> Dict:
        """Принятие решения на основе стратегии"""
        decision_info = {
            "vertex": current,
            "subgraph_size": len(subgraph_vertices),
            "threshold": threshold,
            "found_by_grover": found,
            "probabilities": probabilities,
            "direct_neighbors": direct_neighbors,
            "quantum_time": quantum_time,
            "cached": False
        }

        if self.config.strategy == SearchStrategy.PURE_CLASSICAL:
            decision_info["decision"] = "explore"
            decision_info["reason"] = "pure_classical_strategy"
            decision_info["should_explore"] = True

        elif self.config.strategy == SearchStrategy.HYBRID_CONSERVATIVE:
            decision_info.update(self._conservative_strategy(
                current, f_values, subgraph_vertices, threshold,
                found, probabilities, direct_neighbors
            ))

        elif self.config.strategy == SearchStrategy.HYBRID_AGGRESSIVE:
            decision_info.update(self._aggressive_strategy(
                current, f_values, subgraph_vertices, threshold,
                found, probabilities, direct_neighbors
            ))

        elif self.config.strategy == SearchStrategy.ADAPTIVE:
            decision_info.update(self._adaptive_strategy(
                current, f_values, subgraph_vertices, threshold,
                found, probabilities, direct_neighbors
            ))

        # Логирование решения
        if self.config.log_decisions:
            self.decision_history.append(decision_info)
            logger.debug(f"Decision for vertex {current}: {decision_info['decision']} "
                         f"(reason: {decision_info['reason']})")

        return decision_info

    def _conservative_strategy(self, current: int, f_values: List[float],
                               subgraph_vertices: List[int], threshold: float,
                               found: bool, probabilities: List[float],
                               direct_neighbors: List[int]) -> Dict:
        """Консервативная стратегия: минимальный риск"""
        should_explore = True

        if not found:
            # Гровер не нашел хороших вершин
            return {
                "decision": "prune",
                "reason": "grover_no_good_vertices",
                "should_explore": False
            }

        if probabilities:
            max_prob = max(probabilities)
            max_idx = probabilities.index(max_prob)
            best_vertex = subgraph_vertices[max_idx]

            # Защита от отсечения важных вершин
            if (best_vertex in direct_neighbors or
                    best_vertex == self.goal or
                    self.heuristic(best_vertex, self.goal) < self.heuristic(current, self.goal) * 0.5):
                return {
                    "decision": "explore",
                    "reason": f"important_vertex_{best_vertex}",
                    "should_explore": True
                }

            if max_prob < self.config.confidence_threshold:
                return {
                    "decision": "prune",
                    "reason": f"low_confidence_{max_prob:.3f}",
                    "should_explore": False
                }

        return {
            "decision": "explore",
            "reason": "default_conservative",
            "should_explore": should_explore
        }

    def _aggressive_strategy(self, current: int, f_values: List[float],
                             subgraph_vertices: List[int], threshold: float,
                             found: bool, probabilities: List[float],
                             direct_neighbors: List[int]) -> Dict:
        """Агрессивная стратегия: максимальное отсечение"""
        if not found:
            return {
                "decision": "prune",
                "reason": "grover_no_good_vertices",
                "should_explore": False
            }

        if probabilities:
            max_prob = max(probabilities)
            # Агрессивный порог уверенности
            if max_prob > 0.1:  # Низкий порог для агрессивной стратегии
                max_idx = probabilities.index(max_prob)
                best_vertex = subgraph_vertices[max_idx]

                # Всегда отсекаем, если лучшая вершина не является особо важной
                if (best_vertex not in direct_neighbors and
                        best_vertex != self.goal and
                        not self._is_critical_vertex(best_vertex)):
                    return {
                        "decision": "prune",
                        "reason": f"aggressive_prune_{max_prob:.3f}",
                        "should_explore": False
                    }

        return {
            "decision": "explore",
            "reason": "aggressive_default",
            "should_explore": True
        }

    def _adaptive_strategy(self, current: int, f_values: List[float],
                           subgraph_vertices: List[int], threshold: float,
                           found: bool, probabilities: List[float],
                           direct_neighbors: List[int]) -> Dict:
        """Адаптивная стратегия с обучением"""
        # Базовое решение на основе консервативной стратегии
        base_decision = self._conservative_strategy(
            current, f_values, subgraph_vertices, threshold,
            found, probabilities, direct_neighbors
        )

        # Адаптация параметров на основе истории
        if len(self.decision_history) > 5:
            self._adapt_parameters()

        return base_decision

    def _is_critical_vertex(self, vertex: int) -> bool:
        """Проверка, является ли вершина критически важной"""
        # Вершина критична, если у нее мало соседей или она связывает компоненты
        neighbors = self.graph.adjacency_list.get(vertex, [])
        return len(neighbors) <= 2

    def _adapt_parameters(self):
        """Адаптация параметров на основе истории решений"""
        recent_decisions = self.decision_history[-5:]

        # Анализ качества решений
        error_rate = self._calculate_error_rate(recent_decisions)

        if error_rate > 0.3:  # Высокая ошибка - становимся консервативнее
            self.config.quantum_threshold = min(0.95, self.config.quantum_threshold + 0.02)
            self.config.confidence_threshold = max(0.05, self.config.confidence_threshold - 0.01)
            logger.info(f"Adapting: increased threshold to {self.config.quantum_threshold:.3f}")
        elif error_rate < 0.1:  # Низкая ошибка - можно агрессивнее
            self.config.quantum_threshold = max(0.7, self.config.quantum_threshold - 0.01)
            self.config.confidence_threshold = min(0.3, self.config.confidence_threshold + 0.005)

    def _calculate_error_rate(self, decisions: List[Dict]) -> float:
        """Вычисление частоты ошибок в решениях"""
        if not decisions:
            return 0.0

        error_count = 0
        for decision in decisions:
            if decision.get("decision") == "prune":
                # Проверяем, была ли это ошибка(отсекли хорошую вершину)
                vertex = decision["vertex"]
                # Для простоты считаем ошибкой отсечение вершин с хорошей эвристикой
                if self.heuristic(vertex, self.goal) < 5.0:
                    error_count += 1

        return error_count / len(decisions)

    def find_path(self, start: int, goal: int) -> Tuple[List[int], float, PerformanceMetrics]:
        """Основной метод поиска пути"""
        self.goal = goal
        start_time = time.time()

        if self.config.strategy == SearchStrategy.PURE_CLASSICAL:
            path, cost, stats = self._classical_a_star(start, goal)
            self.metrics = stats
        else:
            path, cost, hybrid_stats = self._hybrid_a_star(start, goal)
            self.metrics.update(hybrid_stats)

        # Fallback механизм
        if (self.config.use_fallback and
                (not path or cost > self._estimate_optimal_cost(start, goal) * 1.5)):
            logger.warning("Using fallback to classical A*")
            fallback_path, fallback_cost, fallback_stats = self._classical_a_star(start, goal)

            if fallback_path and fallback_cost < cost * 0.95:  # Fallback лучше на 5%
                path, cost = fallback_path, fallback_cost
                self.metrics.fallback_used = True

        self.metrics.execution_time = time.time() - start_time
        self.metrics.classical_time = self.metrics.execution_time - self.metrics.quantum_time

        return path, cost, self.metrics

    def _classical_a_star(self, start: int, goal: int) -> Tuple[List[int], float, PerformanceMetrics]:
        """Классический алгоритм A* для сравнения и fallback"""
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        visited_nodes = 0

        while open_set:
            current_f, current = heapq.heappop(open_set)
            visited_nodes += 1

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, g_score[goal], PerformanceMetrics(
                    total_nodes_visited=visited_nodes
                )

            for neighbor, weight in self.graph.adjacency_list.get(current, []):
                tentative_g = g_score[current] + weight

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        return [], float('inf'), PerformanceMetrics(total_nodes_visited=visited_nodes)

    def _hybrid_a_star(self, start: int, goal: int) -> Tuple[List[int], float, Dict]:
        """Гибридный алгоритм A* с квантовым ускорением"""
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        visited_nodes = 0
        pruned_nodes = 0
        best_path_cost = float('inf')

        while open_set:
            current_f, current = heapq.heappop(open_set)

            # Пропускаем отсеченные вершины
            if current in came_from and came_from[current] is None:
                continue

            visited_nodes += 1

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                best_path_cost = g_score[goal]
                break

            # Определение порога F_current
            if best_path_cost < float('inf'):
                F_current = min(best_path_cost, current_f * 1.2)
            else:
                F_current = current_f * 1.2

            # Квантовый анализ подграфа
            should_explore = True
            if len(open_set) > 0:
                subgraph_vertices, f_values = self.collect_subgraph(
                    current, g_score, goal, depth=2
                )

                threshold = F_current * self.config.quantum_threshold
                should_explore, decision_info = self.should_explore_quantum(
                    current, f_values, subgraph_vertices, threshold
                )

            if not should_explore:
                came_from[current] = None
                pruned_nodes += 1
                continue

            # Классическое раскрытие вершина
            for neighbor, weight in self.graph.adjacency_list.get(current, []):
                tentative_g = g_score[current] + weight

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        stats = {
            "total_nodes_visited": visited_nodes,
            "pruned_nodes": pruned_nodes,
            "quantum_calls": self.metrics.quantum_calls
        }

        if 'path' not in locals() or not path:
            return [], float('inf'), stats

        return path, best_path_cost, stats

    def _estimate_optimal_cost(self, start: int, goal: int) -> float:
        """Оценка оптимальной стоимости пути"""
        return self.heuristic(start, goal)

    def get_detailed_report(self) -> Dict:
        """Получение детального отчета о работе алгоритма"""
        report = {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "decision_history_count": len(self.decision_history),
            "cache_hits": len([d for d in self.decision_history if d.get("cached", False)]),
            "cache_misses": len([d for d in self.decision_history if not d.get("cached", False)]),
            "pruning_efficiency": self.metrics.pruned_nodes / max(1, self.metrics.total_nodes_visited),
            "quantum_acceleration": self.metrics.quantum_calls / max(1, self.metrics.total_nodes_visited)
        }

        # Анализ решений
        if self.decision_history:
            decisions_by_type = {}
            for decision in self.decision_history:
                decision_type = decision.get("decision", "unknown")
                decisions_by_type[decision_type] = decisions_by_type.get(decision_type, 0) + 1

            report["decisions_by_type"] = decisions_by_type
            report["avg_quantum_time_per_call"] = self.metrics.quantum_time / max(1, self.metrics.quantum_calls)

        return report