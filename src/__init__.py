"""
Гибридный квантово-классический алгоритм маршрутизации
"""

__version__ = "1.0.0"
__author__ = "Ваше имя"
__email__ = "ваш.email@example.com"

from .architecture import (
    HybridRoutingEngine,
    RouterConfig,
    QuantumBackend,
    SearchStrategy,
    PerformanceMetrics,
    Vertex,
    Edge,
    Graph,
)

from .quantum import (
    MatrixOracle,
    QiskitOracle,
    QiskitHardwareOracle,
)

from .classical import (
    ClassicalAStar,
    create_test_graphs,
)

from .server import (
    QuantumService,
    MainServer,
)

from .client import (
    APIClient,
)

__all__ = [
    # Архитектура
    "HybridRoutingEngine",
    "RouterConfig",
    "QuantumBackend",
    "SearchStrategy",
    "PerformanceMetrics",
    "Vertex",
    "Edge",
    "Graph",

    # Квантовые компоненты
    "MatrixOracle",
    "QiskitOracle",
    "QiskitHardwareOracle",

    # Классические компоненты
    "ClassicalAStar",
    "create_test_graphs",

    # Серверная часть
    "QuantumService",
    "MainServer",

    # Клиент
    "APIClient",
]