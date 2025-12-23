"""
Командная строка для гибридного алгоритма маршрутизации
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Главная функция командной строки"""
    parser = argparse.ArgumentParser(
        description="Гибридный квантово-классический алгоритм маршрутизации"
    )

    subparsers = parser.add_subparsers(dest="command", help="Команды")

    # Команда: Запуск сервера
    server_parser = subparsers.add_parser("server", help="Запуск сервера")
    server_parser.add_argument("--type", choices=["main", "quantum", "both"],
                               default="both", help="Тип сервера")
    server_parser.add_argument("--host", default="localhost", help="Хост")
    server_parser.add_argument("--main-port", type=int, default=8000,
                               help="Порт главного сервера")
    server_parser.add_argument("--quantum-port", type=int, default=8001,
                               help="Порт квантового сервиса")

    # Команда: Запуск тестов
    test_parser = subparsers.add_parser("test", help="Запуск тестов")
    test_parser.add_argument("--test-type", choices=["all", "quantum", "classical", "integration"],
                             default="all", help="Тип тестов")
    test_parser.add_argument("--verbose", "-v", action="store_true",
                             help="Подробный вывод")

    # Команда: Демонстрация
    demo_parser = subparsers.add_parser("demo", help="Запуск демонстрации")
    demo_parser.add_argument("--type", choices=["matrix", "qiskit", "server", "all"],
                             default="all", help="Тип демонстрации")

    # Команда: Поиск пути
    route_parser = subparsers.add_parser("route", help="Поиск пути")
    route_parser.add_argument("--graph", required=True, help="Файл с графом (JSON)")
    route_parser.add_argument("--start", type=int, required=True, help="Стартовая вершина")
    route_parser.add_argument("--goal", type=int, required=True, help="Целевая вершина")
    route_parser.add_argument("--backend", choices=["matrix", "qiskit", "qiskit-hardware"],
                              default="matrix", help="Квантовый бэкенд")
    route_parser.add_argument("--strategy",
                              choices=["classical", "conservative", "aggressive", "adaptive"],
                              default="conservative", help="Стратегия поиска")

    args = parser.parse_args()

    if args.command == "server":
        run_server(args)
    elif args.command == "test":
        run_tests(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "route":
        run_route(args)
    else:
        parser.print_help()


def run_server(args):
    """Запуск сервера"""
    if args.type in ["main", "both"]:
        print(f"Запуск главного сервера на {args.host}:{args.main_port}")
        # Здесь будет код запуска главного сервера

    if args.type in ["quantum", "both"]:
        print(f"Запуск квантового сервиса на {args.host}:{args.quantum_port}")
        # Здесь будет код запуска квантового сервиса


def run_tests(args):
    """Запуск тестов"""
    import pytest

    test_files = []
    if args.test_type in ["all", "quantum"]:
        test_files.append("tests/test_quantum.py")
    if args.test_type in ["all", "classical"]:
        test_files.append("tests/test_classical.py")
    if args.test_type in ["all", "integration"]:
        test_files.append("tests/test_integration.py")

    pytest_args = test_files
    if args.verbose:
        pytest_args.append("-v")

    pytest.main(pytest_args)


def run_demo(args):
    """Запуск демонстрации"""
    if args.type in ["matrix", "all"]:
        print("Запуск демонстрации матричной эмуляции...")
        import examples.demo_matrix
        examples.demo_matrix.demo_matrix_emulation()

    if args.type in ["qiskit", "all"]:
        print("\n" + "=" * 60)
        print("Запуск демонстрации Qiskit...")
        import examples.demo_qiskit
        examples.demo_qiskit.demo_qiskit_implementation()

    if args.type in ["server", "all"]:
        print("\n" + "=" * 60)
        print("Запуск демонстрации серверной архитектуры...")
        import examples.demo_server
        asyncio.run(examples.demo_server.demo_server_architecture())


def run_route(args):
    """Поиск пути"""
    import json
    from src.architecture import (
        HybridRoutingEngine,
        RouterConfig,
        QuantumBackend,
        SearchStrategy,
        Graph,
        Vertex,
        Edge
    )

    with open(args.graph, 'r') as f:
        graph_data = json.load(f)

    vertices = [Vertex(**v) for v in graph_data["vertices"]]
    edges = [Edge(**e) for e in graph_data["edges"]]
    graph = Graph(vertices, edges)

    backend_map = {
        "matrix": QuantumBackend.MATRIX_EMULATION,
        "qiskit": QuantumBackend.QISKIT_SIMULATOR,
        "qiskit-hardware": QuantumBackend.QISKIT_HARDWARE
    }

    strategy_map = {
        "classical": SearchStrategy.PURE_CLASSICAL,
        "conservative": SearchStrategy.HYBRID_CONSERVATIVE,
        "aggressive": SearchStrategy.HYBRID_AGGRESSIVE,
        "adaptive": SearchStrategy.ADAPTIVE
    }

    config = RouterConfig(
        backend=backend_map[args.backend],
        strategy=strategy_map[args.strategy],
        quantum_threshold=0.9,
        confidence_threshold=0.15
    )

    engine = HybridRoutingEngine(config)
    engine.set_graph(graph)

    print(f"Поиск пути от {args.start} до {args.goal}")
    print(f"Бэкенд: {args.backend}, Стратегия: {args.strategy}")
    print("-" * 50)

    import time
    start_time = time.time()
    path, cost, metrics = engine.find_path(args.start, args.goal)
    exec_time = time.time() - start_time

    if path:
        print(f"Путь найден за {exec_time:.4f} сек")
        print(f"Путь: {path}")
        print(f"Стоимость: {cost:.3f}")
        print(f"Длина пути: {len(path)} вершин")
        print(f"Посещено вершин: {metrics.total_nodes_visited}")
        print(f"Отсечено вершин: {metrics.pruned_nodes}")
        print(f"Квантовых запросов: {metrics.quantum_calls}")
        print(f"Общее время: {metrics.execution_time:.4f} сек")
        print(f"Время на квант: {metrics.quantum_time:.4f} сек")
    else:
        print(f"Путь не найден")
        print(f"Время выполнения: {exec_time:.4f} сек")


if __name__ == "__main__":
    main()