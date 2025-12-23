"""
Сервер для выполнения квантовых вычислений
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..architecture import (
    HybridRoutingEngine,
    RouterConfig,
    QuantumBackend,
    Graph,
    Vertex,
    Edge
)

logger = logging.getLogger(__name__)


class RoutingRequest(BaseModel):
    """Модель запроса для маршрутизации"""
    graph_data: Dict
    start: int
    goal: int
    config: Optional[Dict] = None


class GroverRequest(BaseModel):
    """Модель запроса для алгоритма Гровера"""
    f_values: List[float]
    threshold: float
    iterations: Optional[int] = None


class QuantumService:
    """Сервис для выполнения квантовых вычислений"""

    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Quantum Routing Service")
        self.engines = {}
        self.request_count = 0
        self.start_time = datetime.now()

        self._setup_routes()

    def _setup_routes(self):
        """Настройка маршрутов API"""

        @self.app.post("/api/v1/route")
        async def find_route(request: RoutingRequest):
            """Основной endpoint для маршрутизации"""
            self.request_count += 1
            request_id = self.request_count

            logger.info(f"Request #{request_id}: Routing from {request.start} to {request.goal}")

            try:
                # Создание графа
                graph = self._create_graph_from_data(request.graph_data)

                # Конфигурация
                config = RouterConfig.from_dict(request.config) if request.config else RouterConfig()

                # Создание или получение движка из кэша
                engine_key = f"{request.start}_{request.goal}_{hash(str(request.config))}"
                if engine_key not in self.engines:
                    self.engines[engine_key] = HybridRoutingEngine(config)
                    self.engines[engine_key].set_graph(graph)

                engine = self.engines[engine_key]

                # Выполнение поиска
                path, cost, metrics = engine.find_path(request.start, request.goal)
                report = engine.get_detailed_report()

                logger.info(f"Request #{request_id}: Path found with cost {cost}")

                return {
                    "success": True,
                    "request_id": request_id,
                    "path": path,
                    "cost": cost,
                    "metrics": metrics.to_dict(),
                    "report": report
                }

            except Exception as e:
                logger.error(f"Request #{request_id}: Error - {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/grover")
        async def execute_grover(request: GroverRequest):
            """Endpoint для выполнения алгоритма Гровера"""
            self.request_count += 1
            request_id = self.request_count

            logger.info(f"Request #{request_id}: Grover search with {len(request.f_values)} values")

            try:
                # Создаем оракул с матричной эмуляцией
                from ..quantum.matrix_oracle import MatrixOracle
                oracle = MatrixOracle()

                # Выполняем поиск Гровера
                found, probabilities = oracle.grover_search(
                    request.f_values,
                    request.threshold,
                    request.iterations
                )

                return {
                    "success": True,
                    "request_id": request_id,
                    "found": found,
                    "probabilities": probabilities,
                    "quantum_time": oracle.quantum_time
                }

            except Exception as e:
                logger.error(f"Request #{request_id}: Error - {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/health")
        async def health_check():
            """Проверка здоровья сервиса"""
            uptime = (datetime.now() - self.start_time).total_seconds()

            return {
                "status": "healthy",
                "service": "quantum_service",
                "uptime_seconds": uptime,
                "engines_cached": len(self.engines),
                "total_requests": self.request_count,
                "timestamp": datetime.now().isoformat()
            }

        @self.app.get("/api/v1/metrics")
        async def get_metrics():
            """Метрики сервиса"""
            total_quantum_time = sum(e.metrics.quantum_time for e in self.engines.values())
            total_calls = sum(e.metrics.quantum_calls for e in self.engines.values())

            return {
                "total_engines": len(self.engines),
                "total_requests": self.request_count,
                "total_quantum_time": total_quantum_time,
                "total_quantum_calls": total_calls,
                "avg_quantum_time_per_call": total_quantum_time / max(1, total_calls),
                "active_since": self.start_time.isoformat()
            }

        @self.app.get("/api/v1/backends")
        async def get_available_backends():
            """Доступные квантовые бэкенды"""
            backends = [
                {
                    "name": "matrix_emulation",
                    "description": "Матричная эмуляция на NumPy",
                    "requires_qiskit": False
                },
                {
                    "name": "qiskit_simulator",
                    "description": "Симулятор Qiskit Aer",
                    "requires_qiskit": True
                },
                {
                    "name": "qiskit_hardware",
                    "description": "Реальный квантовый компьютер IBM",
                    "requires_qiskit": True,
                    "requires_ibmq_account": True
                }
            ]

            return {
                "backends": backends,
                "default": "matrix_emulation"
            }

    def _create_graph_from_data(self, graph_data: Dict) -> Graph:
        """Создание графа из данных"""
        vertices = [Vertex(**v) for v in graph_data.get("vertices", [])]
        edges = [Edge(**e) for e in graph_data.get("edges", [])]
        return Graph(vertices, edges)

    async def start(self):
        """Запуск сервера"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        logger.info(f"Starting Quantum Service on {self.host}:{self.port}")
        await server.serve()

    def run(self):
        """Запуск сервера (синхронный)"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


def main():
    """Точка входа для запуска сервера"""
    import argparse

    parser = argparse.ArgumentParser(description="Quantum Routing Service")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")

    args = parser.parse_args()

    service = QuantumService(host=args.host, port=args.port)

    print(f"Starting Quantum Service on http://{args.host}:{args.port}")
    print("Available endpoints:")
    print(f"  POST /api/v1/route      - Поиск пути")
    print(f"  POST /api/v1/grover     - Алгоритм Гровера")
    print(f"  GET  /api/v1/health     - Проверка здоровья")
    print(f"  GET  /api/v1/metrics    - Метрики сервиса")
    print(f"  GET  /api/v1/backends   - Доступные бэкенды")

    service.run()


if __name__ == "__main__":
    main()