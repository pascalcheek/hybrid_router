"""
Главный сервер для гибридного алгоритма маршрутизации
"""


import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
import aiohttp
from datetime import datetime

from ..architecture import (
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
    use_quantum_service: bool = True


class MainServer:
    """Главный сервер для координации работы системы"""

    def __init__(self, host: str = "localhost", port: int = 8000,
                 quantum_service_url: str = "http://localhost:8001"):
        self.host = host
        self.port = port
        self.quantum_service_url = quantum_service_url
        self.app = FastAPI(title="Hybrid Quantum Routing Main Server")
        self.session = None
        self.request_count = 0
        self.start_time = datetime.now()

        # Настройка CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

    def _setup_routes(self):
        """Настройка маршрутов API"""

        @self.app.on_event("startup")
        async def startup_event():
            """Событие запуска сервера"""
            self.session = aiohttp.ClientSession()
            logger.info("Main server started")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Событие остановки сервера"""
            if self.session:
                await self.session.close()
            logger.info("Main server stopped")

        @self.app.post("/api/v1/route")
        async def find_route(request: RoutingRequest):
            """Основной endpoint для маршрутизации"""
            self.request_count += 1
            request_id = self.request_count

            logger.info(f"Request #{request_id}: Routing from {request.start} to {request.goal}")

            try:
                # Проверка доступности квантового сервиса
                quantum_service_available = False
                if request.use_quantum_service:
                    quantum_service_available = await self._check_quantum_service()

                # Если квантовый сервис доступен, используем его
                if quantum_service_available:
                    logger.info(f"Request #{request_id}: Using quantum service")

                    async with self.session.post(
                            f"{self.quantum_service_url}/api/v1/route",
                            json=request.dict()
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            result["quantum_service_used"] = True
                            return result
                        else:
                            logger.warning(f"Request #{request_id}: Quantum service error, falling back to local")

                # Fallback: локальный расчет
                logger.info(f"Request #{request_id}: Using local calculation")
                result = await self._local_calculation(request)
                result["quantum_service_used"] = False

                return result

            except Exception as e:
                logger.error(f"Request #{request_id}: Error - {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/health")
        async def health_check():
            """Проверка здоровья сервера"""
            uptime = (datetime.now() - self.start_time).total_seconds()

            # Проверка квантового сервиса
            quantum_service_health = await self._check_quantum_service()

            return {
                "status": "healthy",
                "service": "main_server",
                "uptime_seconds": uptime,
                "total_requests": self.request_count,
                "quantum_service_available": quantum_service_health,
                "quantum_service_url": self.quantum_service_url,
                "timestamp": datetime.now().isoformat()
            }

        @self.app.get("/api/v1/system/status")
        async def system_status():
            """Статус всей системы"""
            main_uptime = (datetime.now() - self.start_time).total_seconds()

            # Получаем статус квантового сервиса
            quantum_status = {}
            try:
                async with self.session.get(f"{self.quantum_service_url}/api/v1/health") as response:
                    if response.status == 200:
                        quantum_status = await response.json()
            except:
                quantum_status = {"status": "unavailable"}

            return {
                "system": "hybrid_quantum_routing",
                "main_server": {
                    "status": "running",
                    "uptime": main_uptime,
                    "requests_handled": self.request_count
                },
                "quantum_service": quantum_status,
                "overall_status": "operational" if quantum_status.get("status") == "healthy" else "degraded"
            }

    async def _check_quantum_service(self) -> bool:
        """Проверка доступности квантового сервиса"""
        try:
            async with self.session.get(f"{self.quantum_service_url}/api/v1/health") as response:
                return response.status == 200
        except:
            return False

    async def _local_calculation(self, request: RoutingRequest) -> Dict:
        """Локальный расчет маршрута"""
        from ..architecture import HybridRoutingEngine, RouterConfig

        # Создание графа
        graph = self._create_graph_from_data(request.graph_data)

        # Конфигурация
        config = RouterConfig.from_dict(request.config) if request.config else RouterConfig()

        # Если квантовый сервис не используется, переключаемся на матричную эмуляцию
        if not request.use_quantum_service:
            config.backend = "matrix_emulation"

        # Создание движка
        engine = HybridRoutingEngine(config)
        engine.set_graph(graph)

        # Выполнение поиска
        path, cost, metrics = engine.find_path(request.start, request.goal)
        report = engine.get_detailed_report()

        return {
            "success": True,
            "path": path,
            "cost": cost,
            "metrics": metrics.to_dict(),
            "report": report,
            "calculated_locally": True
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

        logger.info(f"Starting Main Server on {self.host}:{self.port}")
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
    """Точка входа для запуска главного сервера"""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Quantum Routing Main Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--quantum-url", default="http://localhost:8001",
                        help="Quantum service URL")

    args = parser.parse_args()

    server = MainServer(
        host=args.host,
        port=args.port,
        quantum_service_url=args.quantum_url
    )

    print(f"Starting Main Server on http://{args.host}:{args.port}")
    print(f"Quantum Service: {args.quantum_url}")
    print("\nAvailable endpoints:")
    print(f"  POST /api/v1/route          - Поиск пути")
    print(f"  GET  /api/v1/health         - Проверка здоровья")
    print(f"  GET  /api/v1/system/status  - Статус системы")
    print("\nTo start both servers:")
    print("  Terminal 1: python -m src.server.quantum_service")
    print("  Terminal 2: python -m src.server.main_server")

    server.run()


if __name__ == "__main__":
    main()