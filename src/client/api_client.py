"""
Клиентская библиотека для взаимодействия с сервисами
"""

import aiohttp
from typing import Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class APIClient:
    """Клиент для взаимодействия с сервисами гибридной маршрутизации"""

    def __init__(self,
                 main_server_url: str = "http://localhost:8000",
                 quantum_service_url: str = "http://localhost:8001"):
        self.main_server_url = main_server_url
        self.quantum_service_url = quantum_service_url
        self.session = None
        self.request_id = 0

    async def connect(self):
        """Подключение к сервисам"""
        self.session = aiohttp.ClientSession()

        # Проверка доступности сервисов
        main_available = await self._check_health(self.main_server_url)
        quantum_available = await self._check_health(self.quantum_service_url)

        if main_available:
            logger.info(f"✓ Main server is available: {self.main_server_url}")
        else:
            logger.warning(f"✗ Main server is not available: {self.main_server_url}")

        if quantum_available:
            logger.info(f"✓ Quantum service is available: {self.quantum_service_url}")
        else:
            logger.warning(f"✗ Quantum service is not available: {self.quantum_service_url}")

        return main_available

    async def _check_health(self, url: str) -> bool:
        """Проверка здоровья сервиса"""
        try:
            async with self.session.get(f"{url}/api/v1/health") as response:
                return response.status == 200
        except:
            return False

    async def find_route(self, graph_data: Dict, start: int, goal: int,
                         config: Optional[Dict] = None,
                         use_quantum_service: bool = True) -> Dict:
        """Поиск пути через главный сервер"""
        self.request_id += 1
        request_id = self.request_id

        logger.info(f"Request #{request_id}: Finding route from {start} to {goal}")

        request_data = {
            "graph_data": graph_data,
            "start": start,
            "goal": goal,
            "config": config,
            "use_quantum_service": use_quantum_service
        }

        try:
            async with self.session.post(
                    f"{self.main_server_url}/api/v1/route",
                    json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    result["request_id"] = request_id

                    if result.get("success"):
                        logger.info(f"Request #{request_id}: Route found with cost {result.get('cost', 0):.2f}")
                    else:
                        logger.warning(f"Request #{request_id}: No route found")

                    return result
                else:
                    error = await response.text()
                    logger.error(f"Request #{request_id}: Server error - {error}")
                    raise Exception(f"Server error: {error}")

        except Exception as e:
            logger.error(f"Request #{request_id}: Connection error - {str(e)}")
            return await self._local_fallback(graph_data, start, goal, config, request_id)

    async def execute_grover(self, f_values: List[float], threshold: float,
                             iterations: Optional[int] = None) -> Dict:
        """Выполнение алгоритма Гровера через квантовый сервис"""
        self.request_id += 1
        request_id = self.request_id

        logger.info(f"Request #{request_id}: Grover search with {len(f_values)} values")

        request_data = {
            "f_values": f_values,
            "threshold": threshold,
            "iterations": iterations
        }

        try:
            async with self.session.post(
                    f"{self.quantum_service_url}/api/v1/grover",
                    json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    result["request_id"] = request_id
                    logger.info(f"Request #{request_id}: Grover search completed")
                    return result
                else:
                    error = await response.text()
                    logger.error(f"Request #{request_id}: Grover service error - {error}")

                    # Fallback на локальный расчет
                    return await self._local_grover_fallback(f_values, threshold, iterations, request_id)

        except Exception as e:
            logger.error(f"Request #{request_id}: Connection error - {str(e)}")
            return await self._local_grover_fallback(f_values, threshold, iterations, request_id)

    async def get_system_status(self) -> Dict:
        """Получение статуса системы"""
        try:
            async with self.session.get(
                    f"{self.main_server_url}/api/v1/system/status"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unknown", "error": "Cannot get system status"}
        except:
            return {"status": "unavailable", "error": "Connection failed"}

    async def _local_fallback(self, graph_data: Dict, start: int, goal: int,
                              config: Optional[Dict], request_id: int) -> Dict:
        """Локальный fallback расчет"""
        logger.info(f"Request #{request_id}: Using local fallback calculation")

        from ..architecture import (
            HybridRoutingEngine,
            RouterConfig,
            Graph,
            Vertex,
            Edge
        )

        try:
            # Создание графа
            vertices = [Vertex(**v) for v in graph_data.get("vertices", [])]
            edges = [Edge(**e) for e in graph_data.get("edges", [])]
            graph = Graph(vertices, edges)

            # Конфигурация
            config_obj = RouterConfig.from_dict(config) if config else RouterConfig()

            # Создание движка
            engine = HybridRoutingEngine(config_obj)
            engine.set_graph(graph)

            # Выполнение поиска
            path, cost, metrics = engine.find_path(start, goal)
            report = engine.get_detailed_report()

            return {
                "success": True,
                "request_id": request_id,
                "path": path,
                "cost": cost,
                "metrics": metrics.to_dict(),
                "report": report,
                "fallback": True,
                "message": "Used local fallback calculation"
            }

        except Exception as e:
            logger.error(f"Request #{request_id}: Local fallback error - {str(e)}")
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "fallback": True,
                "message": "Local calculation failed"
            }

    async def _local_grover_fallback(self, f_values: List[float], threshold: float,
                                     iterations: Optional[int], request_id: int) -> Dict:
        """Локальный fallback для алгоритма Гровера"""
        logger.info(f"Request #{request_id}: Using local Grover fallback")

        from ..quantum.matrix_oracle import MatrixOracle

        try:
            oracle = MatrixOracle()
            found, probabilities = oracle.grover_search(f_values, threshold, iterations)

            return {
                "success": True,
                "request_id": request_id,
                "found": found,
                "probabilities": probabilities,
                "quantum_time": oracle.quantum_time,
                "fallback": True,
                "message": "Used local Grover calculation"
            }

        except Exception as e:
            logger.error(f"Request #{request_id}: Local Grover error - {str(e)}")
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "fallback": True,
                "message": "Local Grover calculation failed"
            }

    async def close(self):
        """Закрытие соединений"""
        if self.session:
            await self.session.close()
        logger.info("Client connections closed")