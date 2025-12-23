"""
Классическая реализация алгоритма A*
"""

import heapq
import math
from typing import List, Tuple
from ..architecture import Graph

class ClassicalAStar:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.heuristic_cache = {}

    def heuristic(self, a: int, b: int) -> float:
        """Эвристическая функция - евклидово расстояние между вершинами"""
        if (a, b) not in self.heuristic_cache:
            v1 = self.graph.vertices[a]
            v2 = self.graph.vertices[b]
            self.heuristic_cache[(a, b)] = math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)
        return self.heuristic_cache[(a, b)]

    def find_path(self, start: int, goal: int) -> Tuple[List[int], float, int]:
        """Классический алгоритм A* без квантового ускорения"""
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
                return path, g_score[goal], visited_nodes

            for neighbor, weight in self.graph.adjacency_list.get(current, []):
                tentative_g = g_score[current] + weight

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        return [], float('inf'), visited_nodes