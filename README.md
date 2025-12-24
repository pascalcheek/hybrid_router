# Hybrid Quantum-Classical Routing Algorithm
**Гибридный квантово-классический алгоритм маршрутизации**, объединяющий классический алгоритм A* с квантовым алгоритмом Гровера для эффективного поиска путей в графах.

## Содержание

- [Как это работает?](#-как-это-работает)
- [Установка](#-установка)
- [Быстрый старт](#-быстрый-старт)
- [Примеры использования](#-примеры-использования)
- [Запуск серверов](#-запуск-серверов)
- [Структура проекта](#-структура-проекта)

## Как это работает?

### Основная идея

Алгоритм решает классическую задачу поиска пути в графе, используя квантовое ускорение для анализа "тупиковых веток".

**Проблема классического A*:** При поиске пути алгоритм исследует все возможные направления, включая тупики.

**Наше решение:** Используем алгоритм Гровера для быстрого анализа подграфов и определения, какие ветки стоит исследовать, а какие можно безопасно игнорировать.

### Квантовый компонент: Алгоритм Гровера

Алгоритм Гровера обеспечивает **квадратичное ускорение** при поиске в неструктурированной базе данных:
Классический поиск: O(N) операций
Гровер: O(sqrt(N)) квантовых операций

**Как это применяется:**
1. При раскрытии вершины собираем окружающий подграф
2. Преобразуем информацию о вершинах в квантовые состояния
3. Запускаем Гровера для поиска "хороших" вершин (тех, что могут вести к цели)
4. На основе результата принимаем решение: исследовать ветку или отсечь

### Гибридный протокол
1. Классический A* выбирает вершину для исследования
2. Собирается локальный подграф (2-8 вершин)
3. Квантовый анализ: "Есть ли в этом подграфе вершины лучше текущей?"
4. Решение: 
   * Если да → продолжаем исследовать
   * Если нет → отсекаем эту ветку
5. Повторяем до достижения цели

```
Граф:
Start → A → B → Goal
  ↓     ↓
  C → D (тупик)

Гибридный алгоритм:
1. Start: исследуем A и C
2. Ветка C: квантовый анализ показывает "тупик"
3. Отсекаем C-D ветку
4. Продолжаем по A-B-Goal
5. Найден оптимальный путь!
```

## Установка

```bash
pip install git+https://github.com/pascalcheek/hybrid_router  
```
### Проверка установки

```python
import hybrid_router as hr
print(f"Версия: {hr.__version__}")
```

## Быстрый старт
### Пример 1: Самый простой поиск пути
```python
import hybrid_router as hr

# Создаем тестовый граф (встроенная функция)
graph = hr.create_maze_graph()

# Ищем путь от вершины 0 до 15
path, cost, metrics = hr.find_path(graph, start=0, goal=15)

print(f"Найден путь: {path}")
print(f"Стоимость пути: {cost:.2f}")
print(f"Посещено вершин: {metrics.total_nodes_visited}")
print(f"Отсечено квантовым поиском: {metrics.pruned_nodes}")
```

### Пример 2: Создание своего графа
```python
import hybrid_router as hr

# Определяем вершины и ребра
vertices = [
    {"id": 0, "x": 0, "y": 0},
    {"id": 1, "x": 1, "y": 0},
    {"id": 2, "x": 2, "y": 0},
    {"id": 3, "x": 0, "y": 1},
    {"id": 4, "x": 1, "y": 1},
]

edges = [
    {"from_vertex": 0, "to_vertex": 1, "weight": 1.0},
    {"from_vertex": 1, "to_vertex": 2, "weight": 1.0},
    {"from_vertex": 0, "to_vertex": 3, "weight": 1.5},
    {"from_vertex": 3, "to_vertex": 4, "weight": 1.0},
    {"from_vertex": 4, "to_vertex": 2, "weight": 1.2},
]

# Создаем граф
graph = hr.create_graph(vertices, edges)

# Ищем путь с настройками
path, cost, metrics = hr.find_path(
    graph, 
    start=0, 
    goal=2,
    strategy="hybrid_aggressive",
    quantum_threshold=0.85
)

print(f"Путь: {path}")
```

## Примеры использования
### Сравнение стратегий
```python
import hybrid_router as hr

graph = hr.create_complex_test_graph()

# Сравниваем разные подходы
results = hr.compare_strategies(
    graph,
    start=0,
    goal=14,
    strategies=[
        ("pure_classical", "Только A*"),
        ("hybrid_conservative", "Гибридный (осторожный)"),
        ("hybrid_aggressive", "Гибридный (агрессивный)")
    ]
)

print("Результаты сравнения:")
for strategy, data in results.items():
    print(f"\n{data['name']}:")
    print(f"  Время: {data['time']:.3f} сек")
    print(f"  Посещено вершин: {data['visited']}")
    print(f"  Отсечено: {data['pruned']}")
```

### Работа с квантовыми оракулами напрямую
```python
import hybrid_router as hr

# Создаем оракул для Qiskit симулятора
oracle = hr.QiskitOracle(shots=1024, optimization_level=1)

# Тестовые данные
f_values = [1.0, 2.0, 3.0, 4.0, 5.0]
threshold = 3.5

# Запускаем на Qiskit
found, probabilities = oracle.grover_search(f_values, threshold)

print(f"Qiskit результат: {found}")
print(f"Вероятности: {probabilities}")
```
## Запуск серверов
Система состоит из двух серверов:
* Главный сервер (порт 8000) - Классический A* и координация
* Квантовый сервис (порт 8001) - Алгоритм Гровера

### Запуск в двух терминалах
#### Терминал 1 - квантовый сервис:
```bash
python -m src.server.quantum_service --port 8001
```
#### Терминал 2 - главный сервер:
```bash
python -m src.server.main_server --port 8000 --quantum-url http://localhost:8001
```

### Использование через API клиент
```python
import asyncio
import hybrid_router as hr

async def demo_api():
    # Создаем клиент
    client = hr.APIClient()
    
    # Подключаемся
    await client.connect()
    
    # Проверяем статус
    status = await client.get_system_status()
    print(f"Статус системы: {status['overall_status']}")
    
    # Ищем путь через сервер
    result = await client.find_route(
        graph_data={
            "vertices": [{"id": i, "x": i, "y": 0} for i in range(5)],
            "edges": [
                {"from_vertex": i, "to_vertex": i+1, "weight": 1.0}
                for i in range(4)
            ]
        },
        start=0,
        goal=4
    )
    
    print(f"Результат: {result['path']}")
    
    # Закрываем соединение
    await client.close()

# Запускаем
asyncio.run(demo_api())
```

## Структура проекта
```
hybrid_router/
├── src/
│   ├── __init__.py              # Главный пакет
│   ├── architecture.py          # Основные классы и алгоритм
│   ├── classical/              # Классические компоненты
│   │   ├── a_star.py           # Алгоритм A*
│   │   └── graph_utils.py      # Утилиты для графов
│   ├── quantum/                # Квантовые компоненты
│   │   ├── matrix_oracle.py    # Матричная эмуляция Гровера
│   │   ├── qiskit_oracle.py    # Qiskit реализация
│   │   └── qiskit_hardware_oracle.py  # Реальное железо
│   ├── server/                 # Серверные компоненты
│   │   ├── main_server.py      # Главный сервер
│   │   └── quantum_service.py  # Квантовый сервис
│   └── client/                 # Клиентские компоненты
│       └── api_client.py       # API клиент
├── hybrid_router.py            # Основной модуль для импорта
├── cli.py                      # Командная строка
├── setup.py                    # Установка пакета
├── requirements.txt            # Зависимости
└── README.md                   # Эта документация
```
