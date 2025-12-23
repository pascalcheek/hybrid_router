from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hybrid_router",
    version="1.0.0",
    author="Khuri Pascal",
    author_email="paskal_khuri@mail.ru",
    description="Гибридный квантово-классический алгоритм маршрутизации",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/pascalcheeck/hybrid_router",

    packages=find_packages(),

    py_modules=["hybrid_router"],

    package_dir={
        '': '.',
        'src': 'src'
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "qiskit": [
            "qiskit>=0.34.0",
            "qiskit-aer>=0.10.0",
        ],
        "server": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybrid-router=src.cli:main",
            "quantum-service=src.server.quantum_service:main",
            "main-service=src.server.main_server:main",
        ],
    },
)