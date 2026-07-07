from pathlib import Path
from setuptools import find_packages, setup

# Metadatos
NAME = "model_desempleo"
DESCRIPTION = "Modelo de prediccion de desempleo (GEIH 2023-2025)."
URL = "https://github.com/Erick2626/Desempleo-GEIH"
EMAIL = "e.guerrerov@uniandes.edu.co"
AUTHOR = "Erick Guerrero - Nathalie Bareno"
REQUIRES_PYTHON = ">=3.9.0"

here = Path(__file__).resolve().parent

# Version desde el archivo VERSION
ROOT_DIR = here
PACKAGE_DIR = ROOT_DIR / "model"
about = {}
with open(PACKAGE_DIR / "VERSION" if (PACKAGE_DIR / "VERSION").exists() else ROOT_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

# Requisitos
def list_reqs(fname="requirements.txt"):
    if (here / fname).exists():
        with open(here / fname, encoding="utf-8") as fd:
            return fd.read().splitlines()
    return []

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"model": ["VERSION", "config.yml", "trained_models/*.pkl"]},
    install_requires=list_reqs(),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
