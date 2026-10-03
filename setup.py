"""Optional C++ acceleration; normal installation is entirely Python.

To compile: install pybind11, then python setup.py build_ext --inplace.
Project metadata and dependencies live in pyproject.toml.
"""
import os
import sys
from setuptools import setup

options = {}
if "build_ext" in sys.argv or os.environ.get("SWEEP_BUILD_NATIVE") == "1":
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    options = {
        "ext_modules": [Pybind11Extension(
            "sweep_engine",
            sources=[f"cpp/src/{name}.cpp" for name in (
                "bindings", "html_parser", "text_extractor", "search_ranker",
                "regex_engine", "ml_engine", "data_engine", "nlp_engine",
            )],
            include_dirs=["cpp/include"], cxx_std=20,
        )],
        "cmdclass": {"build_ext": build_ext},
    }
setup(**options)
