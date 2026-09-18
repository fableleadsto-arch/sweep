"""Optional C++ acceleration build for SWEEP.

The production Python CLI must remain installable when a native compiler is not
available. Set SWEEP_BUILD_CPP=1 to compile the pybind11 acceleration module.
"""
from __future__ import annotations
import os
from setuptools import setup

build_cpp = os.getenv("SWEEP_BUILD_CPP", "0").strip().lower() in {"1", "true", "yes"}
ext_modules = []
cmdclass = {}

if build_cpp:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    ext_modules = [
        Pybind11Extension(
            "sweep_engine",
            sources=[
                "cpp/src/bindings.cpp",
                "cpp/src/html_parser.cpp",
                "cpp/src/text_extractor.cpp",
                "cpp/src/search_ranker.cpp",
                "cpp/src/regex_engine.cpp",
                "cpp/src/ml_engine.cpp",
                "cpp/src/data_engine.cpp",
                "cpp/src/nlp_engine.cpp",
            ],
            include_dirs=["cpp/include"],
            language="c++",
            cxx_std=20,
        ),
    ]
    cmdclass = {"build_ext": build_ext}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
