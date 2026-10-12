"""Preserve the sibling DLL directory used by PyAV's Windows wheel."""
from PyInstaller.utils.hooks import collect_delvewheel_libs_directory

datas, binaries = collect_delvewheel_libs_directory("av")
