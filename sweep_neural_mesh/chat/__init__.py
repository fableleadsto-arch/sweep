"""Sweep chat package — conversational engine + CLI.

Layers (bottom-up):
  - commands: English -> shell command parsing + guarded execution
  - engine:   routing (shell / neural mesh / LLM chat) + conversation state
  - cli:      interactive terminal REPL (`python -m sweep_neural_mesh.chat`)
"""
from .engine import SweepChat, ChatResponse

__all__ = ["SweepChat", "ChatResponse"]
