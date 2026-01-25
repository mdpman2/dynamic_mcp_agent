# -*- coding: utf-8 -*-
"""Dynamic MCP Agent Package"""

from .agent import DynamicMCPAgent, create_agent
from .lib import (
    ToolRegistry,
    registry,
    search_available_tools,
    load_tool,
    initialize_mcp_tools,
    register_tool
)

__version__ = "1.0.0"
__all__ = [
    "DynamicMCPAgent",
    "create_agent",
    "ToolRegistry",
    "registry",
    "search_available_tools",
    "load_tool",
    "initialize_mcp_tools",
    "register_tool"
]
