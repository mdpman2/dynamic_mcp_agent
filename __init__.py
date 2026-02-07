# -*- coding: utf-8 -*-
"""
Dynamic MCP Agent Package (v1 Responses API)

Changelog:
  v2.0.0 (2026-02-07)
    - Azure OpenAI v1 API (Next Generation) 전환
    - Responses API 도입 (chat.completions → responses.create)
    - previous_response_id 기반 자동 대화 체이닝 (수동 conversation_history 제거)
    - 네이티브 원격 MCP 서버 도구 통합 지원
    - GPT-5 시리즈 모델 기본값 변경 (gpt-4o → gpt-5)
    - text-embedding-3-large 임베딩 모델 (3072 차원)
    - 스트리밍 응답 모드 추가 (--stream)
    - 5개 신규 도구 추가 (AI Foundry Agent, Deep Research, Web Search, Code Interpreter, Image Generation)
    - 도구 총 20개로 확장 (15 → 20)
    - Gradio 5.x 지원
  v1.0.0 (2025)
    - 초기 릴리스: BM25 + Sentence-Transformers + MCP Registry + GPT-4.1 하이브리드 검색
    - Azure OpenAI Chat Completions API 기반
    - 15개 Azure MCP 도구
"""

from .agent import DynamicMCPAgent, create_agent, DEFAULT_REMOTE_MCP_SERVERS
from .lib import (
    ToolRegistry,
    registry,
    search_available_tools,
    load_tool,
    initialize_mcp_tools,
    register_tool
)

__version__ = "2.0.0"
__all__ = [
    "DynamicMCPAgent",
    "create_agent",
    "DEFAULT_REMOTE_MCP_SERVERS",
    "ToolRegistry",
    "registry",
    "search_available_tools",
    "load_tool",
    "initialize_mcp_tools",
    "register_tool"
]
