# -*- coding: utf-8 -*-
"""
Dynamic MCP Agent Package (v1 Responses API + Agents SDK)

Changelog:
  v3.0.0 (2026-02-26)
    - [NEW] OpenAI Agents SDK 통합 - 멀티 에이전트 오케스트레이션 (--agents 모드)
    - [NEW] Structured Outputs 지원 (Pydantic v2 스키마 기반 응답)
    - [NEW] GPT-5.2 네이티브 추론 지원 (--reasoning 모드, 별도 추론 모델 불필요)
    - [NEW] Reciprocal Rank Fusion (RRF) 하이브리드 검색 알고리즘
    - [NEW] 5개 신규 도구 추가 (Azure AI Agent Service, Computer Use, MCP Discovery, Structured Output, Realtime Audio)
    - [NEW] httpx 기반 비동기 HTTP (aiohttp 대체, Streamable HTTP 지원)
    - [NEW] 에이전트 트레이싱/관찰성 지원
    - [CHANGED] 기본 모델 gpt-5 → gpt-5.2
    - [CHANGED] openai SDK 최소 버전 1.86 → 1.93
    - [CHANGED] sentence-transformers 3.4+ / torch 2.5+ 업데이트
    - [CHANGED] 도구 총 25개로 확장 (20 → 25)
    - [CHANGED] Gradio 5.12+ 지원
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

__version__ = "3.0.0"
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
