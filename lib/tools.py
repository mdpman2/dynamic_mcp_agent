# -*- coding: utf-8 -*-
"""
Dynamic Tool Loading Functions for Azure-based MCP Agent (v1 Responses API + Agents SDK)

이 모듈은 다양한 MCP 도구들을 동적으로 로드하고 관리하는 기능을 제공합니다.
Azure OpenAI, Azure AI Search, Azure Functions 등 다양한 Azure 서비스와 통합됩니다.

v3.0.0 업데이트 (2026-02-26):
- [NEW] azure_ai_agent_service_tool: Azure AI Agent Service 서버리스 에이전트 호출
- [NEW] azure_computer_use_tool: CUA(Computer Use Agent) 기반 GUI 자동화
- [NEW] mcp_server_discovery_tool: MCP Registry에서 서버 검색/연결
- [NEW] structured_output_tool: Pydantic 스키마 기반 구조화된 응답 생성
- [NEW] azure_realtime_audio_tool: GPT-4o-realtime 기반 실시간 음성 대화
- [CHANGED] azure_image_generation_tool 기본 모델 gpt-image-1.5 → gpt-image-2
- [CHANGED] TOOL_DEFINITIONS 20개 → 25개로 확장

v2.0.0 업데이트 (2026-02-07):
- [NEW] azure_ai_foundry_agent_tool: Azure AI Foundry Agent 멀티스텝 작업
- [NEW] azure_deep_research_tool: o3-deep-research 기반 심층 조사
- [NEW] azure_web_search_tool: Responses API 내장 웹 검색 (web_search_preview)
- [NEW] azure_code_interpreter_tool: Responses API 코드 인터프리터
- [NEW] azure_image_generation_tool: GPT-Image 모델 이미지 생성
- [CHANGED] azure_openai_embedding_tool 기본 모델 text-embedding-3-large (3072차원)
- [CHANGED] TOOL_DEFINITIONS 15개 → 20개로 확장

참고:
- https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses
- https://openai.github.io/openai-agents-python/
- https://medium.com/google-cloud/implementing-anthropic-style-dynamic-tool-search-tool-f39d02a35139
"""

import os
import math
import operator
import ast
import logging
from typing import List, Dict, Any, Optional, Callable

from .registry import registry

logger = logging.getLogger(__name__)


# ============================================================================
# calculator_tool에서 사용하는 안전한 연산자/함수 (모듈 레벨 상수)
# ============================================================================

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    'abs': abs, 'round': round, 'min': min, 'max': max,
    'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'pi': math.pi, 'e': math.e,
}


def _safe_eval(node):
    """AST 노드를 안전하게 평가합니다."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"허용되지 않는 값: {node.value}")
    elif isinstance(node, ast.BinOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"허용되지 않는 연산자: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"허용되지 않는 연산자: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            args = [_safe_eval(arg) for arg in node.args]
            return _SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError(f"허용되지 않는 함수: {getattr(node.func, 'id', '?')}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            val = _SAFE_FUNCTIONS[node.id]
            if isinstance(val, (int, float)):
                return val
        raise ValueError(f"허용되지 않는 이름: {node.id}")
    raise ValueError(f"허용되지 않는 표현식: {type(node).__name__}")


# ============================================================================
# 샘플 MCP 도구 정의 (실제 환경에서는 외부 MCP 서버에서 가져옴)
# ============================================================================

def azure_ai_search_tool(
    query: str,
    index_name: str = "default-index",
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Azure AI Search를 사용하여 문서를 검색합니다.

    Args:
        query: 검색 쿼리
        index_name: 검색할 인덱스 이름
        top_k: 반환할 최대 결과 수

    Returns:
        검색 결과를 포함하는 딕셔너리
    """
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_KEY")

    if not search_endpoint or not search_key:
        return {"error": "Azure AI Search 자격 증명이 설정되지 않았습니다."}

    # 실제 구현에서는 Azure Search SDK를 사용
    return {
        "query": query,
        "index_name": index_name,
        "top_k": top_k,
        "results": [],
        "message": "Azure AI Search 도구가 실행되었습니다."
    }


def azure_blob_storage_tool(
    operation: str,
    container_name: str,
    blob_name: Optional[str] = None,
    content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Azure Blob Storage에서 파일을 관리합니다.

    Args:
        operation: 수행할 작업 (list, read, write, delete)
        container_name: 컨테이너 이름
        blob_name: Blob 파일 이름
        content: 쓰기 작업 시 저장할 내용

    Returns:
        작업 결과를 포함하는 딕셔너리
    """
    return {
        "operation": operation,
        "container_name": container_name,
        "blob_name": blob_name,
        "status": "success",
        "message": "Azure Blob Storage 도구가 실행되었습니다."
    }


def azure_sql_query_tool(
    query: str,
    database: str = "default",
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Azure SQL Database에서 쿼리를 실행합니다.

    Args:
        query: 실행할 SQL 쿼리
        database: 데이터베이스 이름
        parameters: 쿼리 매개변수

    Returns:
        쿼리 결과를 포함하는 딕셔너리
    """
    return {
        "query": query,
        "database": database,
        "parameters": parameters,
        "results": [],
        "message": "Azure SQL 도구가 실행되었습니다."
    }


def azure_cosmos_db_tool(
    operation: str,
    database_name: str,
    container_name: str,
    query: Optional[str] = None,
    document: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Azure Cosmos DB에서 데이터를 관리합니다.

    Args:
        operation: 수행할 작업 (query, insert, update, delete)
        database_name: 데이터베이스 이름
        container_name: 컨테이너 이름
        query: 쿼리 문자열 (query 작업 시)
        document: 문서 데이터 (insert, update 작업 시)

    Returns:
        작업 결과를 포함하는 딕셔너리
    """
    return {
        "operation": operation,
        "database_name": database_name,
        "container_name": container_name,
        "status": "success",
        "message": "Azure Cosmos DB 도구가 실행되었습니다."
    }


def azure_openai_embedding_tool(
    text: str,
    model: str = "text-embedding-3-large"
) -> Dict[str, Any]:
    """
    Azure OpenAI를 사용하여 텍스트 임베딩을 생성합니다.

    Args:
        text: 임베딩할 텍스트
        model: 사용할 임베딩 모델 (text-embedding-3-small, text-embedding-3-large)

    Returns:
        임베딩 벡터를 포함하는 딕셔너리
    """
    return {
        "text": text[:50] + "..." if len(text) > 50 else text,
        "model": model,
        "embedding_dimension": 3072 if "large" in model else 1536,
        "message": "Azure OpenAI Embedding 도구가 실행되었습니다."
    }


def azure_computer_vision_tool(
    image_url: str,
    operation: str = "analyze"
) -> Dict[str, Any]:
    """
    Azure Computer Vision을 사용하여 이미지를 분석합니다.

    Args:
        image_url: 분석할 이미지 URL
        operation: 수행할 작업 (analyze, ocr, describe, tag)

    Returns:
        분석 결과를 포함하는 딕셔너리
    """
    return {
        "image_url": image_url,
        "operation": operation,
        "results": {},
        "message": "Azure Computer Vision 도구가 실행되었습니다."
    }


def azure_translator_tool(
    text: str,
    target_language: str,
    source_language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Azure Translator를 사용하여 텍스트를 번역합니다.

    Args:
        text: 번역할 텍스트
        target_language: 대상 언어 코드
        source_language: 원본 언어 코드 (None이면 자동 감지)

    Returns:
        번역 결과를 포함하는 딕셔너리
    """
    return {
        "original_text": text,
        "target_language": target_language,
        "source_language": source_language or "auto-detected",
        "translated_text": f"[번역됨: {text}]",
        "message": "Azure Translator 도구가 실행되었습니다."
    }


def azure_text_analytics_tool(
    text: str,
    operation: str = "sentiment"
) -> Dict[str, Any]:
    """
    Azure Text Analytics를 사용하여 텍스트를 분석합니다.

    Args:
        text: 분석할 텍스트
        operation: 수행할 작업 (sentiment, entities, key_phrases, language)

    Returns:
        분석 결과를 포함하는 딕셔너리
    """
    return {
        "text": text[:50] + "..." if len(text) > 50 else text,
        "operation": operation,
        "results": {},
        "message": "Azure Text Analytics 도구가 실행되었습니다."
    }


def azure_form_recognizer_tool(
    document_url: str,
    model_id: str = "prebuilt-document"
) -> Dict[str, Any]:
    """
    Azure Form Recognizer를 사용하여 문서에서 데이터를 추출합니다.

    Args:
        document_url: 분석할 문서 URL
        model_id: 사용할 모델 ID

    Returns:
        추출된 데이터를 포함하는 딕셔너리
    """
    return {
        "document_url": document_url,
        "model_id": model_id,
        "extracted_data": {},
        "message": "Azure Form Recognizer 도구가 실행되었습니다."
    }


def azure_speech_to_text_tool(
    audio_url: str,
    language: str = "ko-KR"
) -> Dict[str, Any]:
    """
    Azure Speech Services를 사용하여 음성을 텍스트로 변환합니다.

    Args:
        audio_url: 변환할 오디오 파일 URL
        language: 음성 언어 코드

    Returns:
        변환된 텍스트를 포함하는 딕셔너리
    """
    return {
        "audio_url": audio_url,
        "language": language,
        "transcription": "",
        "message": "Azure Speech-to-Text 도구가 실행되었습니다."
    }


def azure_function_invoke_tool(
    function_url: str,
    payload: Dict[str, Any],
    method: str = "POST"
) -> Dict[str, Any]:
    """
    Azure Function을 호출합니다.

    Args:
        function_url: 호출할 Azure Function URL
        payload: 요청 페이로드
        method: HTTP 메서드

    Returns:
        함수 응답을 포함하는 딕셔너리
    """
    return {
        "function_url": function_url,
        "method": method,
        "payload": payload,
        "response": {},
        "message": "Azure Function 도구가 실행되었습니다."
    }


def bing_web_search_tool(
    query: str,
    count: int = 10,
    market: str = "ko-KR"
) -> Dict[str, Any]:
    """
    Bing Web Search API를 사용하여 웹 검색을 수행합니다.

    Args:
        query: 검색 쿼리
        count: 반환할 결과 수
        market: 검색 시장 (언어/지역)

    Returns:
        검색 결과를 포함하는 딕셔너리
    """
    return {
        "query": query,
        "count": count,
        "market": market,
        "results": [],
        "message": "Bing Web Search 도구가 실행되었습니다."
    }


def github_search_tool(
    query: str,
    search_type: str = "repositories",
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    GitHub에서 코드, 저장소, 이슈를 검색합니다.

    Args:
        query: 검색 쿼리
        search_type: 검색 유형 (repositories, code, issues)
        language: 프로그래밍 언어 필터

    Returns:
        검색 결과를 포함하는 딕셔너리
    """
    return {
        "query": query,
        "search_type": search_type,
        "language": language,
        "results": [],
        "message": "GitHub Search 도구가 실행되었습니다."
    }


def weather_api_tool(
    location: str,
    units: str = "metric"
) -> Dict[str, Any]:
    """
    날씨 정보를 조회합니다.

    Args:
        location: 위치 (도시명 또는 좌표)
        units: 온도 단위 (metric, imperial)

    Returns:
        날씨 정보를 포함하는 딕셔너리
    """
    return {
        "location": location,
        "units": units,
        "weather": {},
        "message": "Weather API 도구가 실행되었습니다."
    }


def calculator_tool(
    expression: str
) -> Dict[str, Any]:
    """
    수학 계산을 수행합니다.

    Args:
        expression: 계산할 수학 표현식

    Returns:
        계산 결과를 포함하는 딕셔너리
    """
    try:
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree)
        return {
            "expression": expression,
            "result": result,
            "message": "계산이 완료되었습니다."
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "message": "계산 중 오류가 발생했습니다."
        }


# ============================================================================
# 2026 신규 도구: Azure AI Foundry / Responses API 네이티브 기능
# ============================================================================

def azure_ai_foundry_agent_tool(
    task: str,
    agent_name: str = "default-agent",
    max_steps: int = 10
) -> Dict[str, Any]:
    """
    Azure AI Foundry Agent를 호출하여 복잡한 멀티스텝 작업을 수행합니다.

    Args:
        task: 수행할 작업 설명
        agent_name: 사용할 에이전트 이름
        max_steps: 최대 실행 단계 수

    Returns:
        에이전트 실행 결과를 포함하는 딕셔너리
    """
    return {
        "task": task,
        "agent_name": agent_name,
        "max_steps": max_steps,
        "status": "completed",
        "message": "Azure AI Foundry Agent 도구가 실행되었습니다."
    }


def azure_deep_research_tool(
    query: str,
    sources: Optional[List[str]] = None,
    depth: str = "standard"
) -> Dict[str, Any]:
    """
    Azure OpenAI Deep Research를 사용하여 심층 조사를 수행합니다.
    o3-deep-research 모델을 활용하여 웹 검색, 코드 실행 등 종합적인 리서치를 수행합니다.

    Args:
        query: 조사할 주제
        sources: 참조할 데이터 소스 목록
        depth: 조사 깊이 (quick, standard, thorough)

    Returns:
        조사 결과를 포함하는 딕셔너리
    """
    return {
        "query": query,
        "sources": sources or [],
        "depth": depth,
        "results": {},
        "message": "Azure Deep Research 도구가 실행되었습니다."
    }


def azure_web_search_tool(
    query: str,
    search_context_size: str = "medium",
    user_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Responses API 내장 웹 검색 도구를 사용하여 실시간 웹 정보를 검색합니다.
    web_search_preview 내장 도구를 에이전트 워크플로우에서 호출합니다.

    Args:
        query: 검색 쿼리
        search_context_size: 검색 컨텍스트 크기 (low, medium, high)
        user_location: 사용자 위치 (검색 결과 최적화용)

    Returns:
        검색 결과를 포함하는 딕셔너리
    """
    return {
        "query": query,
        "search_context_size": search_context_size,
        "user_location": user_location,
        "results": [],
        "message": "Azure 웹 검색 도구가 실행되었습니다."
    }


def azure_code_interpreter_tool(
    code: str,
    language: str = "python",
    files: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Responses API 코드 인터프리터를 사용하여 코드를 실행합니다.
    데이터 분석, 시각화, 수학 계산 등에 활용할 수 있습니다.

    Args:
        code: 실행할 코드
        language: 프로그래밍 언어 (python)
        files: 업로드할 파일 경로 목록

    Returns:
        코드 실행 결과를 포함하는 딕셔너리
    """
    return {
        "code": code[:100] + "..." if len(code) > 100 else code,
        "language": language,
        "files": files or [],
        "output": "",
        "message": "Azure Code Interpreter 도구가 실행되었습니다."
    }


def azure_image_generation_tool(
    prompt: str,
    model: str = "gpt-image-2",
    size: str = "1024x1024",
    quality: str = "high"
) -> Dict[str, Any]:
    """
    Azure OpenAI를 사용하여 이미지를 생성합니다. GPT-Image-2 모델을 활용합니다.

    Args:
        prompt: 이미지 생성 프롬프트
        model: 사용할 모델 (gpt-image-2 권장, gpt-image-1 호환)
        size: 이미지 크기
        quality: 이미지 품질 (low, medium, high)

    Returns:
        생성된 이미지 정보를 포함하는 딕셔너리
    """
    return {
        "prompt": prompt,
        "model": model,
        "size": size,
        "quality": quality,
        "image_url": "",
        "message": "Azure 이미지 생성 도구가 실행되었습니다."
    }


# ============================================================================
# 2026-02 신규 도구: Azure AI Agent Service / CUA / MCP / Structured Output
# ============================================================================

def azure_ai_agent_service_tool(
    task: str,
    agent_id: Optional[str] = None,
    tools: Optional[List[str]] = None,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Azure AI Agent Service를 호출하여 서버리스 에이전트 작업을 수행합니다.
    Azure AI Foundry에서 배포된 에이전트를 API로 호출합니다.

    Args:
        task: 수행할 작업 설명
        agent_id: 에이전트 ID (None이면 기본 에이전트)
        tools: 에이전트에서 사용할 도구 목록
        thread_id: 기존 스레드 ID (대화 이어가기)

    Returns:
        에이전트 실행 결과를 포함하는 딕셔너리
    """
    return {
        "task": task,
        "agent_id": agent_id or "default",
        "tools": tools or [],
        "thread_id": thread_id,
        "status": "completed",
        "message": "Azure AI Agent Service 도구가 실행되었습니다."
    }


def azure_computer_use_tool(
    instruction: str,
    environment: str = "browser",
    screenshot: bool = True,
    max_actions: int = 20
) -> Dict[str, Any]:
    """
    CUA(Computer Use Agent)를 사용하여 GUI 기반 자동화 작업을 수행합니다.
    computer-use-preview 모델로 화면을 인식하고 마우스/키보드 동작을 자동화합니다.

    Args:
        instruction: 수행할 GUI 자동화 작업 설명
        environment: 실행 환경 (browser, desktop, mobile)
        screenshot: 스크린샷 캡처 여부
        max_actions: 최대 액션 수

    Returns:
        자동화 결과를 포함하는 딕셔너리
    """
    return {
        "instruction": instruction,
        "environment": environment,
        "screenshot": screenshot,
        "max_actions": max_actions,
        "actions_performed": [],
        "message": "Azure CUA(Computer Use Agent) 도구가 실행되었습니다."
    }


def mcp_server_discovery_tool(
    query: str,
    category: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    MCP Registry에서 공개 MCP 서버를 검색하고 연결 정보를 반환합니다.
    Streamable HTTP 및 SSE 전송 프로토콜을 지원하는 서버를 검색합니다.

    Args:
        query: 검색 쿼리 (예: 'database', 'github', 'slack')
        category: 서버 카테고리 필터
        limit: 최대 결과 수

    Returns:
        검색된 MCP 서버 정보를 포함하는 딕셔너리
    """
    return {
        "query": query,
        "category": category,
        "limit": limit,
        "servers": [],
        "message": "MCP 서버 검색 도구가 실행되었습니다."
    }


def structured_output_tool(
    prompt: str,
    schema_name: str = "generic",
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Structured Outputs를 사용하여 Pydantic 스키마 기반으로 구조화된 응답을 생성합니다.
    JSON Schema를 따르는 정확한 형식의 데이터를 생성합니다.

    Args:
        prompt: 생성할 내용에 대한 프롬프트
        schema_name: 사용할 스키마 이름 (generic, report, analysis, summary)
        output_format: 출력 형식 (json, yaml, table)

    Returns:
        구조화된 출력 결과를 포함하는 딕셔너리
    """
    return {
        "prompt": prompt,
        "schema_name": schema_name,
        "output_format": output_format,
        "structured_data": {},
        "message": "Structured Output 도구가 실행되었습니다."
    }


def azure_realtime_audio_tool(
    audio_input: str,
    mode: str = "conversation",
    voice: str = "alloy",
    language: str = "ko-KR"
) -> Dict[str, Any]:
    """
    GPT-4o-realtime 모델을 사용하여 실시간 음성 대화를 수행합니다.
    WebSocket 기반의 저지연 양방향 음성 통신을 지원합니다.

    Args:
        audio_input: 오디오 입력 소스 (마이크, URL 등)
        mode: 대화 모드 (conversation, dictation, translation)
        voice: 음성 스타일 (alloy, echo, fable, onyx, nova, shimmer)
        language: 언어 코드

    Returns:
        실시간 음성 대화 결과를 포함하는 딕셔너리
    """
    return {
        "audio_input": audio_input,
        "mode": mode,
        "voice": voice,
        "language": language,
        "transcript": "",
        "message": "Azure Realtime Audio 도구가 실행되었습니다."
    }


# ============================================================================
# 도구 정의 (메타데이터 포함)
# ============================================================================

# 도구 등록 정보: (함수, 카테고리, 태그)
TOOL_DEFINITIONS = [
    # Azure AI Services
    (azure_ai_search_tool, "search", ["azure", "search", "ai", "cognitive", "문서검색", "검색"]),
    (azure_blob_storage_tool, "storage", ["azure", "blob", "storage", "파일", "저장소", "file"]),
    (azure_sql_query_tool, "database", ["azure", "sql", "database", "query", "데이터베이스", "쿼리"]),
    (azure_cosmos_db_tool, "database", ["azure", "cosmos", "nosql", "document", "데이터베이스"]),
    (azure_openai_embedding_tool, "ai", ["azure", "openai", "embedding", "vector", "임베딩", "벡터"]),
    (azure_computer_vision_tool, "ai", ["azure", "vision", "image", "ocr", "이미지", "비전", "분석"]),
    (azure_translator_tool, "ai", ["azure", "translator", "translate", "language", "번역", "언어"]),
    (azure_text_analytics_tool, "ai", ["azure", "text", "analytics", "sentiment", "nlp", "감정분석", "텍스트"]),
    (azure_form_recognizer_tool, "ai", ["azure", "form", "document", "ocr", "extraction", "문서", "추출"]),
    (azure_speech_to_text_tool, "ai", ["azure", "speech", "audio", "transcription", "음성", "STT"]),
    (azure_function_invoke_tool, "compute", ["azure", "function", "serverless", "invoke", "함수", "호출"]),
    # 2026 신규: Azure AI Foundry / Responses API 네이티브 도구
    (azure_ai_foundry_agent_tool, "ai", ["azure", "foundry", "agent", "멀티스텝", "에이전트", "자동화"]),
    (azure_deep_research_tool, "ai", ["azure", "research", "deep", "조사", "리서치", "분석", "o3"]),
    (azure_web_search_tool, "search", ["azure", "web", "search", "실시간", "웹검색", "인터넷", "grounding"]),
    (azure_code_interpreter_tool, "compute", ["azure", "code", "interpreter", "python", "코드실행", "데이터분석"]),
    (azure_image_generation_tool, "ai", ["azure", "image", "generation", "gpt-image", "이미지생성", "DALL-E"]),
    # 2026-02 신규: Azure AI Agent Service / CUA / MCP / Structured Output
    (azure_ai_agent_service_tool, "ai", ["azure", "agent", "service", "serverless", "에이전트서비스", "AI에이전트", "foundry"]),
    (azure_computer_use_tool, "ai", ["azure", "computer", "use", "CUA", "GUI", "자동화", "브라우저", "데스크톱"]),
    (mcp_server_discovery_tool, "mcp", ["mcp", "server", "registry", "discovery", "서버검색", "MCP서버", "프로토콜"]),
    (structured_output_tool, "ai", ["structured", "output", "pydantic", "schema", "JSON", "구조화", "스키마"]),
    (azure_realtime_audio_tool, "ai", ["azure", "realtime", "audio", "voice", "websocket", "실시간", "음성대화", "GPT-4o"]),
    # 외부 서비스
    (bing_web_search_tool, "search", ["bing", "web", "search", "internet", "웹검색", "인터넷"]),
    (github_search_tool, "search", ["github", "code", "repository", "코드", "저장소", "개발"]),
    (weather_api_tool, "utility", ["weather", "forecast", "temperature", "날씨", "기온", "예보"]),
    (calculator_tool, "utility", ["calculator", "math", "compute", "계산", "수학"]),
]


# ============================================================================
# 동적 도구 관리 함수
# ============================================================================

def search_available_tools(query: str, top_k: int = 5) -> List[str]:
    """
    사용 가능한 도구 라이브러리에서 적합한 도구를 검색합니다.

    특정 작업에 필요한 도구가 없을 때 이 함수를 사용하세요.
    검색 결과는 '도구이름: 설명' 형식으로 반환됩니다.

    Args:
        query: 검색 키워드 (예: 'search', 'database', 'translate', 'image')
        top_k: 반환할 최대 결과 수

    Returns:
        도구 이름과 설명을 포함하는 문자열 목록

    Example:
        >>> search_available_tools("번역")
        ['azure_translator: Azure Translator를 사용하여 텍스트를 번역합니다.']
    """
    results = registry.search(query, top_k)

    if not results:
        return [
            "검색 결과가 없습니다. 다른 키워드로 시도해 보세요.",
            f"현재 등록된 도구 수: {registry.count()}"
        ]

    logger.info(f"도구 검색 쿼리: '{query}', 결과 수: {len(results)}")
    return results


def load_tool(tool_name: str) -> str:
    """
    특정 도구를 현재 컨텍스트에 로드합니다.

    'search_available_tools'로 도구를 찾은 후 이 함수를 호출하여
    도구를 활성화하세요. 로드된 도구는 다음 턴에서 사용할 수 있습니다.

    Args:
        tool_name: 로드할 도구의 정확한 이름

    Returns:
        도구 로드 결과 메시지

    Example:
        >>> load_tool("azure_translator")
        "도구 'azure_translator'가 성공적으로 로드되었습니다."
    """
    tool = registry.get_tool(tool_name)

    if tool:
        metadata = registry.get_tool_metadata(tool_name)
        description = metadata.get("description", "") if metadata else ""
        logger.info(f"도구 로드됨: {tool_name}")
        return f"도구 '{tool_name}'가 성공적으로 로드되었습니다. 설명: {description[:100]}"

    # 비슷한 도구 제안
    similar_tools = registry.search(tool_name, top_k=3)
    suggestion = ""
    if similar_tools:
        suggestion = f"\n비슷한 도구: {', '.join([t.split(':')[0] for t in similar_tools])}"

    logger.warning(f"도구를 찾을 수 없음: {tool_name}")
    return f"오류: 도구 '{tool_name}'를 찾을 수 없습니다.{suggestion}"


def register_tool(
    tool: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> None:
    """
    새로운 도구를 레지스트리에 등록합니다.

    Args:
        tool: 등록할 도구 함수
        name: 도구 이름 (None이면 함수 이름 사용)
        description: 도구 설명 (None이면 docstring 사용)
        category: 도구 카테고리
        tags: 검색을 위한 태그 목록
    """
    registry.register(tool, name, description, category, tags)


def get_tool_info(tool_name: str) -> Dict[str, Any]:
    """
    도구의 상세 정보를 반환합니다.

    Args:
        tool_name: 도구 이름

    Returns:
        도구 메타데이터 딕셔너리
    """
    tool = registry.get_tool(tool_name)
    metadata = registry.get_tool_metadata(tool_name)

    if not tool:
        return {"error": f"도구 '{tool_name}'를 찾을 수 없습니다."}

    return {
        "name": tool_name,
        "callable": tool is not None,
        **(metadata or {})
    }


def initialize_mcp_tools() -> None:
    """
    모든 MCP 도구를 초기화하고 레지스트리에 등록합니다.
    register_batch()를 사용하여 BM25 인덱스를 1회만 재구축합니다.
    """
    logger.info("=== MCP 도구 초기화 시작 ===")

    # 일괄 등록용 튜플 리스트 생성: (tool, name, description, category, tags)
    batch = []
    for tool_func, category, tags in TOOL_DEFINITIONS:
        batch.append((tool_func, None, None, category, tags))

    registry.register_batch(batch)

    logger.info(f"=== MCP 도구 초기화 완료: {registry.count()}개 도구 등록됨 (v3.0) ===")
