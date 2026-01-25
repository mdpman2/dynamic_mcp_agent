# -*- coding: utf-8 -*-
"""
Dynamic Tool Loading Functions for Azure-based MCP Agent

이 모듈은 다양한 MCP 도구들을 동적으로 로드하고 관리하는 기능을 제공합니다.
Azure OpenAI, Azure AI Search, Azure Functions 등 다양한 Azure 서비스와 통합됩니다.

참고: https://medium.com/google-cloud/implementing-anthropic-style-dynamic-tool-search-tool-f39d02a35139
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dotenv import load_dotenv

from .registry import registry

logger = logging.getLogger(__name__)
load_dotenv()


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
    model: str = "text-embedding-3-small"
) -> Dict[str, Any]:
    """
    Azure OpenAI를 사용하여 텍스트 임베딩을 생성합니다.
    
    Args:
        text: 임베딩할 텍스트
        model: 사용할 임베딩 모델
    
    Returns:
        임베딩 벡터를 포함하는 딕셔너리
    """
    return {
        "text": text[:50] + "..." if len(text) > 50 else text,
        "model": model,
        "embedding_dimension": 1536,
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
        # 안전한 계산 (eval 대신 더 안전한 방법 사용 권장)
        result = eval(expression, {"__builtins__": {}}, {})
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


async def initialize_mcp_tools() -> None:
    """
    모든 MCP 도구를 초기화하고 레지스트리에 등록합니다.
    TOOL_DEFINITIONS 리스트를 순회하며 일괄 등록합니다.
    """
    logger.info("=== MCP 도구 초기화 시작 ===")
    
    for tool_func, category, tags in TOOL_DEFINITIONS:
        register_tool(tool_func, category=category, tags=tags)
    
    logger.info(f"=== MCP 도구 초기화 완료: {registry.count()}개 도구 등록됨 ===")



# 모듈 로드 시 자동 초기화 (필요시 주석 해제)
# def _auto_initialize():
#     """모듈 임포트 시 자동으로 도구를 초기화합니다."""
#     try:
#         asyncio.run(initialize_mcp_tools())
#     except Exception as e:
#         logger.error(f"MCP 도구 자동 초기화 실패: {e}")
# _auto_initialize()
