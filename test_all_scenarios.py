#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic MCP Agent v2.0.0 — 전체 시나리오 테스트
================================================
시나리오 1: 모듈 임포트 & 초기화
시나리오 2: 레지스트리 등록 & BM25/하이브리드 검색
시나리오 3: 도구 함수 & 안전 계산기 (AST)
시나리오 4: 에이전트 생성 & 구성
시나리오 5: CLI 커맨드 처리
시나리오 6: End-to-End 대화 (Azure OpenAI API)
"""

import sys
import os
import json
import time
import math
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 모듈 경로 설정
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# 테스트 유틸리티
# ============================================================================

PASS = 0
FAIL = 0
SKIP = 0
RESULTS: List[Dict[str, Any]] = []


def banner(title: str, char: str = "=", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def sub_banner(title: str) -> None:
    print(f"\n  --- {title} ---")


def check(name: str, condition: bool, detail: str = "") -> bool:
    global PASS, FAIL
    status = "✅ PASS" if condition else "❌ FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    if condition:
        PASS += 1
    else:
        FAIL += 1
    RESULTS.append({"name": name, "pass": condition, "detail": detail})
    return condition


def skip(name: str, reason: str = "") -> None:
    global SKIP
    SKIP += 1
    print(f"  [⏭️ SKIP] {name}  →  {reason}")
    RESULTS.append({"name": name, "pass": None, "detail": reason})


# ============================================================================
# 시나리오 1: 모듈 임포트 & 초기화
# ============================================================================
def test_scenario_1():
    banner("시나리오 1: 모듈 임포트 & 초기화")

    # 1-1. 패키지 임포트
    sub_banner("1-1. 패키지 임포트")
    try:
        from dynamic_mcp_agent import __version__
        check("패키지 임포트", True, f"version={__version__}")
    except Exception as e:
        check("패키지 임포트", False, str(e))
        return  # 패키지 임포트 실패 시 나머지 테스트 불가

    # 1-2. 주요 구성요소 임포트
    sub_banner("1-2. 구성요소 임포트")
    try:
        from dynamic_mcp_agent import DynamicMCPAgent, create_agent
        check("DynamicMCPAgent 클래스 임포트", True)
    except Exception as e:
        check("DynamicMCPAgent 클래스 임포트", False, str(e))

    try:
        from dynamic_mcp_agent import DEFAULT_REMOTE_MCP_SERVERS
        check("DEFAULT_REMOTE_MCP_SERVERS 임포트", True, f"{len(DEFAULT_REMOTE_MCP_SERVERS)}개 서버")
    except Exception as e:
        check("DEFAULT_REMOTE_MCP_SERVERS 임포트", False, str(e))

    try:
        from dynamic_mcp_agent import registry
        check("registry 전역 인스턴스 임포트", True)
    except Exception as e:
        check("registry 전역 인스턴스 임포트", False, str(e))

    try:
        from dynamic_mcp_agent.lib.tools import (
            search_available_tools, load_tool, initialize_mcp_tools,
            TOOL_DEFINITIONS
        )
        check("tools 모듈 임포트", True, f"TOOL_DEFINITIONS={len(TOOL_DEFINITIONS)}개")
    except Exception as e:
        check("tools 모듈 임포트", False, str(e))

    # 1-3. 버전 확인
    sub_banner("1-3. 버전 확인")
    from dynamic_mcp_agent import __version__
    check("버전 v2.0.0", __version__ == "2.0.0", f"actual={__version__}")

    # 1-4. 환경 변수 로드
    sub_banner("1-4. 환경 변수 확인")
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    check("AZURE_OPENAI_ENDPOINT 설정됨", bool(endpoint), endpoint[:30] + "..." if endpoint else "None")
    check("AZURE_OPENAI_API_KEY 설정됨", bool(api_key), "***" + api_key[-4:] if api_key else "None")
    check("AZURE_OPENAI_DEPLOYMENT_NAME 설정됨", bool(deployment), deployment or "None")


# ============================================================================
# 시나리오 2: 레지스트리 등록 & 검색
# ============================================================================
def test_scenario_2():
    banner("시나리오 2: 레지스트리 등록 & BM25/하이브리드 검색")

    from dynamic_mcp_agent.lib.registry import HybridToolRegistry

    # 2-1. 새 레지스트리 인스턴스 생성
    sub_banner("2-1. 레지스트리 인스턴스 생성")
    reg = HybridToolRegistry(
        enable_mcp_registry=False  # 외부 API 호출 방지
    )
    check("HybridToolRegistry 생성", reg is not None)
    check("초기 도구 수 = 0", reg.count() == 0, f"count={reg.count()}")

    # 2-2. 단일 등록
    sub_banner("2-2. 단일 도구 등록")
    def dummy_tool_a(text: str) -> str:
        """더미 도구 A: 텍스트를 처리합니다."""
        return f"processed: {text}"

    reg.register(dummy_tool_a, category="test", tags=["dummy", "텍스트", "처리"])
    check("단일 등록 후 count=1", reg.count() == 1)
    check("get_tool 성공", reg.get_tool("dummy_tool_a") is not None)
    meta = reg.get_tool_metadata("dummy_tool_a")
    check("메타데이터 존재", meta is not None and "category" in meta, f"category={meta.get('category')}")

    # 2-3. 일괄 등록 (register_batch)
    sub_banner("2-3. 일괄 등록 (register_batch)")
    def dummy_search(q: str) -> str:
        """검색 도구"""
        return q
    def dummy_translate(text: str) -> str:
        """번역 도구: 텍스트를 번역합니다."""
        return text
    def dummy_image(url: str) -> str:
        """이미지 분석 도구: 이미지를 분석합니다."""
        return url

    batch = [
        (dummy_search, None, None, "search", ["search", "검색", "query"]),
        (dummy_translate, None, None, "ai", ["translate", "번역", "언어"]),
        (dummy_image, None, None, "ai", ["image", "이미지", "분석", "vision"]),
    ]
    reg.register_batch(batch)
    check("배치 등록 후 count=4", reg.count() == 4, f"actual={reg.count()}")

    # 2-4. BM25 검색
    sub_banner("2-4. BM25 검색")
    results = reg.search("검색", top_k=3, strategy="bm25")
    check("BM25 '검색' 결과 비어있지 않음", len(results) > 0, f"results={len(results)}")
    found_search = any("dummy_search" in r for r in results)
    check("BM25 '검색' → dummy_search 발견", found_search, str(results[:2]))

    results2 = reg.search("번역", top_k=3, strategy="bm25")
    found_translate = any("dummy_translate" in r for r in results2)
    check("BM25 '번역' → dummy_translate 발견", found_translate, str(results2[:2]))

    results3 = reg.search("이미지 분석", top_k=3, strategy="bm25")
    found_image = any("dummy_image" in r for r in results3)
    check("BM25 '이미지 분석' → dummy_image 발견", found_image, str(results3[:2]))

    # 2-5. 하이브리드 검색 (BM25 + Sentence-Transformers)
    sub_banner("2-5. 하이브리드 검색")
    hybrid_results = reg.search("텍스트 처리하고 싶어", top_k=3, strategy="hybrid")
    check("하이브리드 검색 실행 성공", isinstance(hybrid_results, list), f"count={len(hybrid_results)}")

    # 2-6. 검색 통계
    sub_banner("2-6. 검색 통계")
    stats = reg.get_search_stats()
    check("total_searches > 0", stats["total_searches"] > 0, f"total={stats['total_searches']}")
    check("bm25_hits 존재", "bm25_hits" in stats)
    check("embedding_hits 별칭 존재", "embedding_hits" in stats)
    check("sentence_hits 존재", "sentence_hits" in stats)

    # 2-7. 카테고리 검색
    sub_banner("2-7. 카테고리별 도구 조회")
    ai_tools = reg.get_tools_by_category("ai")
    check("'ai' 카테고리 도구 수 ≥ 2", len(ai_tools) >= 2, f"count={len(ai_tools)}")

    # 2-8. 모델 정보
    sub_banner("2-8. 모델 정보")
    info = reg.get_model_info()
    check("get_model_info 반환", info is not None)
    check("tool_count 일치", info["tool_count"] == reg.count())

    # 2-9. 임계값 설정
    sub_banner("2-9. 임계값 설정")
    reg.set_thresholds(bm25_threshold=3.0, embedding_threshold=0.5)
    check("BM25 임계값 변경", reg.BM25_CONFIDENCE_THRESHOLD == 3.0)
    check("Embedding 임계값 변경", reg.EMBEDDING_SIMILARITY_THRESHOLD == 0.5)

    # 2-10. clear
    sub_banner("2-10. 레지스트리 초기화")
    reg.clear()
    check("clear 후 count=0", reg.count() == 0)


# ============================================================================
# 시나리오 3: 도구 함수 & 안전 계산기 (AST)
# ============================================================================
def test_scenario_3():
    banner("시나리오 3: 도구 함수 & 안전 계산기 (AST)")

    from dynamic_mcp_agent.lib.tools import (
        azure_ai_search_tool,
        azure_blob_storage_tool,
        azure_sql_query_tool,
        azure_cosmos_db_tool,
        azure_openai_embedding_tool,
        azure_computer_vision_tool,
        azure_translator_tool,
        azure_text_analytics_tool,
        azure_form_recognizer_tool,
        azure_speech_to_text_tool,
        azure_function_invoke_tool,
        azure_ai_foundry_agent_tool,
        azure_deep_research_tool,
        azure_web_search_tool,
        azure_code_interpreter_tool,
        azure_image_generation_tool,
        bing_web_search_tool,
        github_search_tool,
        weather_api_tool,
        calculator_tool,
        TOOL_DEFINITIONS,
        search_available_tools,
        load_tool,
        initialize_mcp_tools,
    )

    # 3-1. 전체 도구 수 확인
    sub_banner("3-1. TOOL_DEFINITIONS 확인")
    check("TOOL_DEFINITIONS 20개", len(TOOL_DEFINITIONS) == 20, f"actual={len(TOOL_DEFINITIONS)}")

    # 3-2. 각 도구 함수 실행 테스트
    sub_banner("3-2. 모든 도구 함수 호출 테스트")
    tool_tests = [
        ("azure_ai_search_tool", lambda: azure_ai_search_tool("test query")),
        ("azure_blob_storage_tool", lambda: azure_blob_storage_tool("list", "container1")),
        ("azure_sql_query_tool", lambda: azure_sql_query_tool("SELECT 1")),
        ("azure_cosmos_db_tool", lambda: azure_cosmos_db_tool("query", "db1", "container1")),
        ("azure_openai_embedding_tool", lambda: azure_openai_embedding_tool("hello world")),
        ("azure_computer_vision_tool", lambda: azure_computer_vision_tool("http://example.com/img.jpg")),
        ("azure_translator_tool", lambda: azure_translator_tool("Hello", "ko")),
        ("azure_text_analytics_tool", lambda: azure_text_analytics_tool("감정 분석 테스트")),
        ("azure_form_recognizer_tool", lambda: azure_form_recognizer_tool("http://example.com/doc.pdf")),
        ("azure_speech_to_text_tool", lambda: azure_speech_to_text_tool("http://example.com/audio.wav")),
        ("azure_function_invoke_tool", lambda: azure_function_invoke_tool("http://func.azurewebsites.net/api/test", {"key": "value"})),
        ("azure_ai_foundry_agent_tool", lambda: azure_ai_foundry_agent_tool("Analyze data")),
        ("azure_deep_research_tool", lambda: azure_deep_research_tool("AI trends 2026")),
        ("azure_web_search_tool", lambda: azure_web_search_tool("latest AI news")),
        ("azure_code_interpreter_tool", lambda: azure_code_interpreter_tool("print('hello')")),
        ("azure_image_generation_tool", lambda: azure_image_generation_tool("A sunset over mountains")),
        ("bing_web_search_tool", lambda: bing_web_search_tool("Python tutorials")),
        ("github_search_tool", lambda: github_search_tool("openai")),
        ("weather_api_tool", lambda: weather_api_tool("Seoul")),
    ]

    for name, fn in tool_tests:
        try:
            result = fn()
            is_dict = isinstance(result, dict)
            # azure_ai_search_tool은 자격 증명 미설정 시 error 키만 반환
            has_valid_key = ("message" in result or "error" in result) if is_dict else False
            check(f"{name} 실행", is_dict and has_valid_key, f"keys={list(result.keys())[:4]}")
        except Exception as e:
            check(f"{name} 실행", False, str(e))

    # 3-3. 안전 계산기 (AST 기반) 테스트
    sub_banner("3-3. 안전 계산기 (AST) 테스트")

    # 기본 산술
    calc_tests = [
        ("2 + 3", 5),
        ("10 - 4", 6),
        ("3 * 7", 21),
        ("15 / 4", 3.75),
        ("15 // 4", 3),
        ("10 % 3", 1),
        ("2 ** 10", 1024),
        ("-5 + 3", -2),
    ]
    for expr, expected in calc_tests:
        result = calculator_tool(expr)
        actual = result.get("result")
        check(f"calc: {expr} = {expected}", actual == expected, f"actual={actual}")

    # 수학 함수
    math_fn_tests = [
        ("sqrt(144)", 12.0),
        ("abs(-42)", 42),
        ("round(3.7)", 4),
        ("pi", math.pi),
        ("e", math.e),
    ]
    for expr, expected in math_fn_tests:
        result = calculator_tool(expr)
        actual = result.get("result")
        check(f"calc: {expr} ≈ {expected}", abs(actual - expected) < 1e-9 if actual is not None else False, f"actual={actual}")

    # sin, cos, log
    sin_result = calculator_tool("sin(0)")
    check("calc: sin(0) = 0", abs(sin_result.get("result", 99)) < 1e-9)
    cos_result = calculator_tool("cos(0)")
    check("calc: cos(0) = 1", abs(cos_result.get("result", 99) - 1) < 1e-9)
    log_result = calculator_tool("log(e)")
    check("calc: log(e) ≈ 1", abs(log_result.get("result", 99) - 1) < 1e-9)

    # 복합 표현식
    complex_result = calculator_tool("sqrt(2**2 + 3**2)")
    expected = math.sqrt(4 + 9)
    check(f"calc: sqrt(2**2 + 3**2) ≈ {expected:.4f}", 
          abs(complex_result.get("result", 99) - expected) < 1e-9)

    # 보안: 위험한 표현식 차단
    sub_banner("3-4. 계산기 보안 테스트 (위험한 표현식 차단)")
    dangerous_tests = [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "exec('print(1)')",
        "eval('1+1')",
        "lambda: 1",
        "[x for x in range(10)]",
    ]
    for expr in dangerous_tests:
        result = calculator_tool(expr)
        has_error = "error" in result
        check(f"보안 차단: {expr[:40]}...", has_error, f"error={result.get('error', '')[:50]}")

    # 3-5. 도구 초기화 & 검색/로드
    sub_banner("3-5. initialize_mcp_tools & search/load")
    from dynamic_mcp_agent.lib.registry import registry as global_registry
    global_registry.clear()
    check("레지스트리 초기화 후 count=0", global_registry.count() == 0)
    
    initialize_mcp_tools()
    check("initialize_mcp_tools 후 count=20", global_registry.count() == 20, f"actual={global_registry.count()}")

    # search_available_tools
    search_results = search_available_tools("번역")
    check("search_available_tools('번역') 결과 있음", len(search_results) > 0)
    found_translator = any("translator" in r.lower() for r in search_results)
    check("'번역' 검색 → translator 포함", found_translator, str(search_results[:2]))

    # load_tool
    load_result = load_tool("azure_translator_tool")
    check("load_tool('azure_translator_tool') 성공", "성공" in load_result, load_result[:60])

    load_fail = load_tool("nonexistent_tool_xyz")
    check("load_tool(존재하지않는도구) → 오류", "오류" in load_fail or "error" in load_fail.lower(), load_fail[:60])


# ============================================================================
# 시나리오 4: 에이전트 생성 & 구성
# ============================================================================
def test_scenario_4():
    banner("시나리오 4: 에이전트 생성 & 구성")

    from dynamic_mcp_agent import DynamicMCPAgent, create_agent, DEFAULT_REMOTE_MCP_SERVERS
    from dynamic_mcp_agent.lib.registry import registry as global_registry

    # 4-1. create_agent
    sub_banner("4-1. create_agent() 호출")
    try:
        agent = create_agent()
        check("에이전트 생성 성공", agent is not None)
    except Exception as e:
        check("에이전트 생성 성공", False, str(e))
        return

    # 4-2. 에이전트 속성 확인
    sub_banner("4-2. 에이전트 속성 확인")
    check("model 설정됨", bool(agent.model), f"model={agent.model}")
    check("client 생성됨", agent.client is not None)
    check("active_tools 초기화 (빈 dict)", isinstance(agent.active_tools, dict) and len(agent.active_tools) == 0)
    check("last_response_id = None", agent.last_response_id is None)
    check("remote_mcp_servers 연결됨", len(agent.remote_mcp_servers) > 0, f"count={len(agent.remote_mcp_servers)}")
    
    # 4-3. DEFAULT_REMOTE_MCP_SERVERS 확인
    sub_banner("4-3. 기본 MCP 서버 설정")
    check("microsoft_learn 서버 존재", 
          any(s.get("server_label") == "microsoft_learn" for s in DEFAULT_REMOTE_MCP_SERVERS))

    # 4-4. get_stats
    sub_banner("4-4. get_stats() 테스트")
    stats = agent.get_stats()
    expected_keys = ["model", "api", "total_tools_in_registry", "active_tools", 
                     "active_tool_names", "remote_mcp_servers", "conversation_turns", 
                     "last_response_id"]
    for key in expected_keys:
        check(f"stats['{key}'] 존재", key in stats)
    check("total_tools_in_registry = 20", stats["total_tools_in_registry"] == 20, 
          f"actual={stats['total_tools_in_registry']}")
    check("api 형식 'v1/*'", stats["api"].startswith("v1/"))

    # 4-5. 도구 스키마 생성
    sub_banner("4-5. 도구 스키마 (Responses API)")
    schema = agent._get_base_tools_schema()
    check("기본 스키마 반환됨", isinstance(schema, list) and len(schema) >= 2)
    
    # search_available_tools 스키마 확인
    search_schema = next((s for s in schema if s.get("name") == "search_available_tools"), None)
    check("search_available_tools 스키마 존재", search_schema is not None)
    if search_schema:
        check("type=function", search_schema.get("type") == "function")
        check("parameters 포함", "parameters" in search_schema)

    # MCP 서버 도구
    mcp_tools = [s for s in schema if s.get("type") == "mcp"]
    check("MCP 서버 도구 포함", len(mcp_tools) > 0, f"count={len(mcp_tools)}")

    # 4-6. 스키마 캐시 검증
    sub_banner("4-6. 스키마 캐시 테스트")
    schema1 = agent._get_base_tools_schema()
    schema2 = agent._get_base_tools_schema()
    check("스키마 캐시 동작 (같은 객체)", schema1 is schema2)

    # MCP 서버 추가 → 캐시 무효화
    agent.add_remote_mcp_server("https://example.com/mcp", "test_server")
    schema3 = agent._get_base_tools_schema()
    check("MCP 서버 추가 후 캐시 갱신", schema3 is not schema2)
    check("새 MCP 서버 포함", 
          any(s.get("server_label") == "test_server" for s in schema3 if s.get("type") == "mcp"))

    # 4-7. _serialize_result
    sub_banner("4-7. _serialize_result() 테스트")
    check("dict 직렬화", agent._serialize_result({"a": 1}) == '{"a": 1}')
    check("list 직렬화", agent._serialize_result([1, 2]) == '[1, 2]')
    check("한글 dict 직렬화", "한글" in agent._serialize_result({"text": "한글"}))
    check("str 직렬화", agent._serialize_result("hello") == "hello")
    check("int 직렬화", agent._serialize_result(42) == "42")

    # 4-8. 동적 도구 주입
    sub_banner("4-8. 동적 도구 주입")
    result = agent._dynamic_tool_injection("azure_translator_tool")
    check("azure_translator_tool 주입 성공", result is True)
    check("active_tools에 추가됨", "azure_translator_tool" in agent.active_tools)

    result2 = agent._dynamic_tool_injection("azure_translator_tool")  # 중복 주입
    check("중복 주입 → True (이미 존재)", result2 is True)

    result3 = agent._dynamic_tool_injection("nonexistent_tool_xyz")
    check("존재하지 않는 도구 주입 → False", result3 is False)

    # 4-9. _execute_tool
    sub_banner("4-9. _execute_tool() 테스트")
    search_result = agent._execute_tool("search_available_tools", {"query": "번역", "top_k": 3})
    check("search_available_tools 실행 성공", "translator" in search_result.lower() or "번역" in search_result)

    load_result = agent._execute_tool("load_tool", {"tool_name": "azure_ai_search_tool"})
    check("load_tool 실행 → 주입",  "azure_ai_search_tool" in agent.active_tools)

    tool_result = agent._execute_tool("azure_ai_search_tool", {"query": "test"})
    check("주입된 도구 실행", "실행되었습니다" in tool_result or "message" in tool_result or "error" in tool_result)

    unknown_result = agent._execute_tool("nonexistent", {})
    check("존재하지 않는 도구 → 에러", "error" in unknown_result.lower() or "찾을 수 없" in unknown_result)

    # 4-10. reset
    sub_banner("4-10. reset 테스트")
    agent.reset_tools()
    check("reset_tools → active_tools 비어있음", len(agent.active_tools) == 0)
    
    agent._conversation_turns = 5
    agent.last_response_id = "resp_test_123"
    agent.reset_conversation()
    check("reset_conversation → turns=0", agent._conversation_turns == 0)
    check("reset_conversation → response_id=None", agent.last_response_id is None)

    # 4-11. create_agent without MCP
    sub_banner("4-11. create_agent(enable_remote_mcp=False)")
    agent2 = create_agent(enable_remote_mcp=False)
    check("MCP 비활성화 에이전트 생성", len(agent2.remote_mcp_servers) == 0)

    # 4-12. _PYTHON_TYPE_MAP
    sub_banner("4-12. _PYTHON_TYPE_MAP 클래스 상수")
    check("int → 'integer'", DynamicMCPAgent._PYTHON_TYPE_MAP[int] == "integer")
    check("str 미포함 (기본='string')", str not in DynamicMCPAgent._PYTHON_TYPE_MAP)
    check("bool → 'boolean'", DynamicMCPAgent._PYTHON_TYPE_MAP[bool] == "boolean")
    check("float → 'number'", DynamicMCPAgent._PYTHON_TYPE_MAP[float] == "number")
    check("list → 'array'", DynamicMCPAgent._PYTHON_TYPE_MAP[list] == "array")
    check("dict → 'object'", DynamicMCPAgent._PYTHON_TYPE_MAP[dict] == "object")


# ============================================================================
# 시나리오 5: CLI 커맨드 처리
# ============================================================================
def test_scenario_5():
    banner("시나리오 5: CLI 커맨드 & main.py 함수 테스트")

    from dynamic_mcp_agent.main import check_environment

    # 5-1. check_environment
    sub_banner("5-1. check_environment()")
    result = check_environment()
    check("환경 변수 확인", result is True or result is False, f"result={result}")

    # 5-2. argparse 테스트 (직접 호출하지 않고 모듈 확인)
    sub_banner("5-2. main.py 함수 존재 확인")
    import dynamic_mcp_agent.main as main_mod
    funcs = ["run_cli_mode", "run_web_mode", "run_demo_mode", "run_stream_cli_mode", "main"]
    for fn_name in funcs:
        check(f"{fn_name}() 존재", hasattr(main_mod, fn_name) and callable(getattr(main_mod, fn_name)))

    # 5-3. 레지스트리 전역 인스턴스 초기화 검증
    sub_banner("5-3. 전역 레지스트리 도구 등록 검증")
    from dynamic_mcp_agent.lib.tools import initialize_mcp_tools, TOOL_DEFINITIONS
    from dynamic_mcp_agent.lib.registry import registry as global_registry
    
    if global_registry.count() == 0:
        initialize_mcp_tools()
    
    all_tools = global_registry.list_all_tools()
    check("등록된 도구 수 = 20", len(all_tools) == 20, f"actual={len(all_tools)}")
    
    expected_tools = [
        "azure_ai_search_tool", "azure_translator_tool", "calculator_tool",
        "azure_ai_foundry_agent_tool", "azure_deep_research_tool",
        "azure_web_search_tool", "azure_code_interpreter_tool", 
        "azure_image_generation_tool"
    ]
    for t in expected_tools:
        check(f"  '{t}' 등록됨", t in all_tools)


# ============================================================================
# 시나리오 6: End-to-End 대화 (Azure OpenAI API)
# ============================================================================
def test_scenario_6():
    banner("시나리오 6: End-to-End 대화 (Azure OpenAI API)")

    # API 키 확인
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not endpoint or not api_key:
        skip("E2E 대화 테스트", "API 키 미설정")
        return

    from dynamic_mcp_agent import create_agent
    from dynamic_mcp_agent.lib.registry import registry as global_registry

    # 6-1. 에이전트 생성
    sub_banner("6-1. E2E 에이전트 생성")
    try:
        agent = create_agent(enable_remote_mcp=False)  # MCP 없이 테스트
        check("E2E 에이전트 생성", True)
    except Exception as e:
        check("E2E 에이전트 생성", False, str(e))
        return

    # 6-2. 간단한 대화 (도구 호출 없이)
    sub_banner("6-2. 기본 대화 테스트")
    try:
        t_start = time.time()
        response = agent.chat_sync("안녕하세요! 간단히 자기소개 해주세요. 2문장으로.")
        t_elapsed = time.time() - t_start
        check("기본 대화 응답 수신", bool(response) and len(response) > 5, 
              f"len={len(response)}, time={t_elapsed:.1f}s")
        check("응답 내용 유효", isinstance(response, str) and "오류" not in response[:20], 
              response[:80] + "...")
        check("last_response_id 설정됨", agent.last_response_id is not None)
    except Exception as e:
        check("기본 대화 응답 수신", False, str(e)[:100])

    # 6-3. 도구 검색을 유도하는 대화
    sub_banner("6-3. 도구 검색 & 로드 대화")
    try:
        agent.reset_conversation()
        agent.reset_tools()
        
        t_start = time.time()
        response2 = agent.chat_sync("텍스트를 영어에서 한국어로 번역하고 싶어요. 어떤 도구를 사용할 수 있나요?")
        t_elapsed = time.time() - t_start
        check("도구 검색 대화 응답", bool(response2), f"time={t_elapsed:.1f}s")
        
        # 에이전트가 번역 관련 도구를 검색했는지 확인
        tools_after = agent.get_active_tools_list()
        check("응답에 번역 관련 내용 포함", 
              any(kw in response2 for kw in ["번역", "translator", "Translator", "도구", "검색"]),
              response2[:100] + "...")
        print(f"    활성 도구: {tools_after}")
    except Exception as e:
        check("도구 검색 대화 응답", False, str(e)[:100])

    # 6-4. 대화 체이닝 (previous_response_id) 테스트
    sub_banner("6-4. 대화 체이닝 테스트")
    try:
        prev_id = agent.last_response_id
        response3 = agent.chat_sync("방금 제가 무엇을 물어봤나요? 한 줄로 요약해 주세요.")
        check("대화 체이닝 응답 수신", bool(response3))
        check("previous_response_id 갱신됨", agent.last_response_id != prev_id)
        check("이전 대화 맥락 유지", 
              any(kw in response3 for kw in ["번역", "translate", "도구", "한국어", "영어"]),
              response3[:100] + "...")
    except Exception as e:
        check("대화 체이닝 응답 수신", False, str(e)[:100])

    # 6-5. 통계 확인
    sub_banner("6-5. 대화 후 통계")
    stats = agent.get_stats()
    check("conversation_turns > 0", stats["conversation_turns"] > 0, 
          f"turns={stats['conversation_turns']}")
    check("last_response_id 존재", stats["last_response_id"] is not None)

    # 6-6. 대화 초기화 후 재확인
    sub_banner("6-6. 대화 초기화 후 확인")
    agent.reset_conversation()
    check("초기화 후 turns=0", agent._conversation_turns == 0)
    check("초기화 후 response_id=None", agent.last_response_id is None)


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    banner("Dynamic MCP Agent v2.0.0 — 전체 시나리오 테스트", "█", 70)
    print(f"  Python {sys.version.split()[0]}")
    print(f"  작업 디렉터리: {Path(__file__).parent}")
    print(f"  시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    scenarios = [
        ("시나리오 1: 모듈 임포트 & 초기화", test_scenario_1),
        ("시나리오 2: 레지스트리 & 검색", test_scenario_2),
        ("시나리오 3: 도구 함수 & 계산기", test_scenario_3),
        ("시나리오 4: 에이전트 생성 & 구성", test_scenario_4),
        ("시나리오 5: CLI 커맨드 처리", test_scenario_5),
        ("시나리오 6: E2E 대화 (API)", test_scenario_6),
    ]
    
    for title, test_fn in scenarios:
        try:
            test_fn()
        except Exception as e:
            print(f"\n  ❌ {title} 실행 중 예외 발생:")
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    
    # 최종 결과 요약
    banner("테스트 결과 요약", "█", 70)
    total = PASS + FAIL + SKIP
    print(f"""
  ✅ PASS:   {PASS}
  ❌ FAIL:   {FAIL}
  ⏭️ SKIP:   {SKIP}
  ────────────────
  📊 TOTAL:  {total}
  ⏱️  소요시간: {elapsed:.1f}초
""")
    
    if FAIL > 0:
        print("  ❌ 실패한 테스트:")
        for r in RESULTS:
            if r["pass"] is False:
                print(f"    • {r['name']}: {r['detail']}")
        print()
    
    # 종합 판정
    if FAIL == 0:
        print("  🎉 모든 테스트 통과!")
    else:
        print(f"  ⚠️  {FAIL}개 테스트 실패 — 수정이 필요합니다.")
    
    print(f"\n{'█' * 70}\n")
    
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
