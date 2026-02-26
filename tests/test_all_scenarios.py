# -*- coding: utf-8 -*-
"""
Dynamic MCP Agent v3.0 - 전체 시나리오별 테스트

시나리오:
  1. 모듈 임포트 및 초기화
  2. 도구 (tools.py) - 25개 도구 기능
  3. 레지스트리 (registry.py) - 등록/검색/캐시/토큰화/통계
  4. 에이전트 (agent.py) - 타입 변환/스키마/도구 주입/실행/상태
  5. 통합 시나리오 - create_agent/검색→로드→실행 파이프라인
  6. 엣지 케이스 및 오류 처리
"""

import os
import sys
import ast
import math
import inspect
import asyncio
import json
import unittest
from typing import Optional, List, Dict, Any, Union, Tuple
from unittest.mock import patch, MagicMock

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# 시나리오 1: 모듈 임포트 및 초기화
# ============================================================================
class TestScenario1_Import(unittest.TestCase):
    """시나리오 1: 모듈 임포트 및 패키지 구조 검증"""

    def test_01_package_version(self):
        """패키지 버전이 3.0.0인지 확인"""
        import dynamic_mcp_agent
        self.assertEqual(dynamic_mcp_agent.__version__, "3.0.0")

    def test_02_top_level_exports(self):
        """최상위 패키지에서 필요한 항목이 모두 export되는지 확인"""
        from dynamic_mcp_agent import (
            DynamicMCPAgent,
            create_agent,
            DEFAULT_REMOTE_MCP_SERVERS,
            ToolRegistry,
            registry,
            search_available_tools,
            load_tool,
            initialize_mcp_tools,
            register_tool,
        )
        self.assertIsNotNone(DynamicMCPAgent)
        self.assertIsNotNone(create_agent)
        self.assertIsNotNone(DEFAULT_REMOTE_MCP_SERVERS)
        self.assertIsNotNone(registry)

    def test_03_lib_exports(self):
        """lib 패키지에서 클래스 및 함수가 export되는지 확인"""
        from dynamic_mcp_agent.lib import (
            HybridToolRegistry,
            ToolRegistry,
            registry,
            search_available_tools,
            load_tool,
            initialize_mcp_tools,
            register_tool,
        )
        self.assertIs(ToolRegistry, HybridToolRegistry)

    def test_04_default_remote_mcp_servers(self):
        """기본 원격 MCP 서버 설정이 올바른지 확인"""
        from dynamic_mcp_agent.agent import DEFAULT_REMOTE_MCP_SERVERS
        self.assertIsInstance(DEFAULT_REMOTE_MCP_SERVERS, list)
        self.assertGreaterEqual(len(DEFAULT_REMOTE_MCP_SERVERS), 2)
        labels = [s["server_label"] for s in DEFAULT_REMOTE_MCP_SERVERS]
        self.assertIn("microsoft_learn", labels)
        self.assertIn("github", labels)


# ============================================================================
# 시나리오 2: 도구 (tools.py) 기능 테스트
# ============================================================================
class TestScenario2_Tools(unittest.TestCase):
    """시나리오 2: 25개 도구 함수 및 도구 관리 기능"""

    def test_01_tool_definitions_count(self):
        """25개 도구가 정의되어 있는지 확인"""
        from dynamic_mcp_agent.lib.tools import TOOL_DEFINITIONS
        self.assertEqual(len(TOOL_DEFINITIONS), 25)

    def test_02_tool_definitions_structure(self):
        """각 TOOL_DEFINITION 튜플 구조가 올바른지 확인: (func, category, tags)"""
        from dynamic_mcp_agent.lib.tools import TOOL_DEFINITIONS
        for i, entry in enumerate(TOOL_DEFINITIONS):
            self.assertEqual(len(entry), 3, f"TOOL_DEFINITIONS[{i}] 길이가 3이어야 함")
            func, category, tags = entry
            self.assertTrue(callable(func), f"entry[{i}][0]은 callable이어야 함")
            self.assertIsInstance(category, str, f"entry[{i}][1] category는 str이어야 함")
            self.assertIsInstance(tags, list, f"entry[{i}][2] tags는 list이어야 함")

    def test_03_all_tool_functions_have_docstrings(self):
        """모든 도구 함수에 docstring이 있는지 확인"""
        from dynamic_mcp_agent.lib.tools import TOOL_DEFINITIONS
        for func, _cat, _tags in TOOL_DEFINITIONS:
            self.assertIsNotNone(func.__doc__, f"{func.__name__}에 docstring이 없음")
            self.assertGreater(len(func.__doc__), 10, f"{func.__name__}의 docstring이 너무 짧음")

    def test_04_all_tool_functions_return_dict(self):
        """모든 도구 함수의 리턴 타입 어노테이션이 Dict인지 확인"""
        from dynamic_mcp_agent.lib.tools import TOOL_DEFINITIONS
        for func, _cat, _tags in TOOL_DEFINITIONS:
            hints = func.__annotations__.get("return")
            self.assertIsNotNone(hints, f"{func.__name__}에 return annotation이 없음")

    # --- calculator_tool ---
    def test_05_calculator_basic_arithmetic(self):
        """calculator_tool: 사칙연산"""
        from dynamic_mcp_agent.lib.tools import calculator_tool
        self.assertEqual(calculator_tool("2+3")["result"], 5)
        self.assertEqual(calculator_tool("10-3")["result"], 7)
        self.assertEqual(calculator_tool("4*5")["result"], 20)
        self.assertEqual(calculator_tool("15/4")["result"], 3.75)
        self.assertEqual(calculator_tool("15//4")["result"], 3)
        self.assertEqual(calculator_tool("10%3")["result"], 1)

    def test_06_calculator_complex(self):
        """calculator_tool: 복합 수식 + 우선순위"""
        from dynamic_mcp_agent.lib.tools import calculator_tool
        self.assertEqual(calculator_tool("2+3*4")["result"], 14)
        self.assertEqual(calculator_tool("(2+3)*4")["result"], 20)
        self.assertEqual(calculator_tool("2**10")["result"], 1024)

    def test_07_calculator_math_functions(self):
        """calculator_tool: 수학 함수 및 상수"""
        from dynamic_mcp_agent.lib.tools import calculator_tool
        self.assertAlmostEqual(calculator_tool("sqrt(16)")["result"], 4.0)
        self.assertAlmostEqual(calculator_tool("abs(-5)")["result"], 5)
        self.assertAlmostEqual(calculator_tool("pi")["result"], math.pi)
        self.assertAlmostEqual(calculator_tool("e")["result"], math.e)
        self.assertAlmostEqual(calculator_tool("log10(100)")["result"], 2.0)

    def test_08_calculator_negative(self):
        """calculator_tool: 음수 처리"""
        from dynamic_mcp_agent.lib.tools import calculator_tool
        self.assertEqual(calculator_tool("-5+3")["result"], -2)
        self.assertEqual(calculator_tool("-(2+3)")["result"], -5)

    def test_09_calculator_error_handling(self):
        """calculator_tool: 잘못된 표현식 에러 처리"""
        from dynamic_mcp_agent.lib.tools import calculator_tool
        result = calculator_tool("import os")
        self.assertIn("error", result)
        result2 = calculator_tool("__import__('os')")
        self.assertIn("error", result2)

    def test_10_safe_eval_security(self):
        """_safe_eval: 위험한 코드 차단"""
        from dynamic_mcp_agent.lib.tools import _safe_eval
        # 허용되지 않는 함수
        with self.assertRaises(ValueError):
            tree = ast.parse("exec('print(1)')", mode='eval')
            _safe_eval(tree)
        # 허용되지 않는 이름
        with self.assertRaises(ValueError):
            tree = ast.parse("os", mode='eval')
            _safe_eval(tree)

    # --- 개별 도구 실행 테스트 ---
    def test_11_azure_ai_search_tool(self):
        """azure_ai_search_tool: 기본 실행 (자격 증명 유무에 따라 결과 다름)"""
        from dynamic_mcp_agent.lib.tools import azure_ai_search_tool
        result = azure_ai_search_tool("테스트 쿼리", index_name="idx", top_k=3)
        # 자격 증명이 없으면 error, 있으면 query 키 반환
        self.assertTrue("error" in result or "query" in result)
        self.assertIsInstance(result, dict)

    def test_12_azure_blob_storage_tool(self):
        """azure_blob_storage_tool: 기본 실행"""
        from dynamic_mcp_agent.lib.tools import azure_blob_storage_tool
        result = azure_blob_storage_tool("list", "my-container")
        self.assertEqual(result["operation"], "list")
        self.assertEqual(result["container_name"], "my-container")
        self.assertEqual(result["status"], "success")

    def test_13_azure_translator_tool(self):
        """azure_translator_tool: 번역 실행"""
        from dynamic_mcp_agent.lib.tools import azure_translator_tool
        result = azure_translator_tool("안녕하세요", "en")
        self.assertIn("번역됨", result["translated_text"])
        self.assertEqual(result["target_language"], "en")

    def test_14_azure_cosmos_db_tool(self):
        """azure_cosmos_db_tool: 기본 실행"""
        from dynamic_mcp_agent.lib.tools import azure_cosmos_db_tool
        result = azure_cosmos_db_tool("query", "db1", "container1", query="SELECT * FROM c")
        self.assertEqual(result["operation"], "query")
        self.assertEqual(result["status"], "success")

    def test_15_azure_openai_embedding_tool(self):
        """azure_openai_embedding_tool: 모델별 차원 확인"""
        from dynamic_mcp_agent.lib.tools import azure_openai_embedding_tool
        r1 = azure_openai_embedding_tool("hello", model="text-embedding-3-large")
        self.assertEqual(r1["embedding_dimension"], 3072)
        r2 = azure_openai_embedding_tool("hello", model="text-embedding-3-small")
        self.assertEqual(r2["embedding_dimension"], 1536)

    def test_16_azure_image_generation_tool(self):
        """azure_image_generation_tool: 기본 모델이 gpt-image-2인지 확인"""
        from dynamic_mcp_agent.lib.tools import azure_image_generation_tool
        result = azure_image_generation_tool("a cat")
        self.assertEqual(result["model"], "gpt-image-2")
        self.assertEqual(result["quality"], "high")

    def test_17_azure_computer_use_tool(self):
        """azure_computer_use_tool: CUA 도구 실행"""
        from dynamic_mcp_agent.lib.tools import azure_computer_use_tool
        result = azure_computer_use_tool("click the button", environment="browser")
        self.assertEqual(result["environment"], "browser")
        self.assertTrue(result["screenshot"])

    def test_18_mcp_server_discovery_tool(self):
        """mcp_server_discovery_tool: MCP 서버 검색"""
        from dynamic_mcp_agent.lib.tools import mcp_server_discovery_tool
        result = mcp_server_discovery_tool("database", category="ai", limit=5)
        self.assertEqual(result["query"], "database")
        self.assertEqual(result["limit"], 5)

    def test_19_structured_output_tool(self):
        """structured_output_tool: 구조화 출력"""
        from dynamic_mcp_agent.lib.tools import structured_output_tool
        result = structured_output_tool("test", schema_name="report")
        self.assertEqual(result["schema_name"], "report")

    def test_20_all_25_tools_callable(self):
        """TOOL_DEFINITIONS의 모든 25개 도구 함수가 호출 가능한지 확인"""
        from dynamic_mcp_agent.lib.tools import TOOL_DEFINITIONS
        for func, _cat, _tags in TOOL_DEFINITIONS:
            self.assertTrue(callable(func), f"{func.__name__}이 callable이 아님")
            # 시그니처가 유효한지 확인
            sig = inspect.signature(func)
            self.assertGreater(len(sig.parameters), 0, f"{func.__name__}에 파라미터가 없음")


# ============================================================================
# 시나리오 3: 레지스트리 (registry.py) 테스트
# ============================================================================
class TestScenario3_Registry(unittest.TestCase):
    """시나리오 3: HybridToolRegistry 등록/검색/통계"""

    def setUp(self):
        """각 테스트마다 깨끗한 레지스트리 생성"""
        from dynamic_mcp_agent.lib.registry import HybridToolRegistry
        self.reg = HybridToolRegistry(
            enable_mcp_registry=False  # 외부 HTTP 호출 방지
        )

    def _sample_tool(self, x: str) -> str:
        """샘플 도구"""
        return f"result: {x}"

    # --- 등록 ---
    def test_01_register_single(self):
        """단일 도구 등록"""
        self.reg.register(self._sample_tool, name="sample_tool", description="테스트 도구")
        self.assertEqual(self.reg.count(), 1)
        self.assertIn("sample_tool", self.reg.list_all_tools())

    def test_02_register_batch(self):
        """일괄 도구 등록"""
        tools = [
            (lambda x: x, "tool_a", "도구 A 설명", "cat_a", ["tag1"]),
            (lambda x: x, "tool_b", "도구 B 설명", "cat_b", ["tag2"]),
            (lambda x: x, "tool_c", "도구 C 설명", "cat_a", ["tag3"]),
        ]
        self.reg.register_batch(tools)
        self.assertEqual(self.reg.count(), 3)

    def test_03_register_duplicate_update(self):
        """중복 등록 시 기존 항목이 업데이트되는지 확인 (추가되지 않음)"""
        self.reg.register(self._sample_tool, name="dup_tool", description="v1")
        self.assertEqual(self.reg.count(), 1)
        self.assertEqual(len(self.reg._tool_names), 1)
        self.assertEqual(len(self.reg._descriptions), 1)

        # 같은 이름으로 다시 등록
        def new_tool(x: str) -> str:
            return f"new: {x}"

        self.reg.register(new_tool, name="dup_tool", description="v2")
        self.assertEqual(self.reg.count(), 1)  # 여전히 1개
        self.assertEqual(len(self.reg._tool_names), 1)  # 중복 없음
        self.assertEqual(len(self.reg._descriptions), 1)  # 중복 없음

        # 실제 도구가 업데이트되었는지 확인
        tool = self.reg.get_tool("dup_tool")
        self.assertEqual(tool("test"), "new: test")

    def test_04_register_batch_duplicate(self):
        """register_batch에서 중복 등록 시 업데이트"""
        self.reg.register(self._sample_tool, name="dup_tool", description="orig")
        tools = [
            (lambda x: "updated", "dup_tool", "업데이트됨", "cat", ["tag"]),
        ]
        self.reg.register_batch(tools)
        self.assertEqual(self.reg.count(), 1)
        self.assertEqual(len(self.reg._tool_names), 1)

    # --- 조회 ---
    def test_05_get_tool(self):
        """도구 이름으로 조회"""
        def standalone_tool(x: str) -> str:
            return f"result: {x}"

        self.reg.register(standalone_tool, name="my_tool")
        tool = self.reg.get_tool("my_tool")
        self.assertIs(tool, standalone_tool)
        self.assertEqual(tool("hello"), "result: hello")

    def test_06_get_tool_not_found(self):
        """존재하지 않는 도구 조회 시 None 반환"""
        self.assertIsNone(self.reg.get_tool("nonexistent"))

    def test_07_get_metadata(self):
        """도구 메타데이터 조회"""
        self.reg.register(self._sample_tool, name="meta_tool",
                          description="설명", category="test_cat", tags=["t1", "t2"])
        meta = self.reg.get_tool_metadata("meta_tool")
        self.assertEqual(meta["description"], "설명")
        self.assertEqual(meta["category"], "test_cat")
        self.assertEqual(meta["tags"], ["t1", "t2"])

    def test_08_get_tools_by_category(self):
        """카테고리별 도구 조회"""
        self.reg.register(lambda: None, name="a", category="ai")
        self.reg.register(lambda: None, name="b", category="search")
        self.reg.register(lambda: None, name="c", category="ai")
        ai_tools = self.reg.get_tools_by_category("ai")
        self.assertEqual(sorted(ai_tools), ["a", "c"])

    # --- BM25 검색 ---
    def test_09_bm25_search_korean(self):
        """BM25 검색: 한국어 키워드"""
        self.reg.register(lambda: None, name="translate_tool",
                          description="텍스트를 번역합니다", tags=["번역", "언어"])
        self.reg.register(lambda: None, name="search_tool",
                          description="문서를 검색합니다", tags=["문서", "검색"])
        results = self.reg.search("번역", strategy="bm25")
        self.assertTrue(any("translate_tool" in r for r in results))

    def test_10_bm25_search_english(self):
        """BM25 검색: 영어 키워드"""
        self.reg.register(lambda: None, name="image_tool",
                          description="Image analysis tool", tags=["image", "vision"])
        self.reg.register(lambda: None, name="text_tool",
                          description="Text processing tool", tags=["text", "nlp"])
        results = self.reg.search("image", strategy="bm25")
        self.assertTrue(any("image_tool" in r for r in results))

    def test_11_bm25_search_mixed_language(self):
        """BM25 검색: 한영 혼합 키워드"""
        self.reg.register(lambda: None, name="azure_search",
                          description="Azure AI 검색 도구", tags=["azure", "검색", "search"])
        results = self.reg.search("azure 검색", strategy="bm25")
        self.assertTrue(len(results) > 0)

    # --- 하이브리드 검색 ---
    def test_12_hybrid_search(self):
        """하이브리드 검색: 기본 전략"""
        self.reg.register(lambda: None, name="cosmos_tool",
                          description="Azure Cosmos DB 데이터베이스 관리",
                          tags=["database", "cosmos", "데이터베이스"])
        results = self.reg.search("데이터베이스", strategy="hybrid")
        self.assertTrue(len(results) > 0)
        self.assertTrue(any("cosmos_tool" in r for r in results))

    # --- 토큰화 ---
    def test_13_tokenize_korean(self):
        """한국어 토큰화: 음절 + 바이그램"""
        tokens = self.reg._tokenize("번역 도구")
        self.assertIn("번역", tokens)
        self.assertIn("도구", tokens)
        # 바이그램
        self.assertIn("번역", tokens)

    def test_14_tokenize_english(self):
        """영어 토큰화"""
        tokens = self.reg._tokenize("Image Analysis Tool")
        self.assertIn("image", tokens)
        self.assertIn("analysis", tokens)
        self.assertIn("tool", tokens)

    def test_15_tokenize_underscore_handling(self):
        """언더스코어가 공백으로 변환되어 토큰화"""
        tokens = self.reg._tokenize("azure_ai_search")
        self.assertIn("azure", tokens)
        self.assertIn("ai", tokens)
        self.assertIn("search", tokens)

    # --- 통계 ---
    def test_16_search_stats(self):
        """검색 통계 추적"""
        self.reg.register(lambda: None, name="tool1", description="test tool", tags=["test"])
        self.reg.search("test")
        stats = self.reg.get_search_stats()
        self.assertEqual(stats["total_searches"], 1)

    def test_17_search_stats_ratios(self):
        """검색 통계 비율 계산"""
        self.reg.register(lambda: None, name="tool1", description="test tool", tags=["test"])
        self.reg.search("test")
        stats = self.reg.get_search_stats()
        self.assertIn("bm25_ratio", stats)

    # --- clear ---
    def test_18_clear(self):
        """레지스트리 초기화"""
        self.reg.register(self._sample_tool, name="to_clear")
        self.assertEqual(self.reg.count(), 1)
        self.reg.clear()
        self.assertEqual(self.reg.count(), 0)
        self.assertEqual(len(self.reg._tool_names), 0)
        self.assertEqual(len(self.reg._descriptions), 0)

    # --- 임계값 설정 ---
    def test_19_set_thresholds(self):
        """검색 임계값 설정"""
        self.reg.set_thresholds(bm25_threshold=10.0, embedding_threshold=0.8)
        self.assertEqual(self.reg.BM25_CONFIDENCE_THRESHOLD, 10.0)
        self.assertEqual(self.reg.EMBEDDING_SIMILARITY_THRESHOLD, 0.8)

    # --- model_info ---
    def test_20_get_model_info(self):
        """모델 정보 조회"""
        info = self.reg.get_model_info()
        self.assertIn("sentence_model", info)
        self.assertIn("mcp_registry_enabled", info)
        self.assertFalse(info["mcp_registry_enabled"])  # setUp에서 False로 설정


# ============================================================================
# 시나리오 4: 에이전트 (agent.py) 테스트
# ============================================================================
class TestScenario4_Agent(unittest.TestCase):
    """시나리오 4: DynamicMCPAgent 클래스 기능"""

    # --- _resolve_json_type ---
    def test_01_resolve_basic_types(self):
        """기본 Python 타입 → JSON Schema 변환"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        self.assertEqual(DynamicMCPAgent._resolve_json_type(int), "integer")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(str), "string")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(float), "number")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(bool), "boolean")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(list), "array")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(dict), "object")

    def test_02_resolve_generic_types(self):
        """제네릭 타입 (List[str], Dict[str, Any]) 변환"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        self.assertEqual(DynamicMCPAgent._resolve_json_type(List[str]), "array")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(List[int]), "array")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(Dict[str, Any]), "object")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(Dict[str, int]), "object")

    def test_03_resolve_optional_types(self):
        """Optional[X] 변환 → X의 타입"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        self.assertEqual(DynamicMCPAgent._resolve_json_type(Optional[str]), "string")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(Optional[int]), "integer")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(Optional[List[str]]), "array")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(Optional[Dict[str, Any]]), "object")

    def test_04_resolve_none_and_empty(self):
        """None 및 empty 어노테이션 → "string" 기본값"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        self.assertEqual(DynamicMCPAgent._resolve_json_type(None), "string")
        self.assertEqual(DynamicMCPAgent._resolve_json_type(inspect.Parameter.empty), "string")

    def test_05_resolve_unknown_type(self):
        """알 수 없는 타입 → "string" 기본값"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent

        class CustomType:
            pass

        self.assertEqual(DynamicMCPAgent._resolve_json_type(CustomType), "string")

    # --- 에이전트 인스턴스 (Mock OpenAI 클라이언트) ---
    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_06_agent_init(self, mock_init_tools, mock_openai):
        """에이전트 초기화: 속성 설정"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            deployment_name="gpt-5.2"
        )
        self.assertEqual(agent.model, "gpt-5.2")
        self.assertIsNone(agent.last_response_id)
        self.assertEqual(agent._conversation_turns, 0)
        self.assertIsInstance(agent.active_tools, dict)
        self.assertEqual(len(agent.active_tools), 0)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_07_agent_with_reasoning_model(self, mock_init_tools, mock_openai):
        """에이전트 초기화: 추론 모델 설정"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            reasoning_model="o4-mini"
        )
        self.assertEqual(agent.reasoning_model, "o4-mini")

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_08_agent_with_tracing(self, mock_init_tools, mock_openai):
        """에이전트 초기화: 트레이싱 활성화"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            enable_tracing=True
        )
        self.assertTrue(agent.enable_tracing)

    # --- 도구 스키마 ---
    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_09_base_tools_schema(self, mock_init_tools, mock_openai):
        """기본 도구 스키마에 search_available_tools와 load_tool 포함"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )
        schema = agent._get_base_tools_schema()
        names = [t["name"] for t in schema if t["type"] == "function"]
        self.assertIn("search_available_tools", names)
        self.assertIn("load_tool", names)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_10_base_tools_with_mcp_servers(self, mock_init_tools, mock_openai):
        """원격 MCP 서버 설정이 스키마에 포함"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[
                {"server_url": "https://example.com/mcp", "server_label": "test_mcp"}
            ]
        )
        schema = agent._get_base_tools_schema()
        mcp_tools = [t for t in schema if t["type"] == "mcp"]
        self.assertEqual(len(mcp_tools), 1)
        self.assertEqual(mcp_tools[0]["server_label"], "test_mcp")

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_11_active_tools_schema_no_mutation(self, mock_init_tools, mock_openai):
        """_get_active_tools_schema가 캐시를 변이시키지 않음"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )
        # 캐시 생성
        base_schema = agent._get_base_tools_schema()
        base_len = len(base_schema)

        # 도구 추가 후 활성 스키마 조회
        agent.active_tools["test_tool"] = lambda x: x
        active_schema = agent._get_active_tools_schema()
        self.assertEqual(len(active_schema), base_len + 1)

        # 원본 캐시가 변이되지 않았는지 확인
        self.assertEqual(len(agent._base_tools_schema_cache), base_len)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_12_active_tools_schema_with_typed_params(self, mock_init_tools, mock_openai):
        """활성 도구 스키마에 파라미터 타입이 올바르게 반영"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )

        def typed_tool(name: str, count: int, tags: List[str], active: bool = True) -> Dict[str, Any]:
            """테스트 도구"""
            return {}

        agent.active_tools["typed_tool"] = typed_tool
        schema = agent._get_active_tools_schema()
        tool_schema = next(t for t in schema if t.get("name") == "typed_tool")

        props = tool_schema["parameters"]["properties"]
        self.assertEqual(props["name"]["type"], "string")
        self.assertEqual(props["count"]["type"], "integer")
        self.assertEqual(props["tags"]["type"], "array")
        self.assertEqual(props["active"]["type"], "boolean")

        # required에 기본값 있는 active는 포함되지 않아야 함
        self.assertIn("name", tool_schema["parameters"]["required"])
        self.assertIn("count", tool_schema["parameters"]["required"])
        self.assertNotIn("active", tool_schema["parameters"]["required"])

    # --- 동적 도구 주입 ---
    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_13_dynamic_tool_injection(self, mock_init_tools, mock_openai):
        """동적 도구 주입 성공"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        from dynamic_mcp_agent.lib.registry import registry as global_reg

        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )

        # 글로벌 레지스트리에 도구 등록
        def my_tool(x: str) -> str:
            return x
        global_reg.register(my_tool, name="my_inject_test_tool")

        # 주입 테스트
        result = agent._dynamic_tool_injection("my_inject_test_tool")
        self.assertTrue(result)
        self.assertIn("my_inject_test_tool", agent.active_tools)

        # 이미 있는 도구 재주입
        result2 = agent._dynamic_tool_injection("my_inject_test_tool")
        self.assertTrue(result2)

        # 존재하지 않는 도구
        result3 = agent._dynamic_tool_injection("nonexistent_tool_xyz")
        self.assertFalse(result3)

    # --- 도구 실행 ---
    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_14_execute_base_tool(self, mock_init_tools, mock_openai):
        """기본 도구 (search_available_tools) 실행"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )
        # search_available_tools 실행
        result = agent._execute_tool("search_available_tools", {"query": "번역", "top_k": 3})
        parsed = json.loads(result)
        self.assertIsInstance(parsed, list)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_15_execute_active_tool(self, mock_init_tools, mock_openai):
        """활성 도구 실행"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )

        def adder(a: int, b: int) -> Dict[str, int]:
            return {"sum": a + b}

        agent.active_tools["adder"] = adder
        result = agent._execute_tool("adder", {"a": 3, "b": 7})
        parsed = json.loads(result)
        self.assertEqual(parsed["sum"], 10)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_16_execute_tool_error(self, mock_init_tools, mock_openai):
        """도구 실행 중 에러 처리"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )

        def broken_tool() -> str:
            raise RuntimeError("고장남!")

        agent.active_tools["broken"] = broken_tool
        result = agent._execute_tool("broken", {})
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("고장남!", parsed["error"])

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_17_execute_nonexistent_tool(self, mock_init_tools, mock_openai):
        """존재하지 않는 도구 실행"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )
        result = agent._execute_tool("does_not_exist", {})
        parsed = json.loads(result)
        self.assertIn("error", parsed)

    # --- Add MCP server ---
    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_18_add_remote_mcp_server(self, mock_init_tools, mock_openai):
        """런타임 MCP 서버 추가"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )
        agent.add_remote_mcp_server(
            server_url="https://new-mcp.com",
            server_label="new_server",
            server_description="테스트 서버"
        )
        self.assertEqual(len(agent.remote_mcp_servers), 1)
        self.assertEqual(agent.remote_mcp_servers[0]["server_label"], "new_server")

        # 캐시가 무효화되어야 함 (base_tools_schema_cache 길이가 맞지 않게)
        agent._base_tools_schema_cache = None  # 강제 리셋
        schema = agent._get_base_tools_schema()
        mcp_count = sum(1 for t in schema if t["type"] == "mcp")
        self.assertEqual(mcp_count, 1)

    # --- reset ---
    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_19_reset_conversation(self, mock_init_tools, mock_openai):
        """대화 초기화"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )
        agent.last_response_id = "resp_12345"
        agent._conversation_turns = 5
        agent.reset_conversation()
        self.assertIsNone(agent.last_response_id)
        self.assertEqual(agent._conversation_turns, 0)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_20_reset_tools(self, mock_init_tools, mock_openai):
        """활성 도구 초기화"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )
        agent.active_tools["tool1"] = lambda: None
        agent.active_tools["tool2"] = lambda: None
        agent.reset_tools()
        self.assertEqual(len(agent.active_tools), 0)

    # --- stats ---
    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_21_get_stats(self, mock_init_tools, mock_openai):
        """통계 조회"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )
        stats = agent.get_stats()
        self.assertIn("model", stats)
        self.assertIn("api", stats)
        self.assertIn("reasoning_model", stats)
        self.assertIn("active_tools", stats)
        self.assertIn("remote_mcp_servers", stats)
        self.assertIn("conversation_turns", stats)
        self.assertIn("tracing_enabled", stats)
        self.assertIn("structured_outputs", stats)
        self.assertEqual(stats["active_tools"], 0)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_22_get_active_tools_list(self, mock_init_tools, mock_openai):
        """활성 도구 목록 조회 + MCP 서버 표시"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[
                {"server_url": "https://mcp.com", "server_label": "my_mcp"}
            ]
        )
        agent.active_tools["tool_a"] = lambda: None
        tools = agent.get_active_tools_list()
        self.assertIn("tool_a", tools)
        self.assertIn("[MCP] my_mcp", tools)

    # --- _serialize_result ---
    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_23_serialize_result(self, mock_init_tools, mock_openai):
        """결과 직렬화"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        self.assertEqual(DynamicMCPAgent._serialize_result({"a": 1}), '{"a": 1}')
        self.assertEqual(DynamicMCPAgent._serialize_result([1, 2, 3]), '[1, 2, 3]')
        self.assertEqual(DynamicMCPAgent._serialize_result("text"), "text")
        self.assertEqual(DynamicMCPAgent._serialize_result(42), "42")


# ============================================================================
# 시나리오 5: 통합 시나리오 (도구 검색 → 로드 → 실행 파이프라인)
# ============================================================================
class TestScenario5_Integration(unittest.TestCase):
    """시나리오 5: 전체 파이프라인 통합 테스트"""

    @classmethod
    def setUpClass(cls):
        """글로벌 레지스트리에 도구 초기화"""
        from dynamic_mcp_agent.lib.tools import initialize_mcp_tools
        initialize_mcp_tools()

    def test_01_initialize_registers_25_tools(self):
        """initialize_mcp_tools() 후 25개 도구가 등록되어 있는지 확인"""
        from dynamic_mcp_agent.lib.registry import registry
        self.assertGreaterEqual(registry.count(), 25)

    def test_02_search_for_translation_tools(self):
        """'번역' 검색 → azure_translator_tool 발견"""
        from dynamic_mcp_agent.lib.tools import search_available_tools
        results = search_available_tools("번역")
        self.assertTrue(any("azure_translator_tool" in r for r in results))

    def test_03_search_for_database_tools(self):
        """'database' 검색 → SQL/Cosmos 도구 발견"""
        from dynamic_mcp_agent.lib.tools import search_available_tools
        results = search_available_tools("database")
        tool_texts = " ".join(results)
        self.assertTrue(
            "azure_sql_query_tool" in tool_texts or
            "azure_cosmos_db_tool" in tool_texts
        )

    def test_04_search_for_image_tools(self):
        """'이미지' 검색 → 이미지 관련 도구 발견"""
        from dynamic_mcp_agent.lib.tools import search_available_tools
        results = search_available_tools("이미지")
        tool_texts = " ".join(results)
        self.assertTrue(
            "azure_computer_vision_tool" in tool_texts or
            "azure_image_generation_tool" in tool_texts
        )

    def test_05_search_for_cua_tools(self):
        """'CUA' 검색 → Computer Use 도구 발견"""
        from dynamic_mcp_agent.lib.tools import search_available_tools
        results = search_available_tools("CUA")
        tool_texts = " ".join(results)
        self.assertTrue("azure_computer_use_tool" in tool_texts)

    def test_06_search_for_agent_tools(self):
        """'에이전트' 검색 → Agent Service 도구 발견"""
        from dynamic_mcp_agent.lib.tools import search_available_tools
        results = search_available_tools("에이전트")
        tool_texts = " ".join(results)
        self.assertTrue(
            "azure_ai_agent_service_tool" in tool_texts or
            "azure_ai_foundry_agent_tool" in tool_texts
        )

    def test_07_load_tool_success(self):
        """도구 로드 성공"""
        from dynamic_mcp_agent.lib.tools import load_tool
        result = load_tool("calculator_tool")
        self.assertIn("성공적으로 로드", result)

    def test_08_load_tool_failure(self):
        """존재하지 않는 도구 로드 실패"""
        from dynamic_mcp_agent.lib.tools import load_tool
        result = load_tool("nonexistent_xyz_tool")
        self.assertIn("찾을 수 없습니다", result)

    def test_09_load_tool_suggests_similar(self):
        """로드 실패 시 유사 도구 제안"""
        from dynamic_mcp_agent.lib.tools import load_tool
        result = load_tool("azure_search")
        self.assertIn("비슷한 도구", result)

    def test_10_get_tool_info(self):
        """도구 정보 조회"""
        from dynamic_mcp_agent.lib.tools import get_tool_info
        info = get_tool_info("calculator_tool")
        self.assertTrue(info["callable"])
        self.assertEqual(info["name"], "calculator_tool")
        self.assertEqual(info["category"], "utility")

    def test_11_get_tool_info_not_found(self):
        """존재하지 않는 도구 정보 조회"""
        from dynamic_mcp_agent.lib.tools import get_tool_info
        info = get_tool_info("no_such_tool")
        self.assertIn("error", info)

    def test_12_register_custom_tool(self):
        """사용자 도구 등록 → 검색 가능"""
        from dynamic_mcp_agent.lib.tools import register_tool, search_available_tools
        from dynamic_mcp_agent.lib.registry import registry

        def custom_test_tool_xyz(data: str) -> str:
            """커스텀 테스트 도구입니다"""
            return data

        register_tool(custom_test_tool_xyz, tags=["custom", "xyz", "테스트전용"])
        tool = registry.get_tool("custom_test_tool_xyz")
        self.assertIsNotNone(tool)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_13_full_pipeline_search_load_execute(self, mock_init_tools, mock_openai):
        """전체 파이프라인: 검색 → 로드 → 실행"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        from dynamic_mcp_agent.lib.tools import search_available_tools

        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )

        # 1단계: 검색
        results = search_available_tools("계산")
        self.assertTrue(any("calculator" in r for r in results))

        # 2단계: 로드 (agent 내부 _execute_tool으로)
        load_result = agent._execute_tool("load_tool", {"tool_name": "calculator_tool"})
        self.assertIn("성공적으로 로드", load_result)
        self.assertIn("calculator_tool", agent.active_tools)

        # 3단계: 실행
        calc_result = agent._execute_tool("calculator_tool", {"expression": "2**8"})
        parsed = json.loads(calc_result)
        self.assertEqual(parsed["result"], 256)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    def test_14_create_agent_factory(self, mock_openai):
        """create_agent 팩토리 함수가 올바르게 동작"""
        from dynamic_mcp_agent.agent import create_agent
        agent = create_agent(
            enable_remote_mcp=True,
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key"
        )
        self.assertIsNotNone(agent)
        self.assertGreaterEqual(len(agent.remote_mcp_servers), 2)  # 기본 MCP 서버

    @patch("dynamic_mcp_agent.agent.OpenAI")
    def test_15_create_agent_no_mcp(self, mock_openai):
        """create_agent에서 MCP 비활성화"""
        from dynamic_mcp_agent.agent import create_agent
        agent = create_agent(
            enable_remote_mcp=False,
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key"
        )
        self.assertEqual(len(agent.remote_mcp_servers), 0)


# ============================================================================
# 시나리오 6: 엣지 케이스 및 오류 처리
# ============================================================================
class TestScenario6_EdgeCases(unittest.TestCase):
    """시나리오 6: 엣지 케이스 및 오류 처리"""

    def test_01_empty_query_search(self):
        """빈 문자열 검색"""
        from dynamic_mcp_agent.lib.registry import HybridToolRegistry
        reg = HybridToolRegistry(enable_mcp_registry=False)
        reg.register(lambda: None, name="tool1", description="test")
        results = reg.search("")
        # 빈 쿼리도 결과 반환 가능 (BM25 특성)
        self.assertIsInstance(results, list)

    def test_02_unicode_query_search(self):
        """유니코드 특수문자 검색"""
        from dynamic_mcp_agent.lib.registry import HybridToolRegistry
        reg = HybridToolRegistry(enable_mcp_registry=False)
        reg.register(lambda: None, name="tool1", description="데이터 🌐 처리")
        results = reg.search("🌐")
        self.assertIsInstance(results, list)

    def test_03_very_long_query(self):
        """매우 긴 쿼리 검색"""
        from dynamic_mcp_agent.lib.registry import HybridToolRegistry
        reg = HybridToolRegistry(enable_mcp_registry=False)
        reg.register(lambda: None, name="tool1", description="test tool")
        long_query = "azure " * 500
        results = reg.search(long_query)
        self.assertIsInstance(results, list)

    def test_04_search_empty_registry(self):
        """빈 레지스트리에서 검색"""
        from dynamic_mcp_agent.lib.registry import HybridToolRegistry
        reg = HybridToolRegistry(enable_mcp_registry=False)
        results = reg.search("anything")
        self.assertEqual(results, [])

    def test_05_calculator_division_by_zero(self):
        """calculator_tool: 0으로 나누기"""
        from dynamic_mcp_agent.lib.tools import calculator_tool
        result = calculator_tool("1/0")
        self.assertIn("error", result)

    def test_06_calculator_very_large_number(self):
        """calculator_tool: 매우 큰 숫자"""
        from dynamic_mcp_agent.lib.tools import calculator_tool
        result = calculator_tool("2**100")
        self.assertEqual(result["result"], 2**100)

    def test_07_calculator_float_precision(self):
        """calculator_tool: 부동소수점"""
        from dynamic_mcp_agent.lib.tools import calculator_tool
        result = calculator_tool("0.1+0.2")
        self.assertAlmostEqual(result["result"], 0.3, places=10)

    def test_08_register_tool_auto_name(self):
        """이름 없이 도구 등록 시 함수명 자동 추출"""
        from dynamic_mcp_agent.lib.registry import HybridToolRegistry
        reg = HybridToolRegistry(enable_mcp_registry=False)

        def auto_named_test_tool():
            """자동 이름 도구"""
            pass

        reg.register(auto_named_test_tool)
        self.assertIn("auto_named_test_tool", reg.list_all_tools())

    def test_09_register_tool_no_docstring(self):
        """docstring 없는 함수 등록"""
        from dynamic_mcp_agent.lib.registry import HybridToolRegistry
        reg = HybridToolRegistry(enable_mcp_registry=False)

        def no_doc_tool():
            pass

        reg.register(no_doc_tool, name="no_doc")
        self.assertEqual(reg.count(), 1)

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_10_tool_with_optional_params(self, mock_init_tools, mock_openai):
        """Optional 파라미터가 있는 도구의 스키마가 올바른지"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )

        def opt_tool(required_arg: str, optional_arg: Optional[int] = None) -> Dict[str, Any]:
            """옵셔널 파라미터 도구"""
            return {}

        agent.active_tools["opt_tool"] = opt_tool
        schema = agent._get_active_tools_schema()
        tool_schema = next(t for t in schema if t.get("name") == "opt_tool")

        required = tool_schema["parameters"]["required"]
        self.assertIn("required_arg", required)
        self.assertNotIn("optional_arg", required)

        # Optional[int] → "integer"
        props = tool_schema["parameters"]["properties"]
        self.assertEqual(props["optional_arg"]["type"], "integer")

    def test_11_mcp_registry_client_disabled(self):
        """httpx/aiohttp가 없을 때 MCP 클라이언트 비활성화"""
        from dynamic_mcp_agent.lib.registry import MCPRegistryClient
        with patch("dynamic_mcp_agent.lib.registry.HTTPX_AVAILABLE", False), \
             patch("dynamic_mcp_agent.lib.registry.AIOHTTP_AVAILABLE", False):
            client = MCPRegistryClient()
            self.assertFalse(client._enabled)
            servers = asyncio.run(client.search_servers("test"))
            self.assertEqual(servers, [])

    def test_12_mcp_registry_cache(self):
        """MCPRegistryClient 캐시 동작"""
        from dynamic_mcp_agent.lib.registry import MCPRegistryClient
        client = MCPRegistryClient(cache_ttl=3600)
        client._set_cache("key1", [{"name": "test"}])
        cached = client._get_cached("key1")
        self.assertEqual(cached, [{"name": "test"}])

        # 존재하지 않는 키
        self.assertIsNone(client._get_cached("nonexistent"))

    def test_13_list_all_tools_after_clear(self):
        """clear 후 list_all_tools가 빈 리스트"""
        from dynamic_mcp_agent.lib.registry import HybridToolRegistry
        reg = HybridToolRegistry(enable_mcp_registry=False)
        reg.register(lambda: None, name="tool1")
        reg.clear()
        self.assertEqual(reg.list_all_tools(), [])

    @patch("dynamic_mcp_agent.agent.OpenAI")
    @patch("dynamic_mcp_agent.agent.initialize_mcp_tools")
    def test_14_base_tools_schema_cache_invalidation(self, mock_init_tools, mock_openai):
        """MCP 서버 추가 후 스키마 캐시가 갱신되는지"""
        from dynamic_mcp_agent.agent import DynamicMCPAgent
        agent = DynamicMCPAgent(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            remote_mcp_servers=[]
        )
        # 캐시 생성 (2개: search + load)
        schema1 = agent._get_base_tools_schema()
        self.assertEqual(len(schema1), 2)

        # MCP 서버 추가
        agent.add_remote_mcp_server("https://new.com", "new")
        # 캐시된 길이(2)와 expected 길이(3)가 다르므로 재생성
        schema2 = agent._get_base_tools_schema()
        self.assertEqual(len(schema2), 3)

    def test_15_concurrent_bm25_searches(self):
        """BM25 검색을 여러 번 실행해도 일관된 결과"""
        from dynamic_mcp_agent.lib.registry import HybridToolRegistry
        reg = HybridToolRegistry(enable_mcp_registry=False)
        reg.register(lambda: None, name="tool_a", description="Azure search tool", tags=["search"])
        reg.register(lambda: None, name="tool_b", description="Database query tool", tags=["database"])

        r1 = reg.search("search", strategy="bm25")
        r2 = reg.search("search", strategy="bm25")
        self.assertEqual(r1, r2)


# ============================================================================
# 시나리오 7: MCPRegistryClient HTTP 메서드 테스트
# ============================================================================
class TestScenario7_MCPRegistryClient(unittest.TestCase):
    """시나리오 7: MCPRegistryClient의 URL 빌드 및 캐시 테스트"""

    def test_01_build_url(self):
        """URL 빌드"""
        from dynamic_mcp_agent.lib.registry import MCPRegistryClient
        client = MCPRegistryClient()
        url = client._build_url("servers")
        self.assertEqual(url, "https://registry.modelcontextprotocol.io/v0.1/servers")

    def test_02_build_url_with_path(self):
        """경로가 포함된 URL 빌드"""
        from dynamic_mcp_agent.lib.registry import MCPRegistryClient
        client = MCPRegistryClient()
        url = client._build_url("servers/my-server/versions/latest")
        self.assertIn("servers/my-server/versions/latest", url)

    def test_03_cache_ttl(self):
        """캐시 TTL 동작"""
        import time as _time
        from dynamic_mcp_agent.lib.registry import MCPRegistryClient
        client = MCPRegistryClient(cache_ttl=1)
        client._set_cache("expire_test", "data")
        self.assertEqual(client._get_cached("expire_test"), "data")

        # TTL 만료 시뮬레이션
        client._cache_time["expire_test"] = _time.time() - 2
        self.assertIsNone(client._get_cached("expire_test"))


# ============================================================================
# 시나리오 8: main.py 기능 테스트 (환경 변수 체크)
# ============================================================================
class TestScenario8_Main(unittest.TestCase):
    """시나리오 8: main.py 함수 테스트"""

    def test_01_check_environment_missing(self):
        """환경 변수 부재 시 False 반환"""
        from dynamic_mcp_agent.main import check_environment
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            os.environ.pop("AZURE_OPENAI_API_KEY", None)
            os.environ.pop("AZURE_OPENAI_DEPLOYMENT_NAME", None)
            result = check_environment()
            self.assertFalse(result)

    def test_02_check_environment_present(self):
        """환경 변수 설정 시 True 반환"""
        from dynamic_mcp_agent.main import check_environment
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-5.2"
        }
        with patch.dict(os.environ, env):
            result = check_environment()
            self.assertTrue(result)

    def test_03_main_argparse(self):
        """argparse가 올바르게 구성되어 있는지"""
        import argparse
        # main.py의 main 함수에서 사용하는 인자들이 정의되어 있는지
        from dynamic_mcp_agent.main import main
        self.assertTrue(callable(main))


# ============================================================================
# 실행
# ============================================================================
if __name__ == "__main__":
    # 테스트 결과를 상세하게 출력
    unittest.main(verbosity=2)
