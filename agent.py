# -*- coding: utf-8 -*-
"""
Dynamic MCP Agent with Azure OpenAI (v1 Responses API + Agents SDK)

이 모듈은 Azure OpenAI v1 Responses API를 사용하여 동적 도구 검색 및 로딩 기능을
갖춘 에이전트를 구현합니다. LLM에게 "검색 능력"을 부여하여 수많은 MCP 서버 중
현재 태스크에 적합한 도구만 동적으로 로딩합니다.

v3.0.0 업데이트 (2026-02-26):
- [NEW] OpenAI Agents SDK 통합 - 멀티 에이전트 오케스트레이션 (--agents 모드)
- [NEW] Structured Outputs 지원 (Pydantic v2 스키마 기반 응답)
- [NEW] GPT-5.2 네이티브 추론 지원 (--reasoning 모드, 별도 추론 모델 불필요)
- [NEW] 에이전트 트레이싱/관찰성 지원
- [NEW] 내장 도구 (web_search, file_search, code_interpreter) 네이티브 통합
- [CHANGED] 기본 모델 gpt-5 → gpt-5.2
- [CHANGED] 스트리밍에 도구 호출 루프 처리 추가

v2.0.0 업데이트 (2026-02-07):
- [BREAKING] AzureOpenAI 클라이언트 → OpenAI + base_url 방식으로 전환
- [BREAKING] chat.completions.create() → responses.create() API 전환
- [BREAKING] conversation_history 리스트 제거 → previous_response_id 자동 체이닝
- [NEW] 네이티브 원격 MCP 서버 도구 통합 (type: "mcp")
- [NEW] chat_stream() 스트리밍 응답 메서드 추가
- [NEW] add_remote_mcp_server() 런타임 MCP 서버 추가 메서드
- [CHANGED] 기본 모델 gpt-4o → gpt-5

핵심 아키텍처:
1. Tool Search Tool: 사용 가능한 도구를 BM25 + RRF로 검색
2. Tool Load Tool: 선택된 도구를 동적으로 컨텍스트에 주입
3. Dynamic Callback: 도구 로드 후 자동으로 에이전트에 도구 추가
4. Remote MCP Server: 네이티브 MCP 서버 도구 연동
5. Agents SDK: 멀티 에이전트 오케스트레이션 (선택적)

참고:
- https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses
- https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle
- https://openai.github.io/openai-agents-python/
"""

import os
import json
import inspect
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, get_origin, get_args, Union

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

from openai import OpenAI

from .lib.registry import registry
from .lib.tools import (
    search_available_tools,
    load_tool,
    initialize_mcp_tools,
)

logger = logging.getLogger(__name__)


class DynamicMCPAgent:
    """
    동적 도구 검색 및 로딩 기능을 갖춘 Azure OpenAI 에이전트
    (v1 Responses API 기반)

    이 에이전트는 수백 개의 MCP 도구 중에서 필요한 도구만 동적으로
    로딩하여 토큰 비용을 절감하고 추론 정확도를 향상시킵니다.

    2026 최신 기능 (v3.0):
    - Responses API: 상태 기반 API로 대화 체이닝 자동 관리
    - v1 API: 최신 기능에 자동 접근, 버전 관리 불필요
    - 네이티브 MCP 서버 도구: 원격 MCP 서버 직접 연동
    - GPT-5.2 시리즈 지원 (기본 모델)
    - GPT-5.2 네이티브 추론 지원 (수학·논리·코드 분석, 별도 추론 모델 불필요)
    - OpenAI Agents SDK 멀티 에이전트 오케스트레이션
    - Structured Outputs (Pydantic v2 스키마 기반)
    - 스트리밍 응답 지원
    - 에이전트 트레이싱/관찰성

    주요 기능:
    - 5단계 하이브리드 검색: BM25 → Sentence-Transformers → RRF → MCP Registry → GPT-5.2
    - 동적 도구 로딩 및 컨텍스트 주입
    - Azure OpenAI Responses API Function Calling
    - 원격 MCP 서버 네이티브 통합
    - Reciprocal Rank Fusion (RRF) 하이브리드 검색

    Attributes:
        client: OpenAI 클라이언트 (v1 API)
        model: 사용할 모델 배포 이름 (기본: gpt-5.2)
        reasoning_model: 추론 모델 이름 (기본: gpt-5.2, 네이티브 추론 내장)
        active_tools: 현재 활성화된 도구 목록
        last_response_id: 마지막 응답 ID (대화 체이닝용)
        remote_mcp_servers: 연결된 원격 MCP 서버 목록
        enable_tracing: 에이전트 트레이싱 활성화 여부
    """

    # Python 타입 → JSON Schema 타입 매핑 (상수)
    _PYTHON_TYPE_MAP = {
        int: "integer",
        bool: "boolean",
        float: "number",
        str: "string",
        list: "array",
        dict: "object",
    }

    @classmethod
    def _resolve_json_type(cls, annotation) -> str:
        """Python 타입 어노테이션을 JSON Schema 타입으로 변환합니다.

        Optional[X], List[X], Dict[K,V] 등 제네릭 타입을 올바르게 처리합니다.
        """
        if annotation is None or annotation is inspect.Parameter.empty:
            return "string"

        # Optional[X] → Union[X, None] → X의 타입 추출
        origin = get_origin(annotation)
        if origin is Union:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if args:
                return cls._resolve_json_type(args[0])
            return "string"

        # list, List, List[str] 등
        if origin is list or annotation is list:
            return "array"

        # dict, Dict, Dict[str, Any] 등
        if origin is dict or annotation is dict:
            return "object"

        return cls._PYTHON_TYPE_MAP.get(annotation, "string")

    # 시스템 프롬프트
    SYSTEM_PROMPT = """당신은 Azure 기반의 다양한 도구에 접근할 수 있는 고급 AI 어시스턴트입니다.

당신은 수백 개의 MCP(Model Context Protocol) 도구 라이브러리에 접근할 수 있습니다.
모든 도구가 기본적으로 로드되어 있지는 않으므로, 다음 단계를 따르세요:

1. 사용자 요청에 직접적으로 대응하는 도구가 없으면, 먼저 'search_available_tools' 함수를 사용하여 관련 도구를 검색하세요.
2. 검색 결과에서 적절한 도구를 찾으면, 'load_tool' 함수를 사용하여 해당 도구를 로드하세요.
3. 도구가 로드되면, 다음 턴에서 해당 도구를 사용하여 사용자의 요청을 처리할 수 있습니다.
4. 원격 MCP 서버의 도구가 필요하면, 직접 해당 MCP 서버 도구를 활용할 수 있습니다.

도구 검색 및 로드 과정을 사용자에게 알려주세요. 예를 들어:
- "번역 도구를 검색하고 있습니다..."
- "azure_translator 도구를 로드합니다..."
- "Microsoft Learn MCP 서버에서 정보를 검색합니다..."

사용자의 요청을 정확하게 이해하고, 가장 적합한 도구를 선택하여 작업을 수행하세요.
한국어로 응답해 주세요."""

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        remote_mcp_servers: Optional[List[Dict[str, Any]]] = None,
        enable_streaming: bool = False,
        reasoning_model: Optional[str] = None,
        enable_tracing: bool = False
    ):
        """
        에이전트를 초기화합니다.

        Args:
            azure_endpoint: Azure OpenAI 엔드포인트 URL
            api_key: API 키
            api_version: v1 API 버전 ("preview" 또는 "latest")
            deployment_name: 모델 배포 이름
            remote_mcp_servers: 원격 MCP 서버 설정 목록
            enable_streaming: 스트리밍 응답 활성화 여부
            reasoning_model: 추론 모델 이름 (None이면 기본 모델 사용, GPT-5.2 네이티브 추론)
            enable_tracing: 트레이싱/관찰성 활성화 여부
        """
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "preview")
        self.model = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2")
        self.enable_streaming = enable_streaming
        self.reasoning_model = reasoning_model or os.getenv("AZURE_OPENAI_REASONING_MODEL")
        self.enable_tracing = enable_tracing

        # v1 API: OpenAI 클라이언트 + base_url 방식
        base_url = f"{self.azure_endpoint.rstrip('/')}/openai/v1/"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            default_query={"api-version": self.api_version}
        )

        # 활성 도구 초기화
        self.active_tools: Dict[str, Callable] = {}

        # Responses API: previous_response_id로 대화 자동 체이닝
        self.last_response_id: Optional[str] = None
        self._conversation_turns: int = 0

        # 원격 MCP 서버 설정
        self.remote_mcp_servers: List[Dict[str, Any]] = remote_mcp_servers or []

        # 기본 도구 함수 매핑
        self._base_tool_functions = {
            "search_available_tools": search_available_tools,
            "load_tool": load_tool
        }

        # 기본 도구 스키마 캐시 (변경되지 않으므로 한 번만 생성)
        self._base_tools_schema_cache: Optional[List[Dict[str, Any]]] = None

        logger.info(f"DynamicMCPAgent 초기화 완료 - 모델: {self.model}, API: v1/{self.api_version}")
        if self.reasoning_model:
            logger.info(f"  추론 모델: {self.reasoning_model}")
        if self.remote_mcp_servers:
            logger.info(f"  원격 MCP 서버: {len(self.remote_mcp_servers)}개 연결됨")
        if self.enable_tracing:
            logger.info("  트레이싱: 활성화됨")

    @staticmethod
    def _serialize_result(result: Any) -> str:
        """도구 실행 결과를 JSON 문자열로 직렬화합니다."""
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    def _get_base_tools_schema(self) -> List[Dict[str, Any]]:
        """기본 도구(검색, 로드)의 Responses API 함수 스키마를 반환합니다. (캐시됨)"""
        if self._base_tools_schema_cache is not None:
            # MCP 서버 수가 바뀌지 않았으면 캐시 반환
            expected_len = 2 + len(self.remote_mcp_servers)
            if len(self._base_tools_schema_cache) == expected_len:
                return self._base_tools_schema_cache

        tools = [
            {
                "type": "function",
                "name": "search_available_tools",
                "description": "사용 가능한 도구 라이브러리에서 적합한 도구를 검색합니다. 특정 작업에 필요한 도구가 없을 때 이 함수를 사용하세요.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색 키워드 (예: 'search', 'database', 'translate', 'image', '번역', '검색')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "반환할 최대 결과 수 (기본값: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "type": "function",
                "name": "load_tool",
                "description": "특정 도구를 현재 컨텍스트에 로드합니다. 'search_available_tools'로 도구를 찾은 후 이 함수를 호출하세요.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "로드할 도구의 정확한 이름"
                        }
                    },
                    "required": ["tool_name"]
                }
            }
        ]

        # 원격 MCP 서버 도구 추가 (Responses API 네이티브 지원)
        for mcp_server in self.remote_mcp_servers:
            tools.append({
                "type": "mcp",
                "server_label": mcp_server.get("server_label", "mcp_server"),
                "server_url": mcp_server["server_url"],
                "server_description": mcp_server.get("server_description", ""),
                "require_approval": mcp_server.get("require_approval", "never"),
                "allowed_tools": mcp_server.get("allowed_tools"),  # None이면 모든 도구 허용
            })

        self._base_tools_schema_cache = tools
        return tools

    def _get_active_tools_schema(self) -> List[Dict[str, Any]]:
        """현재 활성화된 도구들의 Responses API 함수 스키마를 반환합니다."""
        # 캐시된 리스트를 복사하여 원본 변이 방지
        tools_schema = list(self._get_base_tools_schema())

        for tool_name, tool_func in self.active_tools.items():
            # 도구의 docstring에서 설명 추출
            description = tool_func.__doc__ or f"{tool_name} 도구"
            description = description.strip().split('\n')[0]  # 첫 줄만 사용

            # Responses API 형식: "name"이 최상위에 위치
            tool_schema = {
                "type": "function",
                "name": tool_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }

            # 함수 시그니처에서 파라미터 추출
            sig = inspect.signature(tool_func)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                param_type = self._resolve_json_type(param.annotation)

                tool_schema["parameters"]["properties"][param_name] = {
                    "type": param_type,
                    "description": f"{param_name} 파라미터"
                }

                if param.default == inspect.Parameter.empty:
                    tool_schema["parameters"]["required"].append(param_name)

            tools_schema.append(tool_schema)

        return tools_schema

    def _dynamic_tool_injection(self, tool_name: str) -> bool:
        """
        도구를 동적으로 에이전트에 주입합니다.

        Args:
            tool_name: 주입할 도구 이름

        Returns:
            성공 여부
        """
        if tool_name in self.active_tools:
            logger.info(f"도구 '{tool_name}'가 이미 활성화되어 있습니다.")
            return True

        tool = registry.get_tool(tool_name)
        if tool:
            self.active_tools[tool_name] = tool
            logger.info(f"[동적 주입] 도구 '{tool_name}'가 에이전트에 추가되었습니다.")
            return True

        logger.warning(f"도구 '{tool_name}'를 레지스트리에서 찾을 수 없습니다.")
        return False

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        도구를 실행하고 결과를 반환합니다.

        Args:
            tool_name: 실행할 도구 이름
            arguments: 도구 인자

        Returns:
            도구 실행 결과 (JSON 문자열)
        """
        # 기본 도구 확인
        if tool_name in self._base_tool_functions:
            func = self._base_tool_functions[tool_name]
            result = func(**arguments)

            # load_tool 호출 시 동적 주입 수행
            if tool_name == "load_tool":
                requested_tool = arguments.get("tool_name")
                self._dynamic_tool_injection(requested_tool)

            return self._serialize_result(result)

        # 활성화된 도구 확인
        if tool_name in self.active_tools:
            func = self.active_tools[tool_name]
            try:
                result = func(**arguments)
                return self._serialize_result(result)
            except Exception as e:
                logger.error(f"도구 '{tool_name}' 실행 중 오류: {e}")
                return json.dumps({"error": str(e)}, ensure_ascii=False)

        return json.dumps({
            "error": f"도구 '{tool_name}'를 찾을 수 없습니다. 먼저 load_tool로 도구를 로드하세요."
        }, ensure_ascii=False)

    def add_remote_mcp_server(
        self,
        server_url: str,
        server_label: str,
        server_description: str = "",
        require_approval: str = "never",
        allowed_tools: Optional[List[str]] = None
    ) -> None:
        """원격 MCP 서버를 추가합니다."""
        self.remote_mcp_servers.append({
            "server_url": server_url,
            "server_label": server_label,
            "server_description": server_description,
            "require_approval": require_approval,
            "allowed_tools": allowed_tools,
        })
        logger.info(f"원격 MCP 서버 추가됨: {server_label} ({server_url})")

    async def chat_with_reasoning(self, user_message: str) -> str:
        """
        GPT-5.2 네이티브 추론을 사용하여 복잡한 추론 작업을 수행합니다.

        Args:
            user_message: 사용자 입력 메시지

        Returns:
            에이전트 응답
        """
        model = self.reasoning_model or self.model
        self._conversation_turns += 1

        try:
            response = self.client.responses.create(
                model=model,
                input=[{"role": "user", "content": user_message}],
                instructions=self.SYSTEM_PROMPT,
            )

            self.last_response_id = response.id
            return response.output_text or ""

        except Exception as e:
            logger.error(f"추론 모델 호출 오류: {e}")
            return f"죄송합니다. 추론 모델 호출 중 오류가 발생했습니다: {str(e)}"

    async def chat_structured(
        self,
        user_message: str,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Structured Outputs를 사용하여 스키마 기반 구조화된 응답을 생성합니다.

        Args:
            user_message: 사용자 입력 메시지
            response_schema: JSON Schema 딕셔너리 (None이면 기본 스키마)

        Returns:
            구조화된 응답 딕셔너리
        """
        self._conversation_turns += 1

        if response_schema is None:
            response_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "답변 내용"},
                            "confidence": {"type": "number", "description": "신뢰도 (0-1)"},
                            "sources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "참고 소스"
                            }
                        },
                        "required": ["answer", "confidence", "sources"],
                        "additionalProperties": False
                    }
                }
            }

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": user_message}],
                instructions=self.SYSTEM_PROMPT,
                text=response_schema,
            )

            self.last_response_id = response.id
            return json.loads(response.output_text or "{}")

        except Exception as e:
            logger.error(f"Structured Output 호출 오류: {e}")
            return {"error": str(e)}

    async def chat(self, user_message: str) -> str:
        """
        Responses API를 사용하여 사용자 메시지에 응답합니다.

        previous_response_id를 활용하여 대화 컨텍스트를 자동으로 유지합니다.
        이전 대화의 모든 컨텍스트가 서버 측에서 관리됩니다.

        Args:
            user_message: 사용자 입력 메시지

        Returns:
            에이전트 응답
        """
        self._conversation_turns += 1

        # 최대 반복 횟수 (도구 호출 루프 방지)
        max_iterations = 10
        iteration = 0
        current_response_id = self.last_response_id

        # 첫 번째 요청 입력
        current_input = [{"role": "user", "content": user_message}]

        while iteration < max_iterations:
            iteration += 1

            # 현재 활성 도구 스키마 가져오기
            tools = self._get_active_tools_schema()

            # Responses API 호출
            try:
                create_params = {
                    "model": self.model,
                    "input": current_input,
                    "tools": tools,
                    "tool_choice": "auto",
                    "instructions": self.SYSTEM_PROMPT,
                    "temperature": 0.7,
                }

                # 이전 대화가 있으면 체이닝
                if current_response_id:
                    create_params["previous_response_id"] = current_response_id

                response = self.client.responses.create(**create_params)

            except Exception as e:
                logger.error(f"Responses API 호출 오류: {e}")
                return f"죄송합니다. API 호출 중 오류가 발생했습니다: {str(e)}"

            # 응답 ID 저장 (다음 대화 체이닝용)
            self.last_response_id = response.id

            # 도구 호출 확인
            has_function_calls = False
            function_call_outputs = []

            for output_item in response.output:
                if output_item.type == "function_call":
                    has_function_calls = True
                    tool_name = output_item.name
                    call_id = output_item.call_id

                    try:
                        arguments = json.loads(output_item.arguments)
                    except (json.JSONDecodeError, AttributeError):
                        arguments = {}

                    logger.info(f"도구 호출: {tool_name}({arguments})")

                    # 도구 실행
                    tool_result = self._execute_tool(tool_name, arguments)

                    function_call_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": tool_result
                    })

            # 도구 호출이 없으면 최종 응답 반환
            if not has_function_calls:
                return response.output_text or ""

            # 도구 결과를 다음 요청 입력으로 설정
            current_input = function_call_outputs
            current_response_id = response.id

        return "죄송합니다. 요청을 처리하는 데 너무 많은 단계가 필요합니다. 다시 시도해 주세요."

    async def chat_stream(self, user_message: str):
        """
        스트리밍 방식으로 응답합니다. (Responses API 스트리밍)
        도구 호출이 필요한 경우 도구 실행 후 스트리밍을 재개합니다.

        Args:
            user_message: 사용자 입력 메시지

        Yields:
            응답 텍스트 델타
        """
        self._conversation_turns += 1

        max_iterations = 10
        iteration = 0
        current_input = [{"role": "user", "content": user_message}]
        current_response_id = self.last_response_id

        while iteration < max_iterations:
            iteration += 1
            tools = self._get_active_tools_schema()

            create_params = {
                "model": self.model,
                "input": current_input,
                "tools": tools,
                "tool_choice": "auto",
                "instructions": self.SYSTEM_PROMPT,
                "temperature": 0.7,
                "stream": True,
            }

            if current_response_id:
                create_params["previous_response_id"] = current_response_id

            try:
                stream = self.client.responses.create(**create_params)

                has_function_calls = False
                function_calls = {}  # call_id -> {name, arguments}

                for event in stream:
                    if event.type == 'response.output_text.delta':
                        yield event.delta
                    elif event.type == 'response.function_call_arguments.done':
                        has_function_calls = True
                        function_calls[event.call_id] = {
                            "name": event.name,
                            "arguments": event.arguments
                        }
                    elif event.type == 'response.completed':
                        self.last_response_id = event.response.id
                        current_response_id = event.response.id

                if not has_function_calls:
                    return  # 스트리밍 처리 완료

                # 도구 호출 처리 후 반복
                function_call_outputs = []
                for call_id, call_info in function_calls.items():
                    try:
                        arguments = json.loads(call_info["arguments"])
                    except (json.JSONDecodeError, AttributeError):
                        arguments = {}

                    tool_result = self._execute_tool(call_info["name"], arguments)
                    function_call_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": tool_result
                    })

                current_input = function_call_outputs
                yield "\n"  # 도구 호출 간 줌바꾼

            except Exception as e:
                logger.error(f"스트리밍 오류: {e}")
                yield f"\n⚠️ 스트리밍 중 오류 발생: {str(e)}"
                return

    def chat_sync(self, user_message: str) -> str:
        """동기 방식의 채팅 메서드"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 이미 이벤트 루프 실행 중이면 별도 스레드에서 실행
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.chat(user_message))
                return future.result(timeout=120)
        return asyncio.run(self.chat(user_message))

    def reset_conversation(self) -> None:
        """대화 기록을 초기화합니다. (previous_response_id 리셋)"""
        self.last_response_id = None
        self._conversation_turns = 0
        logger.info("대화 기록이 초기화되었습니다. (response_id 리셋)")

    def reset_tools(self) -> None:
        """활성화된 도구를 초기화합니다."""
        self.active_tools.clear()
        logger.info("활성 도구가 초기화되었습니다.")

    def get_active_tools_list(self) -> List[str]:
        """현재 활성화된 도구 목록을 반환합니다."""
        tool_names = list(self.active_tools.keys())
        # 원격 MCP 서버 도구도 표시
        for mcp in self.remote_mcp_servers:
            tool_names.append(f"[MCP] {mcp.get('server_label', 'unknown')}")
        return tool_names

    def get_stats(self) -> Dict[str, Any]:
        """에이전트 통계를 반환합니다."""
        return {
            "model": self.model,
            "api": f"v1/{self.api_version}",
            "reasoning_model": self.reasoning_model or "None",
            "total_tools_in_registry": registry.count(),
            "active_tools": len(self.active_tools),
            "active_tool_names": self.get_active_tools_list(),
            "remote_mcp_servers": len(self.remote_mcp_servers),
            "conversation_turns": self._conversation_turns,
            "last_response_id": self.last_response_id,
            "tracing_enabled": self.enable_tracing,
            "structured_outputs": PYDANTIC_AVAILABLE,
        }


# 기본 원격 MCP 서버 설정
DEFAULT_REMOTE_MCP_SERVERS = [
    {
        "server_label": "microsoft_learn",
        "server_url": "https://learn.microsoft.com/api/mcp",
        "server_description": "Microsoft Learn MCP 서버 - Microsoft 공식 문서 검색 및 조회",
        "require_approval": "never",
    },
    {
        "server_label": "github",
        "server_url": "https://api.githubcopilot.com/mcp/",
        "server_description": "GitHub MCP 서버 - 코드 검색, PR, 이슈 관리",
        "require_approval": "never",
    },
]


def create_agent(
    enable_remote_mcp: bool = True,
    remote_mcp_servers: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> DynamicMCPAgent:
    """에이전트 인스턴스를 생성하고 도구를 초기화합니다.

    Args:
        enable_remote_mcp: 기본 원격 MCP 서버 활성화 여부
        remote_mcp_servers: 추가 원격 MCP 서버 설정
        **kwargs: DynamicMCPAgent 생성자 인자

    Returns:
        초기화된 DynamicMCPAgent 인스턴스
    """
    # 도구 초기화 (동기 함수로 변경됨)
    initialize_mcp_tools()

    # 원격 MCP 서버 설정 병합
    mcp_servers = []
    if enable_remote_mcp:
        mcp_servers.extend(DEFAULT_REMOTE_MCP_SERVERS)
    if remote_mcp_servers:
        mcp_servers.extend(remote_mcp_servers)

    kwargs.setdefault("remote_mcp_servers", mcp_servers)

    # 에이전트 생성
    return DynamicMCPAgent(**kwargs)
