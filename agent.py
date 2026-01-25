# -*- coding: utf-8 -*-
"""
Dynamic MCP Agent with Azure OpenAI

이 모듈은 Azure OpenAI를 사용하여 동적 도구 검색 및 로딩 기능을 갖춘
에이전트를 구현합니다. LLM에게 "검색 능력"을 부여하여 수많은 MCP 서버 중
현재 태스크에 적합한 도구만 동적으로 로딩합니다.

핵심 아키텍처:
1. Tool Search Tool: 사용 가능한 도구를 BM25로 검색
2. Tool Load Tool: 선택된 도구를 동적으로 컨텍스트에 주입
3. Dynamic Callback: 도구 로드 후 자동으로 에이전트에 도구 추가

참고: https://medium.com/google-cloud/implementing-anthropic-style-dynamic-tool-search-tool-f39d02a35139
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dotenv import load_dotenv

from openai import AzureOpenAI

from .lib.registry import registry
from .lib.tools import (
    search_available_tools,
    load_tool,
    initialize_mcp_tools,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class DynamicMCPAgent:
    """
    동적 도구 검색 및 로딩 기능을 갖춘 Azure OpenAI 에이전트
    
    이 에이전트는 수백 개의 MCP 도구 중에서 필요한 도구만 동적으로 
    로딩하여 토큰 비용을 절감하고 추론 정확도를 향상시킵니다.
    
    주요 기능:
    - BM25 기반 도구 검색
    - 동적 도구 로딩 및 컨텍스트 주입
    - Azure OpenAI Function Calling 통합
    
    Attributes:
        client: Azure OpenAI 클라이언트
        model: 사용할 모델 배포 이름
        active_tools: 현재 활성화된 도구 목록
        conversation_history: 대화 기록
    """
    
    # 시스템 프롬프트
    SYSTEM_PROMPT = """당신은 Azure 기반의 다양한 도구에 접근할 수 있는 고급 AI 어시스턴트입니다.

당신은 수백 개의 MCP(Model Context Protocol) 도구 라이브러리에 접근할 수 있습니다.
모든 도구가 기본적으로 로드되어 있지는 않으므로, 다음 단계를 따르세요:

1. 사용자 요청에 직접적으로 대응하는 도구가 없으면, 먼저 'search_available_tools' 함수를 사용하여 관련 도구를 검색하세요.
2. 검색 결과에서 적절한 도구를 찾으면, 'load_tool' 함수를 사용하여 해당 도구를 로드하세요.
3. 도구가 로드되면, 다음 턴에서 해당 도구를 사용하여 사용자의 요청을 처리할 수 있습니다.

도구 검색 및 로드 과정을 사용자에게 알려주세요. 예를 들어:
- "번역 도구를 검색하고 있습니다..."
- "azure_translator 도구를 로드합니다..."

사용자의 요청을 정확하게 이해하고, 가장 적합한 도구를 선택하여 작업을 수행하세요.
한국어로 응답해 주세요."""

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None
    ):
        """
        에이전트를 초기화합니다.
        
        Args:
            azure_endpoint: Azure OpenAI 엔드포인트 URL
            api_key: API 키
            api_version: API 버전
            deployment_name: 모델 배포 이름
        """
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        self.model = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        
        # Azure OpenAI 클라이언트 초기화
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # 활성 도구 및 대화 기록 초기화
        self.active_tools: Dict[str, Callable] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        
        # 기본 도구 함수 매핑
        self._base_tool_functions = {
            "search_available_tools": search_available_tools,
            "load_tool": load_tool
        }
        
        logger.info(f"DynamicMCPAgent 초기화 완료 - 모델: {self.model}")
    
    def _get_base_tools_schema(self) -> List[Dict[str, Any]]:
        """기본 도구(검색, 로드)의 OpenAI 함수 스키마를 반환합니다."""
        return [
            {
                "type": "function",
                "function": {
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
                }
            },
            {
                "type": "function",
                "function": {
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
            }
        ]
    
    def _get_active_tools_schema(self) -> List[Dict[str, Any]]:
        """현재 활성화된 도구들의 OpenAI 함수 스키마를 반환합니다."""
        tools_schema = self._get_base_tools_schema()
        
        for tool_name, tool_func in self.active_tools.items():
            # 도구의 docstring에서 설명 추출
            description = tool_func.__doc__ or f"{tool_name} 도구"
            description = description.strip().split('\n')[0]  # 첫 줄만 사용
            
            # 간단한 스키마 생성 (실제 구현에서는 더 정교한 스키마 추출 필요)
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # 함수 시그니처에서 파라미터 추출 시도
            import inspect
            sig = inspect.signature(tool_func)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                param_type = "string"  # 기본 타입
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == float:
                        param_type = "number"
                
                tool_schema["function"]["parameters"]["properties"][param_name] = {
                    "type": param_type,
                    "description": f"{param_name} 파라미터"
                }
                
                if param.default == inspect.Parameter.empty:
                    tool_schema["function"]["parameters"]["required"].append(param_name)
            
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
            
            return json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
        
        # 활성화된 도구 확인
        if tool_name in self.active_tools:
            func = self.active_tools[tool_name]
            try:
                result = func(**arguments)
                return json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            except Exception as e:
                logger.error(f"도구 '{tool_name}' 실행 중 오류: {e}")
                return json.dumps({"error": str(e)}, ensure_ascii=False)
        
        return json.dumps({
            "error": f"도구 '{tool_name}'를 찾을 수 없습니다. 먼저 load_tool로 도구를 로드하세요."
        }, ensure_ascii=False)
    
    async def chat(self, user_message: str) -> str:
        """
        사용자 메시지에 대해 응답합니다.
        
        Args:
            user_message: 사용자 입력 메시지
            
        Returns:
            에이전트 응답
        """
        # 대화 기록에 사용자 메시지 추가
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *self.conversation_history
        ]
        
        # 최대 반복 횟수 (도구 호출 루프 방지)
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 현재 활성 도구 스키마 가져오기
            tools = self._get_active_tools_schema()
            
            # Azure OpenAI API 호출
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7
                )
            except Exception as e:
                logger.error(f"API 호출 오류: {e}")
                return f"죄송합니다. API 호출 중 오류가 발생했습니다: {str(e)}"
            
            response_message = response.choices[0].message
            
            # 도구 호출이 없으면 최종 응답 반환
            if not response_message.tool_calls:
                assistant_message = response_message.content or ""
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                return assistant_message
            
            # 도구 호출 처리
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            })
            
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                logger.info(f"도구 호출: {tool_name}({arguments})")
                
                # 도구 실행
                tool_result = self._execute_tool(tool_name, arguments)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        
        return "죄송합니다. 요청을 처리하는 데 너무 많은 단계가 필요합니다. 다시 시도해 주세요."
    
    def chat_sync(self, user_message: str) -> str:
        """동기 방식의 채팅 메서드"""
        return asyncio.run(self.chat(user_message))
    
    def reset_conversation(self) -> None:
        """대화 기록을 초기화합니다."""
        self.conversation_history.clear()
        logger.info("대화 기록이 초기화되었습니다.")
    
    def reset_tools(self) -> None:
        """활성화된 도구를 초기화합니다."""
        self.active_tools.clear()
        logger.info("활성 도구가 초기화되었습니다.")
    
    def get_active_tools_list(self) -> List[str]:
        """현재 활성화된 도구 목록을 반환합니다."""
        return list(self.active_tools.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """에이전트 통계를 반환합니다."""
        return {
            "model": self.model,
            "total_tools_in_registry": registry.count(),
            "active_tools": len(self.active_tools),
            "active_tool_names": self.get_active_tools_list(),
            "conversation_turns": len(self.conversation_history)
        }


# 모듈 레벨 에이전트 인스턴스 (선택적 사용)
def create_agent(**kwargs) -> DynamicMCPAgent:
    """에이전트 인스턴스를 생성하고 도구를 초기화합니다."""
    # 도구 초기화
    asyncio.run(initialize_mcp_tools())
    
    # 에이전트 생성
    return DynamicMCPAgent(**kwargs)
