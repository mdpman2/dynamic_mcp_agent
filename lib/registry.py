# -*- coding: utf-8 -*-
"""
Advanced Hybrid Tool Registry with Multi-layer Search

이 모듈은 MCP 도구들을 인덱싱하고 다층 하이브리드 검색 방식을 사용하여
사용자 쿼리에 가장 적합한 도구를 검색하는 기능을 제공합니다.

하이브리드 검색 전략 (5단계 + RRF):
1️⃣ BM25 (빠르고 무료) - 키워드 매칭
2️⃣ Sentence-Transformers (로컬, 다국어) - 의미론적 유사성
3️⃣ 🔀 RRF (Reciprocal Rank Fusion) - BM25 + Sentence 결과 통합
4️⃣ MCP Registry API (공식) - 외부 도구 발견
5️⃣ LLM 추론 (GPT-5.2) - 복잡한 쿼리 해석 (필요시에만)

v3.0.0 업데이트 (2026-02-26):
- [NEW] Reciprocal Rank Fusion (RRF) 하이브리드 검색 알고리즘
- [NEW] httpx 기반 비동기 HTTP (aiohttp 대체, Streamable HTTP 지원)
- [CHANGED] LLM 검색 기본 모델 gpt-5 → gpt-5.2
- [CHANGED] 하이브리드 검색에 RRF 점수 기반 통합 적용

v2.0.0 업데이트 (2026-02-07):
- [BREAKING] AzureOpenAI 클라이언트 → OpenAI + base_url 방식으로 전환
- [CHANGED] LLM 검색 기본 모델 gpt-4.1 → gpt-5
- [CHANGED] 기본 API 버전 2024-08-01-preview → preview

참고: https://github.com/modelcontextprotocol/registry
"""

import os
import re
import time
import inspect
import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as SentenceTransformerType

# 선택적 임포트: Sentence-Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# 선택적 임포트: httpx (비동기 HTTP - aiohttp 대체)
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

# 하위 호환성: aiohttp 폴백
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

# 현재 폴더의 .env만 로드 (상위 폴더 무시)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
logger = logging.getLogger(__name__)

# 사전 컴파일된 정규식 패턴 (토큰화 성능 최적화)
_RE_KOREAN = re.compile(r'[\uac00-\ud7a3]+')
_RE_ENGLISH = re.compile(r'[a-z0-9]+')


class MCPRegistryClient:
    """
    공식 MCP Registry API 클라이언트

    MCP Registry는 커뮤니티 기반의 MCP 서버 레지스트리입니다.
    https://registry.modelcontextprotocol.io

    API 문서: https://registry.modelcontextprotocol.io/docs
    """

    BASE_URL = "https://registry.modelcontextprotocol.io"
    API_VERSION = "v0.1"
    DEFAULT_TIMEOUT = 10

    def __init__(self, cache_ttl: int = 3600):
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = cache_ttl
        self._enabled = HTTPX_AVAILABLE or AIOHTTP_AVAILABLE

        if not self._enabled:
            logger.warning("httpx/aiohttp가 설치되지 않아 MCP Registry API가 비활성화됩니다.")

    def _build_url(self, path: str) -> str:
        """API URL을 구성합니다."""
        return f"{self.BASE_URL}/{self.API_VERSION}/{path}"

    def _get_cached(self, key: str) -> Optional[Any]:
        """캐시된 데이터를 반환합니다."""
        if key in self._cache and time.time() - self._cache_time.get(key, 0) < self._cache_ttl:
            return self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """캐시에 데이터를 저장합니다."""
        self._cache[key] = value
        self._cache_time[key] = time.time()

    async def search_servers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """MCP Registry에서 서버를 검색합니다. (httpx 우선, aiohttp 폴백)"""
        if not self._enabled:
            return []

        cache_key = f"search:{query}:{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            if HTTPX_AVAILABLE:
                async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                    response = await client.get(
                        self._build_url("servers"),
                        params={"search": query, "limit": limit}
                    )
                    if response.status_code == 200:
                        servers = response.json().get("servers", [])
                        self._set_cache(cache_key, servers)
                        logger.info(f"MCP Registry 검색 성공: '{query}' → {len(servers)}개")
                        return servers
                    logger.warning(f"MCP Registry 검색 실패: HTTP {response.status_code}")
                    return []
            elif AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self._build_url("servers"),
                        params={"search": query, "limit": limit},
                        timeout=aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT)
                    ) as response:
                        if response.status == 200:
                            servers = (await response.json()).get("servers", [])
                            self._set_cache(cache_key, servers)
                            logger.info(f"MCP Registry 검색 성공: '{query}' → {len(servers)}개")
                            return servers
                        logger.warning(f"MCP Registry 검색 실패: HTTP {response.status}")
                        return []
        except asyncio.TimeoutError:
            logger.warning("MCP Registry 검색 타임아웃")
        except Exception as e:
            logger.error(f"MCP Registry 검색 오류: {e}")
        return []

    def search_servers_sync(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """동기 버전의 서버 검색"""
        try:
            return asyncio.run(self.search_servers(query, limit))
        except RuntimeError:
            # 이미 이벤트 루프가 실행 중인 경우 새 스레드에서 실행
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.search_servers(query, limit)).result(timeout=15)
        except Exception as e:
            logger.error(f"MCP Registry 동기 검색 오류: {e}")
            return []

    async def get_server_details(self, server_name: str) -> Optional[Dict[str, Any]]:
        """서버 상세 정보를 가져옵니다. (httpx 우선, aiohttp 폴백)"""
        if not self._enabled:
            return None

        try:
            from urllib.parse import quote
            encoded_name = quote(server_name, safe='')
            url = self._build_url(f"servers/{encoded_name}/versions/latest")

            if HTTPX_AVAILABLE:
                async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                    response = await client.get(url)
                    return response.json() if response.status_code == 200 else None
            elif AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT)
                    ) as response:
                        return await response.json() if response.status == 200 else None
        except Exception as e:
            logger.error(f"MCP Registry 상세 조회 오류: {e}")
            return None

    async def list_all_servers(self, limit: int = 30) -> List[Dict[str, Any]]:
        """서버 목록을 가져옵니다 (검색 없이). (httpx 우선, aiohttp 폴백)"""
        if not self._enabled:
            return []

        try:
            params = {"limit": limit, "version": "latest"}
            if HTTPX_AVAILABLE:
                async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                    response = await client.get(
                        self._build_url("servers"), params=params
                    )
                    return response.json().get("servers", []) if response.status_code == 200 else []
            elif AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self._build_url("servers"),
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT)
                    ) as response:
                        return (await response.json()).get("servers", []) if response.status == 200 else []
        except Exception as e:
            logger.error(f"MCP Registry 서버 목록 조회 오류: {e}")
            return []


class HybridToolRegistry:
    """
    다층 하이브리드 검색 + RRF 기반 도구 레지스트리

    검색 계층:
    1. BM25 (키워드 매칭) - 빠르고 무료
    2. Sentence-Transformers (로컬 임베딩) - 다국어 지원, 의미론적
    3. MCP Registry API (외부) - 새로운 도구 발견
    4. GPT-5.2 LLM (추론) - 복잡한 쿼리
    5. RRF (Reciprocal Rank Fusion) - 다층 결과 통합

    Attributes:
        _tools: 도구 이름을 키로 하는 도구 딕셔너리
        _descriptions: 각 도구의 설명 목록
        _tool_names: 등록된 도구 이름 목록
        _bm25: BM25 검색 인덱스
        _sentence_model: Sentence-Transformers 모델
        _mcp_registry: MCP Registry API 클라이언트
    """

    # 검색 전략 상수
    BM25_CONFIDENCE_THRESHOLD = 5.0
    EMBEDDING_SIMILARITY_THRESHOLD = 0.65  # Sentence-Transformers용 임계값 (낮춤)
    RRF_K = 60  # Reciprocal Rank Fusion 파라미터 (k)

    # 추천 다국어 모델 (한국어 포함)
    DEFAULT_SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        sentence_model: Optional[str] = None,
        enable_mcp_registry: bool = True,
        mcp_registry_cache_ttl: int = 3600
    ):
        """
        하이브리드 레지스트리를 초기화합니다.

        Args:
            azure_endpoint: Azure OpenAI 엔드포인트
            api_key: API 키
            api_version: API 버전
            sentence_model: Sentence-Transformers 모델명 (None이면 기본 다국어 모델)
            enable_mcp_registry: MCP Registry API 사용 여부
            mcp_registry_cache_ttl: MCP Registry 캐시 TTL (초)
        """
        # 도구 저장소
        self._tools: Dict[str, Any] = {}
        self._descriptions: List[str] = []
        self._tool_names: List[str] = []
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}

        # BM25 인덱스
        self._bm25: Optional[BM25Okapi] = None

        # Sentence-Transformers (로컬 다국어 임베딩)
        self._sentence_model: Optional["SentenceTransformerType"] = None
        self._sentence_model_name = sentence_model or self.DEFAULT_SENTENCE_MODEL
        self._sentence_embeddings: Optional[np.ndarray] = None
        self._init_sentence_transformers()

        # MCP Registry API
        self._mcp_registry: Optional[MCPRegistryClient] = None
        self._mcp_registry_enabled = enable_mcp_registry
        if enable_mcp_registry:
            self._mcp_registry = MCPRegistryClient(cache_ttl=mcp_registry_cache_ttl)

        # Azure OpenAI v1 API 클라이언트 (LLM용)
        self._azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self._api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "preview")
        self._llm_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2")
        self._openai_client: Optional[OpenAI] = None
        self._init_openai_client()

        # 검색 통계
        self._search_stats = {
            "bm25_hits": 0,
            "sentence_hits": 0,
            "mcp_registry_hits": 0,
            "llm_hits": 0,
            "total_searches": 0
        }

        logger.info("HybridToolRegistry 초기화 완료")
        logger.info(f"  - Sentence-Transformers: {'✅ ' + self._sentence_model_name if self._sentence_model else '❌ 비활성화'}")
        logger.info(f"  - MCP Registry API: {'✅ 활성화' if self._mcp_registry_enabled else '❌ 비활성화'}")
        logger.info(f"  - LLM (GPT-5.2): {'✅ 활성화' if self._openai_client else '❌ 비활성화'}")

    def _init_sentence_transformers(self) -> None:
        """Sentence-Transformers 모델을 초기화합니다."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers가 설치되지 않았습니다. pip install sentence-transformers")
            return

        try:
            logger.info(f"Sentence-Transformers 모델 로딩 중: {self._sentence_model_name}")
            self._sentence_model = SentenceTransformer(self._sentence_model_name)
            logger.info("Sentence-Transformers 모델 로딩 완료")
        except Exception as e:
            logger.error(f"Sentence-Transformers 초기화 실패: {e}")
            self._sentence_model = None

    def _init_openai_client(self) -> None:
        """Azure OpenAI v1 API 클라이언트를 초기화합니다."""
        if self._azure_endpoint and self._api_key:
            try:
                base_url = f"{self._azure_endpoint.rstrip('/')}/openai/v1/"
                self._openai_client = OpenAI(
                    api_key=self._api_key,
                    base_url=base_url,
                    default_query={"api-version": self._api_version}
                )
                logger.info(f"Azure OpenAI v1 API 클라이언트 초기화 성공 (LLM용, api-version={self._api_version})")
            except Exception as e:
                logger.warning(f"Azure OpenAI 클라이언트 초기화 실패: {e}")
                self._openai_client = None
        else:
            logger.warning("Azure OpenAI 자격 증명이 없어 LLM 검색이 비활성화됩니다.")

    def register(
        self,
        tool: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        도구를 레지스트리에 등록합니다.

        Args:
            tool: 등록할 도구 (함수, 클래스 또는 도구 객체)
            name: 도구 이름 (None이면 자동 추출)
            description: 도구 설명 (None이면 docstring에서 추출)
            category: 도구 카테고리 (예: "search", "data", "ai")
            tags: 검색을 위한 추가 태그 목록
        """
        # 도구 이름 추출
        if name is None:
            if hasattr(tool, 'name'):
                name = tool.name
            elif hasattr(tool, '__name__'):
                name = tool.__name__
            else:
                name = str(tool)

        # 설명 추출
        if description is None:
            if hasattr(tool, 'description'):
                description = tool.description or ""
            else:
                description = inspect.getdoc(tool) or ""

        # 태그 결합하여 검색 가능한 설명 생성
        tag_string = " ".join(tags) if tags else ""
        category_string = category if category else ""

        # 검색용 인덱스 문자열 생성
        searchable_text = f"{name} {description} {category_string} {tag_string}"

        # 중복 등록 시 기존 항목 업데이트
        if name in self._tools:
            idx = self._tool_names.index(name)
            self._descriptions[idx] = searchable_text
        else:
            self._tool_names.append(name)
            self._descriptions.append(searchable_text)

        # 레지스트리에 등록
        self._tools[name] = tool

        # 메타데이터 저장
        self._tool_metadata[name] = {
            "description": description,
            "category": category,
            "tags": tags or [],
            "searchable_text": searchable_text
        }

        # 인덱스 재구축
        self._rebuild_bm25_index()
        self._sentence_embeddings = None  # lazy rebuild

        logger.debug(f"도구 등록됨: {name}")

    def register_batch(
        self,
        tools: List[tuple],
    ) -> None:
        """
        여러 도구를 일괄 등록합니다. (인덱스 재구축은 마지막에 1회만 수행)

        Args:
            tools: (tool, name, description, category, tags) 튜플 리스트
        """
        for entry in tools:
            tool = entry[0]
            name = entry[1] if len(entry) > 1 else None
            description = entry[2] if len(entry) > 2 else None
            category = entry[3] if len(entry) > 3 else None
            tags = entry[4] if len(entry) > 4 else None

            # 도구 이름 추출
            if name is None:
                if hasattr(tool, 'name'):
                    name = tool.name
                elif hasattr(tool, '__name__'):
                    name = tool.__name__
                else:
                    name = str(tool)

            # 설명 추출
            if description is None:
                if hasattr(tool, 'description'):
                    description = tool.description or ""
                else:
                    description = inspect.getdoc(tool) or ""

            tag_string = " ".join(tags) if tags else ""
            category_string = category if category else ""
            searchable_text = f"{name} {description} {category_string} {tag_string}"

            # 중복 등록 시 기존 항목 업데이트
            if name in self._tools:
                idx = self._tool_names.index(name)
                self._descriptions[idx] = searchable_text
            else:
                self._tool_names.append(name)
                self._descriptions.append(searchable_text)

            self._tools[name] = tool
            self._tool_metadata[name] = {
                "description": description,
                "category": category,
                "tags": tags or [],
                "searchable_text": searchable_text
            }

        # 인덱스 재구축은 마지막에 1회만
        self._rebuild_bm25_index()
        self._sentence_embeddings = None
        logger.info(f"도구 일괄 등록 완료: {len(tools)}개")

    def _rebuild_bm25_index(self) -> None:
        """BM25 인덱스를 재구축합니다."""
        if not self._descriptions:
            self._bm25 = None
            return

        tokenized_corpus = [self._tokenize(desc) for desc in self._descriptions]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """
        텍스트를 토큰으로 분리합니다. (한국어/영어 지원)
        사전 컴파일된 정규식 패턴을 사용하여 성능을 최적화합니다.
        """
        text = text.lower().replace("_", " ").replace("-", " ")
        tokens = []

        # 한글 토큰 추출 (음절 + 바이그램)
        for match in _RE_KOREAN.finditer(text):
            word = match.group()
            tokens.append(word)
            if len(word) >= 2:
                tokens.extend(word[i:i+2] for i in range(len(word) - 1))

        # 영어 토큰 추출
        tokens.extend(match.group() for match in _RE_ENGLISH.finditer(text))

        return tokens

    # =========================================================================
    # Sentence-Transformers 검색
    # =========================================================================

    def _build_sentence_embeddings(self) -> None:
        """Sentence-Transformers 임베딩을 구축합니다."""
        if self._sentence_embeddings is not None:
            return

        if not self._sentence_model or not self._descriptions:
            return

        logger.info("Sentence-Transformers 임베딩 구축 중...")
        self._sentence_embeddings = self._sentence_model.encode(
            self._descriptions,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        logger.info(f"Sentence-Transformers 임베딩 구축 완료: {self._sentence_embeddings.shape}")

    def _sentence_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Sentence-Transformers 기반 시맨틱 검색"""
        if not self._sentence_model:
            return []

        self._build_sentence_embeddings()

        if self._sentence_embeddings is None:
            return []

        # 쿼리 임베딩
        query_embedding = self._sentence_model.encode(
            query,
            convert_to_numpy=True
        )

        # 코사인 유사도 계산
        similarities = np.dot(self._sentence_embeddings, query_embedding)
        similarities = similarities / (
            np.linalg.norm(self._sentence_embeddings, axis=1) *
            np.linalg.norm(query_embedding) + 1e-8
        )

        # 상위 k개 결과
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self._tool_names[i], float(similarities[i]))
            for i in top_indices
        ]

        return results

    # =========================================================================
    # MCP Registry API 검색
    # =========================================================================

    def _mcp_registry_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """MCP Registry API에서 도구를 검색합니다."""
        if not self._mcp_registry or not self._mcp_registry_enabled:
            return []

        try:
            servers = self._mcp_registry.search_servers_sync(query, limit)
            return servers
        except Exception as e:
            logger.error(f"MCP Registry 검색 오류: {e}")
            return []

    def discover_external_tools(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        MCP Registry에서 외부 도구를 발견합니다.

        Args:
            query: 검색 쿼리
            limit: 최대 결과 수

        Returns:
            발견된 MCP 서버 정보 목록
        """
        return self._mcp_registry_search(query, limit)

    # =========================================================================
    # LLM 기반 검색
    # =========================================================================

    def _llm_search(self, query: str, candidates: List[str], top_k: int = 5) -> List[str]:
        """
        LLM을 사용하여 가장 적합한 도구를 선택합니다.
        Responses API를 사용하여 도구 선택을 수행합니다.
        """
        if not self._openai_client:
            return candidates[:top_k]

        # 도구 목록 생성
        tool_descriptions = []
        for name in candidates:
            metadata = self._tool_metadata.get(name, {})
            desc = metadata.get("description", "")[:200]
            tool_descriptions.append(f"- {name}: {desc}")

        tool_list = "\n".join(tool_descriptions)

        prompt = f"""사용자 쿼리에 가장 적합한 도구를 선택하세요.

사용자 쿼리: "{query}"

사용 가능한 도구:
{tool_list}

위 도구 중에서 사용자의 요청에 가장 적합한 도구를 최대 {top_k}개 선택하여,
도구 이름만 쉼표로 구분하여 반환하세요. 설명은 포함하지 마세요.

예시 응답: azure_translator_tool, azure_text_analytics_tool"""

        try:
            # Responses API 사용 (v3.0 통일)
            response = self._openai_client.responses.create(
                model=self._llm_model,
                instructions="당신은 사용자 요청에 가장 적합한 도구를 선택하는 전문가입니다. 한국어와 영어 쿼리를 모두 이해하고, 정확한 도구를 추천합니다.",
                input=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            result = response.output_text or ""
            selected_tools = [t.strip() for t in result.split(",")]
            valid_tools = [t for t in selected_tools if t in self._tools]

            logger.info(f"LLM 검색 결과: {valid_tools}")
            return valid_tools if valid_tools else candidates[:top_k]

        except Exception as e:
            logger.error(f"LLM 검색 실패: {e}")
            return candidates[:top_k]

    # =========================================================================
    # 하이브리드 검색
    # =========================================================================

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """BM25 검색을 수행합니다."""
        if not self._bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        scored_results = list(zip(self._tool_names, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results[:top_k]

    def _hybrid_search(self, query: str, top_k: int) -> List[str]:
        """
        다층 하이브리드 검색을 수행합니다.
        Reciprocal Rank Fusion (RRF)을 사용하여 BM25와 Sentence-Transformers 결과를 통합합니다.
        """

        # 1️⃣ 1차: BM25 검색 (빠르고 무료)
        bm25_results = self._bm25_search(query, top_k * 2)

        if bm25_results:
            top_score = bm25_results[0][1]

            # BM25 점수가 충분히 높으면 BM25 결과만 사용
            if top_score >= self.BM25_CONFIDENCE_THRESHOLD:
                logger.info(f"[BM25 HIT] 쿼리: '{query}', 최고점수: {top_score:.2f}")
                self._search_stats["bm25_hits"] += 1
                return self._format_results(bm25_results[:top_k])

        # 2️⃣ 2차: Sentence-Transformers 검색 (로컬, 다국어)
        if self._sentence_model:
            sentence_results = self._sentence_search(query, top_k * 2)

            if sentence_results:
                top_similarity = sentence_results[0][1]

                if top_similarity >= self.EMBEDDING_SIMILARITY_THRESHOLD:
                    logger.info(f"[SENTENCE HIT] 쿼리: '{query}', 최고유사도: {top_similarity:.3f}")
                    self._search_stats["sentence_hits"] += 1
                    return self._format_results(sentence_results[:top_k])

                # 3️⃣ RRF: BM25 + Sentence 결과를 Reciprocal Rank Fusion으로 통합
                rrf_scores: Dict[str, float] = {}

                for rank, (name, _score) in enumerate(bm25_results):
                    rrf_scores[name] = rrf_scores.get(name, 0.0) + 1.0 / (self.RRF_K + rank + 1)

                for rank, (name, _score) in enumerate(sentence_results):
                    rrf_scores[name] = rrf_scores.get(name, 0.0) + 1.0 / (self.RRF_K + rank + 1)

                rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

                if rrf_ranked:
                    logger.info(f"[RRF FUSION] 쿼리: '{query}', 통합 후보: {len(rrf_ranked)}개")
                    self._search_stats["bm25_hits"] += 1
                    self._search_stats["sentence_hits"] += 1
                    return self._format_results([(name, score) for name, score in rrf_ranked])

        # 4️⃣ BM25만 있는 경우 LLM 폴백
        if self._openai_client and bm25_results:
            candidates = [name for name, _ in bm25_results[:top_k * 2]]
            logger.info(f"[LLM FALLBACK] 쿼리: '{query}', 후보: {len(candidates)}개")
            self._search_stats["llm_hits"] += 1
            selected = self._llm_search(query, candidates, top_k)
            return self._format_results([(name, 1.0) for name in selected])

        # Fallback: BM25 결과 반환
        logger.info(f"[BM25 FALLBACK] 쿼리: '{query}'")
        self._search_stats["bm25_hits"] += 1
        return self._format_results(bm25_results[:top_k])

    def _format_results(self, results: List[Tuple[str, float]]) -> List[str]:
        """검색 결과를 포맷팅합니다."""
        return [
            f"{name}: {self._tool_metadata.get(name, {}).get('description', '').split(chr(10))[0][:150]}"
            for name, _score in results
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        include_external: bool = False
    ) -> List[str]:
        """
        다층 하이브리드 방식으로 도구를 검색합니다.

        검색 전략:
        1. BM25로 1차 검색 (빠르고 무료)
        2. Sentence-Transformers로 2차 검색 (로컬 다국어 임베딩)
        3. 필요시 LLM 재순위화 (고비용, 최고 정확도)
        4. include_external=True면 MCP Registry에서 외부 도구 발견

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            strategy: 검색 전략 ("bm25", "sentence", "llm", "hybrid")
            include_external: MCP Registry에서 외부 도구 검색 여부

        Returns:
            "도구이름: 설명 요약" 형식의 문자열 목록
        """
        self._search_stats["total_searches"] += 1

        results = []

        # 로컬 도구 검색
        if self._descriptions:
            if strategy == "bm25":
                results = self._format_results(self._bm25_search(query, top_k))
            elif strategy == "sentence":
                sentence_results = self._sentence_search(query, top_k)
                results = self._format_results(sentence_results)
            elif strategy == "llm":
                all_tools = list(self._tools.keys())
                selected = self._llm_search(query, all_tools, top_k)
                results = self._format_results([(name, 1.0) for name in selected])
            else:
                results = self._hybrid_search(query, top_k)

        # 외부 도구 검색 (MCP Registry)
        if include_external and self._mcp_registry_enabled:
            external = self._mcp_registry_search(query, top_k)
            if external:
                self._search_stats["mcp_registry_hits"] += 1
                for server in external:
                    name = server.get("name", "unknown")
                    desc = server.get("description", "")[:100]
                    results.append(f"[MCP Registry] {name}: {desc}")

        return results

    def get_tool(self, name: str) -> Optional[Any]:
        """이름으로 도구를 가져옵니다."""
        return self._tools.get(name)

    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """도구의 메타데이터를 가져옵니다."""
        return self._tool_metadata.get(name)

    def list_all_tools(self) -> List[str]:
        """등록된 모든 도구 이름을 반환합니다."""
        return list(self._tools.keys())

    def get_tools_by_category(self, category: str) -> List[str]:
        """특정 카테고리의 도구 이름을 반환합니다."""
        return [
            name for name, metadata in self._tool_metadata.items()
            if metadata.get("category") == category
        ]

    def count(self) -> int:
        """등록된 도구 수를 반환합니다."""
        return len(self._tools)

    def clear(self) -> None:
        """모든 도구를 레지스트리에서 제거합니다."""
        self._tools.clear()
        self._descriptions.clear()
        self._tool_names.clear()
        self._tool_metadata.clear()
        self._bm25 = None
        self._sentence_embeddings = None
        logger.info("레지스트리가 초기화되었습니다.")

    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계를 반환합니다."""
        stats = self._search_stats.copy()
        total = stats["total_searches"]
        if total > 0:
            stats["bm25_ratio"] = f"{stats['bm25_hits'] / total * 100:.1f}%"
            stats["embedding_ratio"] = f"{stats['sentence_hits'] / total * 100:.1f}%"
            stats["embedding_hits"] = stats["sentence_hits"]  # 호환성 별칭
            stats["mcp_registry_ratio"] = f"{stats['mcp_registry_hits'] / total * 100:.1f}%"
            stats["llm_ratio"] = f"{stats['llm_hits'] / total * 100:.1f}%"
        else:
            stats["embedding_hits"] = 0
        return stats

    def set_thresholds(
        self,
        bm25_threshold: Optional[float] = None,
        embedding_threshold: Optional[float] = None
    ) -> None:
        """검색 임계값을 설정합니다."""
        if bm25_threshold is not None:
            self.BM25_CONFIDENCE_THRESHOLD = bm25_threshold
        if embedding_threshold is not None:
            self.EMBEDDING_SIMILARITY_THRESHOLD = embedding_threshold
        logger.info(f"임계값 설정됨 - BM25: {self.BM25_CONFIDENCE_THRESHOLD}, Embedding: {self.EMBEDDING_SIMILARITY_THRESHOLD}")

    def get_model_info(self) -> Dict[str, Any]:
        """현재 사용 중인 모델 정보를 반환합니다."""
        return {
            "sentence_model": self._sentence_model_name if self._sentence_model else None,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "mcp_registry_enabled": self._mcp_registry_enabled,
            "llm_model": self._llm_model if self._openai_client else None,
            "tool_count": len(self._tools)
        }


# 하위 호환성을 위한 별칭
ToolRegistry = HybridToolRegistry

# 전역 레지스트리 인스턴스
registry = HybridToolRegistry()
