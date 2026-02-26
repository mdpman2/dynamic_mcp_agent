# 🔍 Dynamic MCP Agent v3.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI%20v1-0078D4.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![OpenAI Agents SDK](https://img.shields.io/badge/OpenAI-Agents%20SDK-412991.svg)](https://openai.github.io/openai-agents-python/)
[![MCP Registry](https://img.shields.io/badge/MCP-Registry-green.svg)](https://registry.modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Azure OpenAI v1 Responses API + Agents SDK 기반 다층 하이브리드 도구 검색 에이전트**
> BM25 → Sentence-Transformers → RRF Fusion → MCP Registry → GPT-5.2 LLM 5단계 검색
> 네이티브 원격 MCP 서버 · Structured Outputs · o4-mini 추론 · 멀티 에이전트 오케스트레이션

[English](#english) | [한국어](#한국어)

---

## ⚡ v3.0.0 업데이트 (2026-02-26)

### v2.0 → v3.0 주요 변경

| 항목 | v2.0.0 | v3.0.0 |
|------|--------|--------|
| 기본 모델 | gpt-5 | **gpt-5.2** |
| 추론 모델 | _(없음)_ | **o4-mini** (`--reasoning`) |
| 멀티 에이전트 | _(없음)_ | **OpenAI Agents SDK** (`--agents`) |
| 구조화 출력 | _(없음)_ | **Structured Outputs** (Pydantic v2) |
| 검색 알고리즘 | 4단계 순차 | **5단계 + RRF Fusion** |
| HTTP 클라이언트 | aiohttp | **httpx** (Streamable HTTP) |
| 등록 도구 수 | 20개 | **25개** |
| 트레이싱 | _(없음)_ | **에이전트 트레이싱/관찰성** |
| openai SDK | ≥ 1.86.0 | **≥ 1.93.0** |

### v3.0 신규 기능
- 🤝 **멀티 에이전트 오케스트레이션** — OpenAI Agents SDK로 전문 에이전트 핸드오프
- 🧠 **o4-mini 추론 모드** — 복잡한 수학·논리·코드 분석에 특화
- 📊 **Structured Outputs** — Pydantic v2 스키마 기반 JSON 응답 보장
- 🔀 **RRF (Reciprocal Rank Fusion)** — BM25 + Sentence-Transformers 결과 통합
- 🔭 **에이전트 트레이싱** — 멀티 에이전트 실행 추적 및 관찰성
- 🤖 **5개 신규 도구** — AI Agent Service, Computer Use, MCP Discovery, Structured Output, Realtime Audio
- ⚡ **httpx 비동기 HTTP** — aiohttp 대체, Streamable HTTP 지원

### v3.0 코드 최적화 (v3.0.1 패치)
- 🔧 `_get_active_tools_schema()` 캐시 뮤테이션 버그 수정 (`list()` 복사)
- 🔧 `register()` / `register_batch()` 중복 등록 시 `_tool_names`/`_descriptions` 정합성 수정
- 🔧 `_resolve_json_type()` 신규 — `Optional[X]`, `List[str]`, `Dict[K,V]` 제네릭 타입 정확 변환
- 🔧 MCPRegistryClient `get_server_details()` / `list_all_servers()` httpx 마이그레이션
- ⚡ `calculator_tool` 상수 모듈 레벨 이동 (호출당 재생성 방지)
- ⚡ 스트리밍 CLI 단일 이벤트 루프 최적화

---

## 한국어

### 📌 개요

MCP(Model Context Protocol) 생태계가 확장되면서 사용 가능한 도구(Tool)가 기하급수적으로 늘어나고 있습니다. 하지만 LLM의 제한된 Context Window에 수많은 도구 정의를 모두 넣을 수는 없습니다.

이 프로젝트는 **"도구를 찾기 위한 도구(Tool Search Tool)"** 패턴을 Azure OpenAI 기반의 **다층 하이브리드 검색**으로 구현하여, 수많은 MCP 서버 중 현재 태스크에 적합한 도구만 동적으로 로딩합니다.

v3.0.0에서는 **OpenAI Agents SDK 멀티 에이전트 오케스트레이션**, **Structured Outputs**, **o4-mini 추론 모델**, **RRF 하이브리드 검색 알고리즘**을 도입하여 한층 강력해졌습니다.

### ✨ 주요 기능

| 기능 | 설명 |
|-----|------|
| 🔍 **5단계 하이브리드 검색 + RRF** | BM25 → Sentence-Transformers → RRF Fusion → MCP Registry → GPT-5.2 |
| 🌐 **네이티브 MCP 서버 도구** | Responses API `type: "mcp"` 으로 원격 MCP 서버 직접 연동 |
| 🤝 **멀티 에이전트 오케스트레이션** | OpenAI Agents SDK 기반 전문 에이전트 핸드오프 |
| 🧠 **추론 모델 지원** | o4-mini 기반 깊은 사고 (수학, 논리, 코드 분석) |
| 📊 **Structured Outputs** | Pydantic v2 스키마 기반 구조화된 JSON 응답 |
| 🔄 **자동 대화 체이닝** | `previous_response_id` 서버 측 상태 관리 |
| ⚡ **스트리밍 응답** | 실시간 토큰 출력 + 도구 호출 루프 지원 |
| 🌍 **다국어 지원** | 한국어/영어/50+ 언어 쿼리 완벽 지원 |
| 🧠 **Sentence-Transformers** | 로컬 다국어 임베딩 (무료, 빠름) |
| 🌐 **MCP Registry API** | 공식 레지스트리에서 외부 도구 발견 |
| 💰 **비용 최적화** | BM25/임베딩/RRF로 LLM 호출 최소화 (~98% 절감) |
| 🔄 **동적 도구 로딩** | 필요한 도구만 런타임에 주입 |
| 📈 **검색 통계** | 검색 계층별 히트율 실시간 모니터링 |
| 🔭 **트레이싱/관찰성** | 에이전트 실행 추적 지원 |
| ✅ **103개 테스트** | 8개 시나리오 전체 커버리지 |

### 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                     사용자 쿼리 입력                                  │
│                "문서를 영어로 번역해 줘"                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│            🔍 Multi-layer Hybrid Search + RRF Fusion                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1️⃣ BM25 검색 (무료, ~1ms)                                          │
│     ├─ 점수 ≥ 5.0 → ✅ 즉시 반환                                     │
│     └─ 점수 < 5.0 → ⬇️ 다음 단계로                                   │
│                                                                      │
│  2️⃣ Sentence-Transformers (로컬, ~50ms)                             │
│     ├─ 모델: paraphrase-multilingual-MiniLM-L12-v2                   │
│     ├─ 유사도 ≥ 0.65 → ✅ 반환                                       │
│     └─ 유사도 < 0.65 → ⬇️ RRF 통합                                  │
│                                                                      │
│  3️⃣ 🔀 RRF (Reciprocal Rank Fusion)                                │
│     └─ BM25 + Sentence 결과를 RRF 점수로 통합 재순위                 │
│                                                                      │
│  4️⃣ MCP Registry API (외부, ~200ms)                                 │
│     └─ 공식 레지스트리에서 새로운 도구 발견                           │
│                                                                      │
│  5️⃣ GPT-5.2 LLM 추론 (~1-2s)                                       │
│     └─ 후보군 중 최적 도구 선택 & 재순위화                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              📦 동적 도구 로딩 & 실행                                 │
│                                                                      │
│  load_tool("azure_translator_tool")                                 │
│       └─→ 에이전트 컨텍스트에 도구 주입                               │
│       └─→ azure_translator_tool(text, target_lang)                  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│              🌐 네이티브 원격 MCP 서버 도구                           │
│                                                                      │
│  Responses API type: "mcp" 네이티브 통합                             │
│       └─→ Microsoft Learn MCP 서버                                   │
│       └─→ GitHub MCP 서버                                            │
│       └─→ 런타임에 mcp-add로 추가 가능                               │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│              🤝 멀티 에이전트 오케스트레이션 (v3.0 신규)              │
│                                                                      │
│  OpenAI Agents SDK 기반 전문 에이전트 핸드오프                       │
│       └─→ 트리아지 에이전트 → 검색 전문가 / 코드 전문가             │
│       └─→ 에이전트 트레이싱/관찰성 지원                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 🔬 검색 기술 스택

| 계층 | 기술 | 특징 |
|-----|------|------|
| 1️⃣ | **BM25** | 키워드 매칭, 무료, 초고속 (~1ms) |
| 2️⃣ | **Sentence-Transformers** | 다국어 시맨틱 검색, 로컬, 무료 |
| 3️⃣ | **🔀 RRF Fusion** | BM25 + Sentence 결과 통합 (k=60) |
| 4️⃣ | **MCP Registry API** | 공식 레지스트리, 외부 도구 발견 |
| 5️⃣ | **GPT-5.2 LLM** | 추론 기반 최종 선택 & 재순위화 |

**사용 모델**: `paraphrase-multilingual-MiniLM-L12-v2`
- 50+ 언어 지원 (한국어 포함)
- 384차원 임베딩
- 로컬 실행 (API 비용 없음)

### 📁 프로젝트 구조

```
dynamic_mcp_agent/
├── agent.py                 # 메인 에이전트 (v1 Responses API + Agents SDK)
│   ├── DynamicMCPAgent          # 동적 도구 검색/로딩 에이전트
│   │   ├── chat()                   # Responses API 비동기 대화
│   │   ├── chat_stream()            # 스트리밍 응답 (도구 호출 루프 포함)
│   │   ├── chat_sync()             # 동기 래퍼
│   │   ├── chat_with_reasoning()    # o4-mini 추론 모드
│   │   ├── chat_structured()        # Structured Outputs (Pydantic v2)
│   │   ├── _resolve_json_type()     # 제네릭 타입 → JSON Schema 변환
│   │   └── _dynamic_tool_injection()# 런타임 도구 주입
│   ├── create_agent()           # 팩토리 함수
│   └── DEFAULT_REMOTE_MCP_SERVERS   # 기본 MCP 서버 설정
├── main.py                  # CLI / Web / Demo / Stream / Reasoning / Agents 실행
│   ├── run_cli_mode()           # 대화형 CLI
│   ├── run_web_mode()           # Gradio 웹 UI
│   ├── run_demo_mode()          # 데모 시나리오
│   ├── run_stream_cli_mode()    # 스트리밍 CLI (단일 이벤트 루프)
│   ├── run_reasoning_cli_mode() # o4-mini 추론 CLI
│   └── run_agents_mode()        # Agents SDK 멀티 에이전트
├── requirements.txt         # Python 의존성 (openai≥1.93, openai-agents≥0.3)
├── __init__.py              # 패키지 초기화 (v3.0.0)
├── tests/
│   ├── __init__.py
│   └── test_all_scenarios.py    # 103개 시나리오별 테스트 (8개 시나리오)
└── lib/
    ├── __init__.py              # 라이브러리 exports
    ├── registry.py              # 다층 하이브리드 검색 + RRF 레지스트리
    │   ├── MCPRegistryClient        # MCP Registry API 클라이언트 (httpx/aiohttp)
    │   └── HybridToolRegistry       # 5단계 하이브리드 검색 + RRF
    │       ├── register()               # 단일 도구 등록 (중복 감지)
    │       ├── register_batch()         # 일괄 등록 (인덱스 1회 재구축)
    │       ├── search()                 # 다층 하이브리드 검색
    │       ├── _bm25_search()           # BM25 키워드 검색
    │       ├── _sentence_search()       # Sentence-Transformers 시맨틱 검색
    │       ├── _hybrid_search()         # RRF Fusion 통합 검색
    │       └── _llm_search()            # GPT-5.2 LLM 재순위화
    └── tools.py                 # 25개 Azure MCP 도구 정의
        ├── TOOL_DEFINITIONS         # 도구 메타데이터 리스트 (25개)
        ├── search_available_tools() # 도구 검색 함수
        ├── load_tool()              # 도구 로드 함수
        ├── register_tool()          # 도구 등록 함수
        ├── initialize_mcp_tools()   # 일괄 초기화 (register_batch)
        └── calculator_tool()        # AST 기반 안전한 수식 계산
```

### 🚀 빠른 시작

#### 1. 설치

```bash
git clone https://github.com/your-username/dynamic-mcp-agent.git
cd dynamic-mcp-agent

# 가상 환경 생성
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 의존성 설치
pip install -r requirements.txt
```

> 💡 **Sentence-Transformers**는 첫 실행 시 모델을 자동 다운로드합니다 (~100MB)

#### 2. 환경 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 참고하여 설정하세요:

```env
# =============================================================
# Dynamic MCP Agent v3.0 - Azure OpenAI v1 Responses API
# =============================================================
# v1 API 사용 (버전 관리 불필요, preview/latest만 지정)

# Azure OpenAI Configuration (Required)
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here

# v1 API 버전: "preview" (최신 프리뷰 기능) 또는 "latest" (GA 안정 버전)
AZURE_OPENAI_API_VERSION=preview

# 모델 배포명 (GPT-5.2 시리즈 권장)
# 사용 가능: gpt-5, gpt-5.1, gpt-5.2, gpt-5-mini, gpt-5-nano, gpt-5-pro
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.2

# 추론 모델 (Optional - --reasoning 모드)
AZURE_OPENAI_REASONING_MODEL=o4-mini

# Azure AI Search (Optional - for azure_ai_search_tool)
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-key-here
```

#### 3. 실행

```bash
# CLI 모드 (대화형)
python main.py

# 스트리밍 모드 (실시간 토큰 출력)
python main.py --stream

# 추론 모드 (o4-mini) ← v3.0 신규
python main.py --reasoning

# 멀티 에이전트 모드 (Agents SDK) ← v3.0 신규
python main.py --agents

# 데모 모드
python main.py --demo

# 웹 인터페이스 (Gradio)
python main.py --web
```

### 💬 사용 예시

```
👤 You: 번역 도구가 필요해요

🤖 Agent: 번역 도구를 검색하고 있습니다...
   [SENTENCE HIT] 쿼리: '번역', 유사도: 0.73

   검색 결과:
   1. azure_translator_tool: Azure Translator를 사용하여 텍스트를 번역합니다.

   이 도구를 로드해드릴까요?

👤 You: 네

🤖 Agent: ✅ 도구 'azure_translator_tool'가 로드되었습니다.
```

**원격 MCP 서버 추가:**
```
👤 You: mcp-add https://learn.microsoft.com/api/mcp microsoft_learn

🌐 MCP 서버 추가됨: microsoft_learn (https://learn.microsoft.com/api/mcp)
```

**멀티 에이전트 모드 (v3.0 신규):**
```bash
python main.py --agents
```
```
🤝 Dynamic MCP Agent - 멀티 에이전트 모드 (Agents SDK)
🤝 3개의 전문 에이전트가 준비되었습니다: 트리아지, 검색, 코드

👤 You: Azure Functions 관련 문서를 찾아줘

🤝 Agents: [트리아지 → 검색 전문가에게 핸드오프]
   Microsoft Learn에서 Azure Functions 관련 문서를 검색합니다...
```

### 🛠️ 등록된 도구 (25개)

| 카테고리 | 도구 | 설명 | 비고 |
|---------|------|------|------|
| 🔍 Search | `azure_ai_search_tool` | Azure AI Search 문서 검색 | |
| 🔍 Search | `bing_web_search_tool` | Bing 웹 검색 | |
| 🔍 Search | `github_search_tool` | GitHub 코드/저장소 검색 | |
| 🔍 Search | `azure_web_search_tool` | Responses API 내장 웹 검색 | v2.0 |
| 🗄️ Database | `azure_sql_query_tool` | Azure SQL 쿼리 실행 | |
| 🗄️ Database | `azure_cosmos_db_tool` | Cosmos DB 데이터 관리 | |
| 📦 Storage | `azure_blob_storage_tool` | Blob Storage 파일 관리 | |
| 🤖 AI | `azure_openai_embedding_tool` | 텍스트 임베딩 (3072차원) | |
| 🤖 AI | `azure_computer_vision_tool` | 이미지 분석 | |
| 🤖 AI | `azure_translator_tool` | 텍스트 번역 | |
| 🤖 AI | `azure_text_analytics_tool` | 텍스트 분석/감정 분석 | |
| 🤖 AI | `azure_form_recognizer_tool` | 문서 데이터 추출 | |
| 🤖 AI | `azure_speech_to_text_tool` | 음성→텍스트 변환 | |
| 🤖 AI | `azure_ai_foundry_agent_tool` | AI Foundry 멀티스텝 에이전트 | v2.0 |
| 🤖 AI | `azure_deep_research_tool` | o3 기반 심층 조사 | v2.0 |
| 🤖 AI | `azure_image_generation_tool` | GPT-Image-2 이미지 생성 | v2.0→v3.0 |
| 🤖 AI | `azure_ai_agent_service_tool` | Azure AI Agent Service 서버리스 | ✨ v3.0 |
| 🤖 AI | `azure_computer_use_tool` | CUA 기반 GUI 자동화 | ✨ v3.0 |
| 🤖 AI | `azure_realtime_audio_tool` | GPT-4o-realtime 실시간 음성 | ✨ v3.0 |
| 📊 AI | `structured_output_tool` | Pydantic 스키마 구조화 출력 | ✨ v3.0 |
| 🌐 MCP | `mcp_server_discovery_tool` | MCP Registry 서버 검색 | ✨ v3.0 |
| ⚡ Compute | `azure_function_invoke_tool` | Azure Function 호출 | |
| ⚡ Compute | `azure_code_interpreter_tool` | 코드 인터프리터 실행 | v2.0 |
| 🔧 Utility | `weather_api_tool` | 날씨 정보 조회 | |
| 🔧 Utility | `calculator_tool` | AST 기반 안전한 수학 계산 | |

### 🧪 테스트

```bash
# 전체 시나리오별 테스트 실행 (103개)
python -m pytest dynamic_mcp_agent/tests/test_all_scenarios.py -v

# BM25 검색 성능 테스트
python test_bm25.py
```

**테스트 시나리오 (8개, 103개 테스트):**

| 시나리오 | 설명 | 테스트 수 |
|---------|------|----------|
| 1. 모듈 임포트 | 패키지 버전, exports, MCP 서버 설정 | 4 |
| 2. 도구 기능 | 25개 도구 정의/구조, calculator 사칙연산·함수·보안, 개별 도구 실행 | 20 |
| 3. 레지스트리 | 등록/중복등록/일괄등록, BM25 검색(한/영/혼합), 토큰화, 통계, clear | 20 |
| 4. 에이전트 | 타입변환(기본/제네릭/Optional), 스키마 캐시·비변이, 도구 주입·실행·에러, 상태관리 | 23 |
| 5. 통합 | 검색→로드→실행 파이프라인, create_agent 팩토리, 사용자 도구 등록 | 15 |
| 6. 엣지 케이스 | 빈 쿼리, 유니코드, 긴 쿼리, 빈 레지스트리, 0나누기, 큰 수, Optional 파라미터 | 15 |
| 7. MCPRegistryClient | URL 빌드, 캐시 TTL 만료 | 3 |
| 8. main.py | 환경변수 체크 (부재/존재), argparse 구성 | 3 |

### 📊 검색 성능

```
======================================================================
[SEARCH] Multi-layer Hybrid Search + RRF Test
   BM25 -> Sentence-Transformers -> RRF -> MCP Registry -> GPT-5.2 LLM
   Korean/English multilingual support
======================================================================

[INFO] Model Info:
   - Sentence-Transformers: paraphrase-multilingual-MiniLM-L12-v2
   - MCP Registry API: ENABLED
   - LLM Model: gpt-5.2
   - RRF k-parameter: 60
   - Registered tools: 25

----------------------------------------------------------------------
[KO] Korean Query Test
----------------------------------------------------------------------

[Q] Query: '문서에서 텍스트 추출하고 싶어'
   1. azure_form_recognizer_tool: Azure Form Recognizer를 사용하여 문서에서 데이터를 추출...

[Q] Query: '영어를 한국어로 번역해줘'
   1. azure_translator_tool: Azure Translator를 사용하여 텍스트를 번역합니다...
```

### 🔧 커스텀 도구 추가

```python
from dynamic_mcp_agent import register_tool

def my_custom_tool(param1: str) -> dict:
    """나만의 커스텀 도구입니다."""
    return {"result": f"처리됨: {param1}"}

# 도구 등록 (한국어 태그 지원)
register_tool(
    my_custom_tool,
    category="custom",
    tags=["나만의", "도구", "custom", "tool"]
)
```

### 🌐 원격 MCP 서버 연동

```python
from dynamic_mcp_agent import create_agent

# 기본 Microsoft Learn + GitHub MCP 서버 포함하여 에이전트 생성
agent = create_agent(enable_remote_mcp=True)

# 런타임에 추가 MCP 서버 연결
agent.add_remote_mcp_server(
    server_url="https://your-mcp-server.com/sse",
    server_label="my_mcp_server",
    server_description="커스텀 MCP 서버"
)

# Responses API가 네이티브로 MCP 서버 도구를 호출
response = agent.chat_sync("Microsoft Learn에서 Azure Functions 관련 문서를 찾아줘")
```

### 🤝 멀티 에이전트 (v3.0 신규)

```python
from agents import Agent, Runner
from dynamic_mcp_agent import create_agent

# 기본 에이전트 생성
base_agent = create_agent(enable_tracing=True)

# 전문 에이전트 정의
search_agent = Agent(name="검색 전문가", instructions="...", model=base_agent.model)
code_agent   = Agent(name="코드 전문가", instructions="...", model=base_agent.model)
triage_agent = Agent(
    name="트리아지",
    instructions="사용자 요청을 분석하여 적절한 전문가에게 전달",
    handoffs=[search_agent, code_agent],
)

# 실행
result = Runner.run_sync(triage_agent, "Azure Functions 관련 코드를 분석해 줘")
print(result.final_output)
```

### 🌐 MCP Registry에서 외부 도구 발견

```python
from dynamic_mcp_agent import registry

# 외부 MCP 서버 검색 (API v0.1, httpx 기반)
external_tools = registry.discover_external_tools("database", limit=5)
for tool in external_tools:
    server = tool.get("server", {})
    print(f"{server.get('name')}: {server.get('description')}")
```

### 📈 비용 최적화 효과

| 방식 | 100회 검색 비용 | 정확도 |
|-----|----------------|-------|
| LLM만 사용 | ~$3.00 | ⭐⭐⭐⭐⭐ |
| **다층 하이브리드 + RRF** | **~$0.05** | ⭐⭐⭐⭐⭐ |
| BM25만 사용 | $0 | ⭐⭐⭐ |

> 💡 **~98% 비용 절감** + 동일한 정확도 (Sentence-Transformers 로컬 실행 + RRF 통합)

### ⚙️ 코드 아키텍처

**agent.py (v3.0 — Responses API + Agents SDK):**

```python
# v1 API: OpenAI 클라이언트 + base_url 방식
from openai import OpenAI

base_url = f"{endpoint}/openai/v1/"
client = OpenAI(api_key=key, base_url=base_url, default_query={"api-version": "preview"})

# Responses API 호출 + previous_response_id 자동 체이닝
response = client.responses.create(
    model="gpt-5.2",
    input=[{"role": "user", "content": message}],
    tools=tools,                             # 네이티브 MCP 서버 도구 포함
    previous_response_id=last_response_id,   # 대화 자동 체이닝
)

# 네이티브 원격 MCP 서버 도구 정의
mcp_tool = {
    "type": "mcp",
    "server_label": "microsoft_learn",
    "server_url": "https://learn.microsoft.com/api/mcp",
    "require_approval": "never",
}

# 제네릭 타입 변환 (Optional, List, Dict 등)
DynamicMCPAgent._resolve_json_type(Optional[List[str]])  # → "array"
```

**registry.py 주요 클래스:**

```python
# MCPRegistryClient - 공식 MCP Registry API 클라이언트 (httpx 우선)
class MCPRegistryClient:
    BASE_URL = "https://registry.modelcontextprotocol.io"
    API_VERSION = "v0.1"

    async def search_servers(self, query, limit=10): ...   # httpx → aiohttp 폴백
    async def get_server_details(self, server_name): ...
    async def list_all_servers(self, limit=30): ...

# HybridToolRegistry - 5단계 하이브리드 검색 + RRF
class HybridToolRegistry:
    BM25_CONFIDENCE_THRESHOLD = 5.0
    EMBEDDING_SIMILARITY_THRESHOLD = 0.65
    RRF_K = 60                               # Reciprocal Rank Fusion 파라미터

    def search(self, query, top_k=5, strategy="hybrid"): ...
    def register(self, tool, name, ...): ...     # 중복 감지 + in-place 업데이트
    def register_batch(self, tools): ...         # 인덱스 1회 재구축
```

**tools.py 도구 등록:**

```python
# 도구 정의를 리스트로 관리하여 유지보수성 향상
TOOL_DEFINITIONS = [
    (azure_ai_search_tool, "search", ["azure", "search", "문서검색"]),
    (azure_translator_tool, "ai", ["translate", "번역", "언어"]),
    # v3.0 신규 도구
    (azure_ai_agent_service_tool, "ai", ["azure", "agent", "서버리스"]),
    (azure_computer_use_tool, "ai", ["CUA", "GUI", "자동화"]),
    (mcp_server_discovery_tool, "mcp", ["mcp", "서버검색"]),
    (structured_output_tool, "ai", ["structured", "pydantic", "스키마"]),
    (azure_realtime_audio_tool, "ai", ["realtime", "음성대화"]),
    # ... 총 25개 도구
]

# 모듈 레벨 안전한 계산기 상수 (호출당 재생성 방지)
_SAFE_OPERATORS = { ast.Add: operator.add, ... }
_SAFE_FUNCTIONS = { 'sqrt': math.sqrt, 'pi': math.pi, ... }
```

---

## English

### 📌 Overview

As the MCP (Model Context Protocol) ecosystem expands, the number of available tools is growing exponentially. However, you can't fit all tool definitions into an LLM's limited context window.

This project implements the **"Tool Search Tool"** pattern with **multi-layer hybrid search + RRF fusion** based on Azure OpenAI, dynamically loading only the tools suitable for the current task.

v3.0 adds **OpenAI Agents SDK multi-agent orchestration**, **Structured Outputs**, **o4-mini reasoning model**, and **Reciprocal Rank Fusion (RRF)** hybrid search.

### ✨ Key Features

- 🔍 **5-Layer Hybrid Search + RRF**: BM25 → Sentence-Transformers → RRF Fusion → MCP Registry → GPT-5.2
- 🌐 **Native Remote MCP Server Tools**: Responses API `type: "mcp"` integration
- 🤝 **Multi-Agent Orchestration**: OpenAI Agents SDK with agent handoffs
- 🧠 **Reasoning Model**: o4-mini for complex math, logic, code analysis
- 📊 **Structured Outputs**: Pydantic v2 schema-based JSON responses
- 🔄 **Auto Conversation Chaining**: `previous_response_id` server-side state
- ⚡ **Streaming Responses**: `--stream` CLI with tool call loop support
- 🌍 **Multi-language**: Full Korean/English/50+ languages support
- 🧠 **Sentence-Transformers**: Local multilingual embeddings (free, fast)
- 🌐 **MCP Registry API**: Discover external tools from official registry
- 💰 **Cost Optimized**: ~98% cost reduction with local embeddings + RRF
- 🔄 **Dynamic Loading**: Runtime tool injection
- 🔭 **Tracing**: Agent execution observability
- 🆕 **25 Azure tools** including AI Agent Service, Computer Use, MCP Discovery, Structured Output, Realtime Audio

### 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/dynamic-mcp-agent.git
cd dynamic-mcp-agent
pip install -r requirements.txt

# Configure environment — create .env file
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.2

# Run
python main.py                # CLI mode
python main.py --stream       # Streaming mode
python main.py --reasoning    # o4-mini reasoning (NEW v3.0)
python main.py --agents       # Multi-agent mode (NEW v3.0)
python main.py --web          # Gradio web UI
python main.py --demo         # Demo scenarios

# Test (103 tests, 8 scenarios)
python -m pytest dynamic_mcp_agent/tests/test_all_scenarios.py -v
```

### 🔬 Search Technology Stack

| Layer | Technology | Features |
|-------|-----------|----------|
| 1️⃣ | **BM25** | Keyword matching, free, ultra-fast (~1ms) |
| 2️⃣ | **Sentence-Transformers** | Multilingual semantic search, local, free |
| 3️⃣ | **🔀 RRF Fusion** | Reciprocal Rank Fusion of BM25 + Sentence (k=60) |
| 4️⃣ | **MCP Registry API** | Official registry, external tool discovery |
| 5️⃣ | **GPT-5.2 LLM** | Reasoning-based final selection & reranking |

### 📜 Changelog

#### v3.0.0 (2026-02-26)
- **NEW**: OpenAI Agents SDK multi-agent orchestration (`--agents` mode)
- **NEW**: Structured Outputs with Pydantic v2 schema validation
- **NEW**: o4-mini reasoning model support (`--reasoning` mode)
- **NEW**: Reciprocal Rank Fusion (RRF) hybrid search algorithm
- **NEW**: httpx-based async HTTP (replaces aiohttp, Streamable HTTP support)
- **NEW**: Agent tracing/observability support
- **NEW**: 5 new tools — AI Agent Service, Computer Use, MCP Discovery, Structured Output, Realtime Audio
- **FIX**: `_get_active_tools_schema()` cache mutation bug (returns `list()` copy)
- **FIX**: `register()` / `register_batch()` duplicate entry consistency
- **FIX**: `_resolve_json_type()` — proper `Optional[X]`, `List[str]`, `Dict[K,V]` handling
- **FIX**: MCPRegistryClient httpx migration for `get_server_details()` / `list_all_servers()`
- **PERF**: Module-level calculator constants (no per-call recreation)
- **PERF**: Stream CLI single event loop optimization
- **CHANGED**: Default model `gpt-5` → `gpt-5.2`
- **CHANGED**: `openai>=1.93.0`, `openai-agents>=0.3.0`, `pydantic>=2.10.0`, `httpx>=0.28.0`
- **CHANGED**: Default image model `gpt-image-1.5` → `gpt-image-2`
- **CHANGED**: 25 tools (20 → 25)

#### v2.0.0 (2026-02-07)
- **BREAKING**: Migrated from `AzureOpenAI` to `OpenAI` with `base_url` (v1 API)
- **BREAKING**: Replaced `chat.completions.create()` with `responses.create()`
- **BREAKING**: Removed manual `conversation_history` → `previous_response_id` auto-chaining
- **NEW**: Native remote MCP server tools via Responses API `type: "mcp"`
- **NEW**: Streaming responses (`--stream`, `chat_stream()`)
- **NEW**: Runtime MCP server addition (`mcp-add`, `add_remote_mcp_server()`)
- **NEW**: 5 new tools — AI Foundry Agent, Deep Research, Web Search, Code Interpreter, Image Generation
- **CHANGED**: Default model `gpt-4o` → `gpt-5`
- **CHANGED**: Embedding model → `text-embedding-3-large` (3072 dims)
- **CHANGED**: `openai>=1.86.0`, `gradio>=5.0.0`

#### v1.0.0 (2025)
- Initial release: BM25 + Sentence-Transformers + MCP Registry + GPT-4.1 hybrid search
- Azure OpenAI Chat Completions API
- 15 Azure MCP tools

### 📚 References

- [Azure OpenAI v1 Responses API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses)
- [Azure OpenAI API Version Lifecycle](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Implementing Dynamic Tool Search Tool](https://medium.com/google-cloud/implementing-anthropic-style-dynamic-tool-search-tool-f39d02a35139)
- [MCP Registry](https://registry.modelcontextprotocol.io)
- [MCP Registry API Documentation](https://registry.modelcontextprotocol.io/docs)
- [Sentence-Transformers](https://www.sbert.net/)
- [ToolGen: Unified Tool Retrieval and Calling (ICLR 2025)](https://github.com/Reason-Wang/ToolGen)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Issues and Pull Requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

Made with ❤️ using Azure OpenAI v1 Responses API + GPT-5.2 + Agents SDK + Sentence-Transformers + MCP Registry
