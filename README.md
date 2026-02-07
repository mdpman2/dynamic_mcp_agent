# 🔍 Dynamic MCP Agent v2.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI%20v1-0078D4.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![MCP Registry](https://img.shields.io/badge/MCP-Registry-green.svg)](https://registry.modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Azure OpenAI v1 Responses API 기반 다층 하이브리드 도구 검색 에이전트**  
> BM25 → Sentence-Transformers → MCP Registry API → GPT-5 LLM 4단계 검색으로 최적의 도구 동적 로딩  
> 네이티브 원격 MCP 서버 도구 통합 · 스트리밍 응답 · previous_response_id 자동 대화 체이닝

[English](#english) | [한국어](#한국어)

---

## ⚡ v2.0.0 업데이트 (2026-02-07)

### Breaking Changes
| 항목 | v1.0.0 (이전) | v2.0.0 (현재) |
|------|--------------|--------------|
| API 클라이언트 | `AzureOpenAI()` | `OpenAI(base_url=...)` |
| API 호출 | `chat.completions.create()` | `responses.create()` |
| 대화 관리 | 수동 `conversation_history` 리스트 | `previous_response_id` 자동 체이닝 |
| API 버전 | `2024-08-01-preview` (월별 관리) | `preview` 또는 `latest` (v1 자동관리) |
| 기본 모델 | gpt-4o / gpt-4.1 | **gpt-5** |
| 임베딩 모델 | text-embedding-ada-002 | **text-embedding-3-large** (3072차원) |
| 도구 스키마 | `{"function": {"name": "..."}}` (중첩) | `{"name": "..."}` (평탄화) |
| 도구 응답 | `{"role": "tool"}` | `{"type": "function_call_output"}` |

### 신규 기능
- 🌐 **네이티브 원격 MCP 서버 통합** — Responses API `type: "mcp"` 도구로 직접 연동
- ⚡ **스트리밍 응답** — `--stream` CLI 모드, `chat_stream()` 비동기 제너레이터
- 🔌 **런타임 MCP 서버 추가** — `mcp-add <url> <label>` CLI 명령
- 🤖 **5개 신규 도구** — AI Foundry Agent, Deep Research, Web Search, Code Interpreter, Image Generation
- 📦 **도구 20개로 확장** (15 → 20)

### 참고 문서
- [Azure OpenAI v1 Responses API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses)
- [Azure OpenAI API 버전 라이프사이클](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle)

---

## 한국어

### 📌 개요

MCP(Model Context Protocol) 생태계가 확장되면서 사용 가능한 도구(Tool)가 기하급수적으로 늘어나고 있습니다. 하지만 LLM의 제한된 Context Window에 수많은 도구 정의를 모두 넣을 수는 없습니다.

이 프로젝트는 **"도구를 찾기 위한 도구(Tool Search Tool)"** 패턴을 Azure OpenAI 기반의 **다층 하이브리드 검색**으로 구현하여, 수많은 MCP 서버 중 현재 태스크에 적합한 도구만 동적으로 로딩합니다.

v2.0.0에서는 **Azure OpenAI v1 Responses API**를 채택하여 대화 상태를 서버 측에서 자동 관리하고, **네이티브 원격 MCP 서버 도구**를 직접 연동할 수 있게 되었습니다.

### ✨ 주요 기능

| 기능 | 설명 |
|-----|------|
| 🔍 **4단계 하이브리드 검색** | BM25 → Sentence-Transformers → MCP Registry → GPT-5 |
| 🌐 **네이티브 MCP 서버 도구** | Responses API `type: "mcp"` 으로 원격 MCP 서버 직접 연동 |
| 🔄 **자동 대화 체이닝** | `previous_response_id` 서버 측 상태 관리 |
| ⚡ **스트리밍 응답** | 실시간 토큰 출력 (CLI `--stream`, `chat_stream()`) |
| 🌍 **다국어 지원** | 한국어/영어/다국어 쿼리 완벽 지원 |
| 🧠 **Sentence-Transformers** | 로컬 다국어 임베딩 (무료, 빠름) |
| 🌐 **MCP Registry API** | 공식 레지스트리에서 외부 도구 발견 |
| 💰 **비용 최적화** | BM25/임베딩으로 1차 필터링하여 LLM 호출 최소화 |
| 🔄 **동적 도구 로딩** | 필요한 도구만 런타임에 주입 |
| 📊 **검색 통계** | 검색 계층별 히트율 실시간 모니터링 |

### 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                     사용자 쿼리 입력                                  │
│                "문서를 영어로 번역해 줘"                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  🔍 Multi-layer Hybrid Search                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1️⃣ BM25 검색 (무료, ~1ms)                                          │
│     ├─ 점수 ≥ 5.0 → ✅ 즉시 반환                                     │
│     └─ 점수 < 5.0 → ⬇️ 다음 단계로                                   │
│                                                                      │
│  2️⃣ Sentence-Transformers (로컬, ~50ms)                             │
│     ├─ 모델: paraphrase-multilingual-MiniLM-L12-v2                   │
│     ├─ 유사도 ≥ 0.65 → ✅ 반환                                       │
│     └─ 유사도 < 0.65 → ⬇️ 다음 단계로                                │
│                                                                      │
│  3️⃣ MCP Registry API (외부, ~200ms)                                 │
│     └─ 공식 레지스트리에서 새로운 도구 발견                           │
│                                                                      │
│  4️⃣ GPT-5 LLM 추론 (~1-2s)                                          │
│     └─ BM25 + 임베딩 후보군 중 최적 도구 선택 & 재순위화              │
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
│              🌐 네이티브 원격 MCP 서버 도구 (v2.0 신규)               │
│                                                                      │
│  Responses API type: "mcp" 네이티브 통합                             │
│       └─→ Microsoft Learn MCP 서버                                   │
│       └─→ 런타임에 mcp-add로 추가 가능                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 🔬 검색 기술 스택

| 계층 | 기술 | 특징 |
|-----|------|------|
| 1️⃣ | **BM25** | 키워드 매칭, 무료, 초고속 (~1ms) |
| 2️⃣ | **Sentence-Transformers** | 다국어 시맨틱 검색, 로컬, 무료 |
| 3️⃣ | **MCP Registry API** | 공식 레지스트리, 외부 도구 발견 |
| 4️⃣ | **GPT-5 LLM** | 추론 기반 최종 선택 & 재순위화 |

**사용 모델**: `paraphrase-multilingual-MiniLM-L12-v2`
- 50+ 언어 지원 (한국어 포함)
- 384차원 임베딩
- 로컬 실행 (API 비용 없음)

### 📁 프로젝트 구조

```
dynamic_mcp_agent/
├── agent.py                 # 메인 에이전트 (v1 Responses API + 네이티브 MCP)
├── main.py                  # CLI / Web / Demo / Stream 실행
├── requirements.txt         # Python 의존성 (openai>=1.86.0)
├── test_bm25.py             # 검색 테스트 스크립트
├── __init__.py              # 패키지 초기화 (v2.0.0)
└── lib/
    ├── registry.py          # 다층 하이브리드 검색 레지스트리
    │   ├── MCPRegistryClient    # MCP Registry API 클라이언트 (v0.1)
    │   └── HybridToolRegistry   # 4단계 하이브리드 검색
    └── tools.py             # 20개 Azure MCP 도구 정의
        └── TOOL_DEFINITIONS     # 도구 메타데이터 리스트
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
# Dynamic MCP Agent - Azure OpenAI v1 Responses API Configuration
# =============================================================
# 2026 최신: v1 API 사용 (버전 관리 불필요, preview/latest만 지정)
# 참고: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle

# Azure OpenAI Configuration (Required)
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here

# v1 API 버전: "preview" (최신 프리뷰 기능) 또는 "latest" (GA 안정 버전)
AZURE_OPENAI_API_VERSION=preview

# 모델 배포명 (GPT-5 시리즈 권장)
# 사용 가능: gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro, gpt-5.1, gpt-5.2
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5

# Azure AI Search (Optional - for azure_ai_search_tool)
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-key-here
```

> ⚠️ **v1.0 → v2.0 마이그레이션**: `API_VERSION`을 날짜 형식(예: `2024-08-01-preview`)에서 `preview` 또는 `latest`로 변경하세요.

#### 3. 실행

```bash
# CLI 모드 (대화형)
python main.py

# 스트리밍 모드 (실시간 토큰 출력) ← v2.0 신규
python main.py --stream

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

**원격 MCP 서버 추가 (v2.0 신규):**
```
👤 You: mcp-add https://learn.microsoft.com/api/mcp microsoft_learn

🌐 MCP 서버 추가됨: microsoft_learn (https://learn.microsoft.com/api/mcp)
```

### 🛠️ 등록된 도구 (20개)

| 카테고리 | 도구 | 설명 | 비고 |
|---------|------|------|------|
| 🔍 Search | `azure_ai_search_tool` | Azure AI Search 문서 검색 | |
| 🔍 Search | `bing_web_search_tool` | Bing 웹 검색 | |
| 🔍 Search | `github_search_tool` | GitHub 코드/저장소 검색 | |
| 🔍 Search | `azure_web_search_tool` | Responses API 내장 웹 검색 | ✨ v2.0 신규 |
| 🗄️ Database | `azure_sql_query_tool` | Azure SQL 쿼리 실행 | |
| 🗄️ Database | `azure_cosmos_db_tool` | Cosmos DB 데이터 관리 | |
| 📦 Storage | `azure_blob_storage_tool` | Blob Storage 파일 관리 | |
| 🤖 AI | `azure_openai_embedding_tool` | 텍스트 임베딩 (3072차원) | ⬆️ 모델 업그레이드 |
| 🤖 AI | `azure_computer_vision_tool` | 이미지 분석 | |
| 🤖 AI | `azure_translator_tool` | 텍스트 번역 | |
| 🤖 AI | `azure_text_analytics_tool` | 텍스트 분석 | |
| 🤖 AI | `azure_form_recognizer_tool` | 문서 데이터 추출 | |
| 🤖 AI | `azure_speech_to_text_tool` | 음성→텍스트 변환 | |
| 🤖 AI | `azure_ai_foundry_agent_tool` | AI Foundry 멀티스텝 에이전트 | ✨ v2.0 신규 |
| 🤖 AI | `azure_deep_research_tool` | o3 기반 심층 조사 | ✨ v2.0 신규 |
| 🤖 AI | `azure_image_generation_tool` | GPT-Image 이미지 생성 | ✨ v2.0 신규 |
| ⚡ Compute | `azure_function_invoke_tool` | Azure Function 호출 | |
| ⚡ Compute | `azure_code_interpreter_tool` | 코드 인터프리터 실행 | ✨ v2.0 신규 |
| 🔧 Utility | `weather_api_tool` | 날씨 정보 조회 | |
| 🔧 Utility | `calculator_tool` | 수학 계산 | |

### 📊 검색 성능

```bash
python test_bm25.py
```

```
======================================================================
[SEARCH] Multi-layer Hybrid Search Test
   BM25 -> Sentence-Transformers -> MCP Registry -> GPT-5 LLM
   Korean/English multilingual support
======================================================================

[INFO] Model Info:
   - Sentence-Transformers: paraphrase-multilingual-MiniLM-L12-v2
   - MCP Registry API: ENABLED
   - LLM Model: gpt-5
   - Registered tools: 20

----------------------------------------------------------------------
[KO] Korean Query Test
----------------------------------------------------------------------

[Q] Query: '문서에서 텍스트 추출하고 싶어'
   1. azure_form_recognizer_tool: Azure Form Recognizer를 사용하여 문서에서 데이터를 추출...

[Q] Query: '영어를 한국어로 번역해줘'
   1. azure_translator_tool: Azure Translator를 사용하여 텍스트를 번역합니다...

======================================================================
[STATS] Search Statistics
======================================================================
   BM25 hits: 3 (30%)
   Sentence-Transformers hits: 5 (50%)
   MCP Registry hits: 0 (0%)
   LLM hits: 2 (20%)
   Total searches: 10
```

### 🔧 커스텀 도구 추가

```python
from lib.tools import register_tool

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

### 🌐 원격 MCP 서버 연동 (v2.0 신규)

```python
from dynamic_mcp_agent import create_agent

# 기본 Microsoft Learn MCP 서버 포함하여 에이전트 생성
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

### 🌐 MCP Registry에서 외부 도구 발견

```python
from lib.registry import registry

# 외부 MCP 서버 검색 (API v0.1)
external_tools = registry.discover_external_tools("database", limit=5)
for tool in external_tools:
    server = tool.get("server", {})
    print(f"{server.get('name')}: {server.get('description')}")
```

### 📈 비용 최적화 효과

| 방식 | 100회 검색 비용 | 정확도 |
|-----|----------------|-------|
| LLM만 사용 | ~$3.00 | ⭐⭐⭐⭐⭐ |
| **다층 하이브리드** | **~$0.05** | ⭐⭐⭐⭐⭐ |
| BM25만 사용 | $0 | ⭐⭐⭐ |

> 💡 **~98% 비용 절감** + 동일한 정확도 (Sentence-Transformers 로컬 실행)

### ⚙️ 코드 아키텍처

**agent.py (v2.0 — Responses API):**

```python
# v1 API: OpenAI 클라이언트 + base_url 방식
from openai import OpenAI

base_url = f"{endpoint}/openai/v1/"
client = OpenAI(api_key=key, base_url=base_url, default_query={"api-version": "preview"})

# Responses API 호출 + previous_response_id 자동 체이닝
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": message}],
    tools=tools,                             # 네이티브 MCP 서버 도구 포함
    previous_response_id=last_response_id,   # 대화 자동 체이닝
)

# 도구 결과는 function_call_output 형식으로 반환
tool_results = [{
    "type": "function_call_output",
    "call_id": output.call_id,
    "output": result_json
}]

# 네이티브 원격 MCP 서버 도구 정의
mcp_tool = {
    "type": "mcp",
    "server_label": "microsoft_learn",
    "server_url": "https://learn.microsoft.com/api/mcp",
    "require_approval": "never",
}
```

**registry.py 주요 클래스:**

```python
# MCPRegistryClient - 공식 MCP Registry API 클라이언트 (v0.1)
class MCPRegistryClient:
    BASE_URL = "https://registry.modelcontextprotocol.io"
    API_VERSION = "v0.1"
    
    async def search_servers(self, query: str, limit: int = 10): ...

# HybridToolRegistry - 4단계 하이브리드 검색
class HybridToolRegistry:
    BM25_CONFIDENCE_THRESHOLD = 5.0
    EMBEDDING_SIMILARITY_THRESHOLD = 0.65
    
    def search(self, query, top_k=5, strategy="hybrid"): ...
    def discover_external_tools(self, query, limit=5): ...
```

**tools.py 도구 등록:**

```python
# 도구 정의를 리스트로 관리하여 유지보수성 향상
TOOL_DEFINITIONS = [
    (azure_ai_search_tool, "search", ["azure", "search", "문서검색"]),
    (azure_translator_tool, "ai", ["translate", "번역", "언어"]),
    # v2.0 신규 도구
    (azure_ai_foundry_agent_tool, "ai", ["foundry", "agent", "멀티스텝"]),
    (azure_deep_research_tool, "ai", ["research", "deep", "조사", "o3"]),
    (azure_web_search_tool, "search", ["web", "search", "실시간", "grounding"]),
    (azure_code_interpreter_tool, "compute", ["code", "interpreter", "python"]),
    (azure_image_generation_tool, "ai", ["image", "generation", "gpt-image"]),
    # ... 총 20개 도구
]
```

---

## English

### 📌 Overview

As the MCP (Model Context Protocol) ecosystem expands, the number of available tools is growing exponentially. However, you can't fit all tool definitions into an LLM's limited context window.

This project implements the **"Tool Search Tool"** pattern with **multi-layer hybrid search** based on Azure OpenAI, dynamically loading only the tools suitable for the current task.

v2.0 adopts the **Azure OpenAI v1 Responses API** for server-side conversation state management and **native remote MCP server tool integration**.

### ✨ Key Features

- 🔍 **4-Layer Hybrid Search**: BM25 → Sentence-Transformers → MCP Registry → GPT-5
- 🌐 **Native Remote MCP Server Tools**: Responses API `type: "mcp"` integration
- 🔄 **Auto Conversation Chaining**: `previous_response_id` server-side state
- ⚡ **Streaming Responses**: `--stream` CLI mode, `chat_stream()` async generator
- 🌍 **Multi-language**: Full Korean/English/50+ languages support
- 🧠 **Sentence-Transformers**: Local multilingual embeddings (free, fast)
- 🌐 **MCP Registry API**: Discover external tools from official registry
- 💰 **Cost Optimized**: ~98% cost reduction with local embeddings
- 🔄 **Dynamic Loading**: Runtime tool injection
- 🆕 **20 Azure tools** including AI Foundry Agent, Deep Research, Web Search, Code Interpreter, Image Generation

### 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/dynamic-mcp-agent.git
cd dynamic-mcp-agent
pip install -r requirements.txt

# Configure environment — create .env file (see Korean section above for full template)
# Set required variables:
#   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
#   AZURE_OPENAI_API_KEY=your-api-key
#   AZURE_OPENAI_API_VERSION=preview          (v1 API)
#   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5        (GPT-5 series)

# Run
python main.py             # CLI mode
python main.py --stream    # Streaming mode (NEW in v2.0)
python main.py --web       # Gradio web UI
python main.py --demo      # Demo scenarios
```

### 🔬 Search Technology Stack

| Layer | Technology | Features |
|-------|-----------|----------|
| 1️⃣ | **BM25** | Keyword matching, free, ultra-fast (~1ms) |
| 2️⃣ | **Sentence-Transformers** | Multilingual semantic search, local, free |
| 3️⃣ | **MCP Registry API** | Official registry, external tool discovery |
| 4️⃣ | **GPT-5 LLM** | Reasoning-based final selection & reranking |

**Model Used**: `paraphrase-multilingual-MiniLM-L12-v2`
- 50+ languages supported (including Korean, Japanese, Chinese)
- 384-dimensional embeddings
- Local execution (no API cost)

### 📜 Changelog

#### v2.0.0 (2026-02-07)
- **BREAKING**: Migrated from `AzureOpenAI` client to `OpenAI` with `base_url` (v1 API)
- **BREAKING**: Replaced `chat.completions.create()` with `responses.create()` (Responses API)
- **BREAKING**: Removed manual `conversation_history` in favor of `previous_response_id` auto-chaining
- **NEW**: Native remote MCP server tools via Responses API `type: "mcp"`
- **NEW**: Streaming responses (`--stream`, `chat_stream()`)
- **NEW**: Runtime MCP server addition (`mcp-add` command, `add_remote_mcp_server()`)
- **NEW**: 5 new tools — AI Foundry Agent, Deep Research, Web Search, Code Interpreter, Image Generation
- **CHANGED**: Default model `gpt-4o` → `gpt-5`
- **CHANGED**: Default API version `2024-08-01-preview` → `preview`
- **CHANGED**: Embedding model → `text-embedding-3-large` (3072 dimensions)
- **CHANGED**: Tool schema flattened (`name` at top level per Responses API format)
- **CHANGED**: Tool results use `function_call_output` type
- **CHANGED**: `openai>=1.86.0`, `gradio>=5.0.0`

#### v1.0.0 (2025)
- Initial release: BM25 + Sentence-Transformers + MCP Registry + GPT-4.1 hybrid search
- Azure OpenAI Chat Completions API
- 15 Azure MCP tools

### 📚 References

- [Azure OpenAI v1 Responses API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses)
- [Azure OpenAI API Version Lifecycle](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle)
- [Implementing Anthropic-style Dynamic Tool Search Tool](https://medium.com/google-cloud/implementing-anthropic-style-dynamic-tool-search-tool-f39d02a35139)
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

Made with ❤️ using Azure OpenAI v1 Responses API + GPT-5 + Sentence-Transformers + MCP Registry
