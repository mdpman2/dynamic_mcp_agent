# 🔍 Dynamic MCP Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![MCP Registry](https://img.shields.io/badge/MCP-Registry-green.svg)](https://registry.modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Azure OpenAI 기반 다층 하이브리드 도구 검색 에이전트**  
> BM25 → Sentence-Transformers → MCP Registry API → GPT-4.1 LLM 4단계 검색으로 최적의 도구 동적 로딩

[English](#english) | [한국어](#한국어)

---

## 한국어

### 📌 개요

MCP(Model Context Protocol) 생태계가 확장되면서 사용 가능한 도구(Tool)가 기하급수적으로 늘어나고 있습니다. 하지만 LLM의 제한된 Context Window에 수많은 도구 정의를 모두 넣을 수는 없습니다.

이 프로젝트는 **"도구를 찾기 위한 도구(Tool Search Tool)"** 패턴을 Azure OpenAI 기반의 **다층 하이브리드 검색**으로 구현하여, 수많은 MCP 서버 중 현재 태스크에 적합한 도구만 동적으로 로딩합니다.

### ✨ 주요 기능

| 기능 | 설명 |
|-----|------|
| 🔍 **4단계 하이브리드 검색** | BM25 → Sentence-Transformers → MCP Registry → GPT-4.1 |
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
│  4️⃣ GPT-4.1 LLM 추론 (~1-2s)                                        │
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
└─────────────────────────────────────────────────────────────────────┘
```

### 🔬 검색 기술 스택

| 계층 | 기술 | 특징 |
|-----|------|------|
| 1️⃣ | **BM25** | 키워드 매칭, 무료, 초고속 (~1ms) |
| 2️⃣ | **Sentence-Transformers** | 다국어 시맨틱 검색, 로컬, 무료 |
| 3️⃣ | **MCP Registry API** | 공식 레지스트리, 외부 도구 발견 |
| 4️⃣ | **GPT-4.1 LLM** | 추론 기반 최종 선택 & 재순위화 |

**사용 모델**: `paraphrase-multilingual-MiniLM-L12-v2`
- 50+ 언어 지원 (한국어 포함)
- 384차원 임베딩
- 로컬 실행 (API 비용 없음)

### 📁 프로젝트 구조

```
dynamic_mcp_agent/
├── agent.py                 # 메인 에이전트 (Azure OpenAI Function Calling)
├── main.py                  # CLI / Web / Demo 실행
├── requirements.txt         # Python 의존성
├── .env.example             # 환경 변수 템플릿
├── test_bm25.py             # 검색 테스트 스크립트
└── lib/
    ├── registry.py          # 다층 하이브리드 검색 레지스트리
    │   ├── MCPRegistryClient    # MCP Registry API 클라이언트 (v0.1)
    │   └── HybridToolRegistry   # 4단계 하이브리드 검색
    └── tools.py             # 15개 Azure MCP 도구 정의
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

```bash
# .env 파일 생성
cp .env.example .env
```

`.env` 파일 편집:
```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
```

#### 3. 실행

```bash
# CLI 모드 (대화형)
python main.py

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

### 🛠️ 등록된 도구 (15개)

| 카테고리 | 도구 | 설명 |
|---------|------|------|
| 🔍 Search | `azure_ai_search_tool` | Azure AI Search 문서 검색 |
| 🔍 Search | `bing_web_search_tool` | Bing 웹 검색 |
| 🔍 Search | `github_search_tool` | GitHub 코드/저장소 검색 |
| 🗄️ Database | `azure_sql_query_tool` | Azure SQL 쿼리 실행 |
| 🗄️ Database | `azure_cosmos_db_tool` | Cosmos DB 데이터 관리 |
| 📦 Storage | `azure_blob_storage_tool` | Blob Storage 파일 관리 |
| 🤖 AI | `azure_openai_embedding_tool` | 텍스트 임베딩 생성 |
| 🤖 AI | `azure_computer_vision_tool` | 이미지 분석 |
| 🤖 AI | `azure_translator_tool` | 텍스트 번역 |
| 🤖 AI | `azure_text_analytics_tool` | 텍스트 분석 |
| 🤖 AI | `azure_form_recognizer_tool` | 문서 데이터 추출 |
| 🤖 AI | `azure_speech_to_text_tool` | 음성→텍스트 변환 |
| ⚡ Compute | `azure_function_invoke_tool` | Azure Function 호출 |
| 🔧 Utility | `weather_api_tool` | 날씨 정보 조회 |
| 🔧 Utility | `calculator_tool` | 수학 계산 |

### 📊 검색 성능

```bash
python test_bm25.py
```

```
======================================================================
[SEARCH] Multi-layer Hybrid Search Test
   BM25 -> Sentence-Transformers -> MCP Registry -> GPT-4.1 LLM
   Korean/English multilingual support
======================================================================

[INFO] Model Info:
   - Sentence-Transformers: paraphrase-multilingual-MiniLM-L12-v2
   - MCP Registry API: ENABLED
   - LLM Model: gpt-4.1
   - Registered tools: 15

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

### 🌐 MCP Registry에서 외부 도구 발견

```python
from lib.registry import registry

# 외부 MCP 서버 검색 (API v0.1)
external_tools = registry.discover_external_tools("database", limit=5)
for tool in external_tools:
    server = tool.get("server", {})
    print(f"{server.get('name')}: {server.get('description')}")

# 예시 출력:
# io.github.cybeleri/database-admin: Database admin MCP: schema inspection, query optimization...
```

### 📈 비용 최적화 효과

| 방식 | 100회 검색 비용 | 정확도 |
|-----|----------------|-------|
| LLM만 사용 | ~$3.00 | ⭐⭐⭐⭐⭐ |
| **다층 하이브리드** | **~$0.05** | ⭐⭐⭐⭐⭐ |
| BM25만 사용 | $0 | ⭐⭐⭐ |

> 💡 **~98% 비용 절감** + 동일한 정확도 (Sentence-Transformers 로컬 실행)

### ⚙️ 코드 아키텍처

**registry.py 주요 클래스:**

```python
# MCPRegistryClient - 공식 MCP Registry API 클라이언트 (v0.1)
class MCPRegistryClient:
    BASE_URL = "https://registry.modelcontextprotocol.io"
    API_VERSION = "v0.1"  # 2025년 업데이트된 API 버전
    
    def _build_url(self, path: str) -> str: ...   # URL 헬퍼
    def _get_cached(self, key: str) -> Any: ...   # 캐시 조회
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
    # ... 15개 도구
]

async def initialize_mcp_tools():
    for tool_func, category, tags in TOOL_DEFINITIONS:
        register_tool(tool_func, category=category, tags=tags)
```

---

## English

### 📌 Overview

As the MCP (Model Context Protocol) ecosystem expands, the number of available tools is growing exponentially. However, you can't fit all tool definitions into an LLM's limited context window.

This project implements the **"Tool Search Tool"** pattern with **multi-layer hybrid search** based on Azure OpenAI, dynamically loading only the tools suitable for the current task.

### ✨ Key Features

- 🔍 **4-Layer Hybrid Search**: BM25 → Sentence-Transformers → MCP Registry → GPT-4.1
- 🌍 **Multi-language**: Full Korean/English/50+ languages support
- 🧠 **Sentence-Transformers**: Local multilingual embeddings (free, fast)
- 🌐 **MCP Registry API**: Discover external tools from official registry
- 💰 **Cost Optimized**: ~98% cost reduction with local embeddings
- 🔄 **Dynamic Loading**: Runtime tool injection

### 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/dynamic-mcp-agent.git
cd dynamic-mcp-agent
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Run
python main.py
```

### 🔬 Search Technology Stack

| Layer | Technology | Features |
|-------|-----------|----------|
| 1️⃣ | **BM25** | Keyword matching, free, ultra-fast (~1ms) |
| 2️⃣ | **Sentence-Transformers** | Multilingual semantic search, local, free |
| 3️⃣ | **MCP Registry API** | Official registry, external tool discovery |
| 4️⃣ | **GPT-4.1 LLM** | Reasoning-based final selection & reranking |

**Model Used**: `paraphrase-multilingual-MiniLM-L12-v2`
- 50+ languages supported (including Korean, Japanese, Chinese)
- 384-dimensional embeddings
- Local execution (no API cost)

### 📚 References

- [Implementing Anthropic-style Dynamic Tool Search Tool](https://medium.com/google-cloud/implementing-anthropic-style-dynamic-tool-search-tool-f39d02a35139)
- [MCP Registry](https://registry.modelcontextprotocol.io)
- [MCP Registry API Documentation](https://registry.modelcontextprotocol.io/docs)
- [Sentence-Transformers](https://www.sbert.net/)
- [ToolGen: Unified Tool Retrieval and Calling (ICLR 2025)](https://github.com/Reason-Wang/ToolGen)
- [Azure OpenAI Function Calling](https://learn.microsoft.com/azure/ai-services/openai/how-to/function-calling)
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

Made with ❤️ using Azure OpenAI + Sentence-Transformers + MCP Registry
