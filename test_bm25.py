#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic MCP Agent - Multi-layer Hybrid Search Test

Search Layers:
1. BM25 (keyword matching) - fast and free
2. Sentence-Transformers (local multilingual embedding) - semantic
3. MCP Registry API (external) - discover new tools
4. GPT-4.1 LLM (reasoning) - complex queries
"""
import sys
import asyncio
from pathlib import Path

# 모듈 경로 설정
sys.path.insert(0, str(Path(__file__).parent))

from lib.tools import initialize_mcp_tools
from lib.registry import registry


def print_separator(title: str = "", char: str = "=", width: int = 70) -> None:
    """구분선 출력 헬퍼"""
    if title:
        print(f"\n{char * width}")
        print(title)
        print(char * width)
    else:
        print(char * width)


async def test_hybrid_search() -> None:
    """다층 하이브리드 검색 테스트"""
    await initialize_mcp_tools()
    
    print_separator("[SEARCH] Multi-layer Hybrid Search Test\n   BM25 -> Sentence-Transformers -> MCP Registry -> GPT-4.1 LLM\n   Korean/English multilingual support")
    
    # 모델 정보 출력
    info = registry.get_model_info()
    print(f"\n[INFO] Model Info:")
    print(f"   - Sentence-Transformers: {info.get('sentence_model') or 'DISABLED'}")
    print(f"   - MCP Registry API: {'ENABLED' if info.get('mcp_registry_enabled') else 'DISABLED'}")
    print(f"   - LLM Model: {info.get('llm_model') or 'DISABLED'}")
    print(f"   - Registered tools: {info.get('tool_count', 0)}")

    # 한국어 테스트 쿼리
    queries = [
        "문서에서 텍스트 추출하고 싶어",
        "영어를 한국어로 번역해줘",
        "사진 분석 기능이 필요해",
        "데이터베이스에서 데이터 조회",
        "음성을 텍스트로 바꾸고 싶어",
        "웹에서 정보 검색",
        "파일 저장하고 관리"
    ]

    print("\n" + "-" * 70)
    print("[KO] Korean Query Test")
    print("-" * 70)
    
    _run_queries(queries, "Query")

    # 영어 테스트 쿼리
    english_queries = [
        "translate text to Korean",
        "analyze image content",
        "search documents with semantic similarity"
    ]
    
    print("\n" + "-" * 70)
    print("[EN] English Query Test")
    print("-" * 70)
    
    _run_queries(english_queries, "Query")

    # 통계 출력
    _print_stats()


def _run_queries(queries: list, label: str = "Query") -> None:
    """쿼리 실행 및 결과 출력 헬퍼"""
    for q in queries:
        print(f"\n[Q] {label}: '{q}'")
        for i, r in enumerate(registry.search(q, top_k=3), 1):
            parts = r.split(": ", 1)
            name, desc = parts[0], (parts[1][:40] + "..." if len(parts) > 1 else "")
            print(f"   {i}. {name}: {desc}")


def _print_stats() -> None:
    """검색 통계 출력 헬퍼"""
    print_separator("[STATS] Search Statistics")
    stats = registry.get_search_stats()
    print(f"   BM25 hits: {stats['bm25_hits']} ({stats.get('bm25_ratio', '0%')})")
    print(f"   Sentence-Transformers hits: {stats['sentence_hits']} ({stats.get('sentence_ratio', '0%')})")
    print(f"   MCP Registry hits: {stats['mcp_registry_hits']} ({stats.get('mcp_registry_ratio', '0%')})")
    print(f"   LLM hits: {stats['llm_hits']} ({stats.get('llm_ratio', '0%')})")
    print(f"   Total searches: {stats['total_searches']}")


async def test_mcp_registry() -> None:
    """MCP Registry API 테스트"""
    print_separator("[MCP] MCP Registry API Test\n   External MCP Server Discovery")
    
    for q in ["database", "translation", "image analysis"]:
        print(f"\n[Q] MCP Registry Search: '{q}'")
        external = registry.discover_external_tools(q, limit=3)
        
        if external:
            for i, tool in enumerate(external, 1):
                print(f"   {i}. {tool.get('name', 'unknown')}: {tool.get('description', '')[:60]}...")
        else:
            print("   (No results or API disabled)")


async def test_search_strategies() -> None:
    """검색 전략 비교 테스트"""
    await initialize_mcp_tools()
    
    print_separator("[COMPARE] Search Strategy Comparison Test")
    
    query = "텍스트를 다른 언어로 변환"
    
    for strategy in ["bm25", "sentence", "hybrid"]:
        print(f"\n[Q] Strategy: {strategy.upper()} | Query: '{query}'")
        for i, r in enumerate(registry.search(query, top_k=3, strategy=strategy), 1):
            print(f"   {i}. {r.split(': ', 1)[0]}")


if __name__ == "__main__":
    print("\n[START] Dynamic MCP Agent - 다층 하이브리드 검색 테스트 시작\n")
    asyncio.run(test_hybrid_search())
    asyncio.run(test_mcp_registry())
    asyncio.run(test_search_strategies())
    
    print("\n[DONE] 모든 테스트 완료!")

