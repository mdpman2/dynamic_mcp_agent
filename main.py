# -*- coding: utf-8 -*-
"""
Dynamic MCP Agent - Main Application (v1 Responses API)

Azure OpenAI v1 Responses API 기반의 동적 도구 검색 및 로딩 에이전트를 실행합니다.

v2.0.0 업데이트 (2026-02-07):
- [NEW] --stream 모드: Responses API 스트리밍 응답 지원
- [NEW] mcp-add 명령: 런타임에 원격 MCP 서버 추가
- [CHANGED] UI 텍스트에 v1 Responses API 정보 반영
- [CHANGED] stats 표시에 api, remote_mcp_servers, last_response_id 추가
- [CHANGED] Gradio 웹 UI에 2026 최신 기술 설명 반영
- [CHANGED] 데모 시나리오에 Microsoft Learn MCP 서버 검색 시나리오 추가

사용법:
    python main.py              # CLI 모드로 실행
    python main.py --web        # Gradio 웹 인터페이스로 실행
    python main.py --demo       # 데모 시나리오 실행
    python main.py --stream     # 스트리밍 CLI 모드로 실행
"""

import os
import sys
import asyncio
import argparse
import logging
from dotenv import load_dotenv

# 모듈 경로 추가 (상위 디렉토리에서도 실행 가능하도록)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def check_environment():
    """환경 변수가 올바르게 설정되었는지 확인합니다."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print("=" * 60)
        print("⚠️  환경 변수가 설정되지 않았습니다!")
        print("=" * 60)
        print("\n다음 환경 변수를 .env 파일에 설정해 주세요:\n")
        for var in missing:
            print(f"  - {var}")
        print("\n필수 환경 변수:")
        print("  AZURE_OPENAI_ENDPOINT       - Azure OpenAI 엔드포인트")
        print("  AZURE_OPENAI_API_KEY        - API 키")
        print("  AZURE_OPENAI_DEPLOYMENT_NAME - 모델 배포명 (예: gpt-5, gpt-5-mini)")
        print("\n선택 환경 변수:")
        print("  AZURE_OPENAI_API_VERSION    - v1 API 버전 (preview/latest, 기본: preview)")
        print("\n.env.example 파일을 참고하여 .env 파일을 생성하세요.")
        print("=" * 60)
        return False
    
    return True


def run_cli_mode():
    """CLI 모드로 에이전트를 실행합니다."""
    from dynamic_mcp_agent import create_agent, registry
    
    print("\n" + "=" * 60)
    print("🤖 Dynamic MCP Agent - Azure OpenAI v1 Responses API")
    print("=" * 60)
    print("하이브리드 검색 + 네이티브 MCP 서버 통합 AI 에이전트입니다.")
    print("검색 전략: BM25 → Embedding → LLM (비용 최적화)")
    print("API: v1 Responses API | 대화 체이닝: previous_response_id")
    print("-" * 60)
    print("'quit' 또는 'exit'를 입력하면 종료됩니다.")
    print("'stats'를 입력하면 에이전트 통계를 확인할 수 있습니다.")
    print("'search-stats'를 입력하면 검색 통계를 확인할 수 있습니다.")
    print("'tools'를 입력하면 활성화된 도구 목록을 확인할 수 있습니다.")
    print("'mcp-add <url> <label>'로 원격 MCP 서버를 추가할 수 있습니다.")
    print("'reset'을 입력하면 대화를 초기화합니다.")
    print("=" * 60 + "\n")
    
    # 에이전트 생성
    agent = create_agent()
    
    print(f"📦 레지스트리에 {registry.count()}개의 도구가 등록되었습니다.")
    if agent.remote_mcp_servers:
        print(f"🌐 {len(agent.remote_mcp_servers)}개의 원격 MCP 서버가 연결되었습니다.")
    print()
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("\n👋 안녕히 가세요!")
                break
            
            if user_input.lower() == 'stats':
                stats = agent.get_stats()
                print("\n📊 에이전트 통계:")
                print(f"   - 모델: {stats['model']}")
                print(f"   - API: {stats['api']}")
                print(f"   - 전체 도구 수: {stats['total_tools_in_registry']}")
                print(f"   - 활성 도구 수: {stats['active_tools']}")
                print(f"   - 원격 MCP 서버: {stats['remote_mcp_servers']}개")
                print(f"   - 대화 턴 수: {stats['conversation_turns']}")
                print(f"   - 마지막 response_id: {stats['last_response_id'] or 'None'}")
                if stats['active_tool_names']:
                    print(f"   - 활성 도구: {', '.join(stats['active_tool_names'])}")
                print()
                continue
            
            if user_input.lower() == 'search-stats':
                search_stats = registry.get_search_stats()
                print("\n🔍 하이브리드 검색 통계:")
                print(f"   - 총 검색 수: {search_stats['total_searches']}")
                print(f"   - BM25 히트: {search_stats['bm25_hits']} ({search_stats.get('bm25_ratio', '0%')})")
                print(f"   - Embedding 히트: {search_stats['embedding_hits']} ({search_stats.get('embedding_ratio', '0%')})")
                print(f"   - LLM 히트: {search_stats['llm_hits']} ({search_stats.get('llm_ratio', '0%')})")
                print()
                continue
            
            if user_input.lower() == 'tools':
                tools = agent.get_active_tools_list()
                if tools:
                    print(f"\n🔧 활성화된 도구 ({len(tools)}개):")
                    for tool in tools:
                        print(f"   - {tool}")
                else:
                    print("\n🔧 활성화된 도구가 없습니다.")
                print()
                continue
            
            if user_input.lower().startswith('mcp-add '):
                parts = user_input.split(maxsplit=2)
                if len(parts) >= 3:
                    url, label = parts[1], parts[2]
                    agent.add_remote_mcp_server(
                        server_url=url,
                        server_label=label
                    )
                    print(f"\n🌐 MCP 서버 추가됨: {label} ({url})\n")
                else:
                    print("\n⚠️ 사용법: mcp-add <server_url> <server_label>\n")
                continue
            
            if user_input.lower() == 'reset':
                agent.reset_conversation()
                agent.reset_tools()
                print("\n🔄 대화와 도구가 초기화되었습니다.\n")
                continue
            
            # 에이전트 응답 생성
            print("\n🤖 Agent: ", end="", flush=True)
            response = agent.chat_sync(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 안녕히 가세요!")
            break
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"\n⚠️ 오류가 발생했습니다: {e}\n")


def run_web_mode():
    """Gradio 웹 인터페이스로 에이전트를 실행합니다."""
    try:
        import gradio as gr
    except ImportError:
        print("⚠️ Gradio가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요: pip install gradio")
        return
    
    from dynamic_mcp_agent import create_agent, registry
    
    # 에이전트 생성
    agent = create_agent()
    
    def chat_fn(message, history):
        """Gradio 채팅 함수"""
        response = agent.chat_sync(message)
        return response
    
    def reset_fn():
        """대화 초기화 함수"""
        agent.reset_conversation()
        agent.reset_tools()
        return None, "대화와 도구가 초기화되었습니다."
    
    def get_stats_fn():
        """통계 조회 함수"""
        stats = agent.get_stats()
        return f"""
📊 **에이전트 통계**
- 모델: {stats['model']}
- API: {stats['api']}
- 전체 도구 수: {stats['total_tools_in_registry']}
- 활성 도구 수: {stats['active_tools']}
- 원격 MCP 서버: {stats['remote_mcp_servers']}개
- 활성 도구: {', '.join(stats['active_tool_names']) if stats['active_tool_names'] else '없음'}
- 대화 턴 수: {stats['conversation_turns']}
- Response ID: {stats['last_response_id'] or 'None'}
"""
    
    # Gradio 인터페이스 구성
    with gr.Blocks(title="Dynamic MCP Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Dynamic MCP Agent - Azure OpenAI v1 Responses API
        
        동적 도구 검색 + 네이티브 MCP 서버 통합 AI 에이전트입니다.
        
        **2026 최신 기술:**
        - 🚀 v1 Responses API - 상태 기반 대화 체이닝 (previous_response_id)
        - 🌐 네이티브 원격 MCP 서버 도구 통합
        - 🧠 GPT-5 시리즈 모델 지원
        - ⚡ 스트리밍 응답 지원
        
        **사용 방법:**
        1. 자연어로 질문하면 에이전트가 필요한 도구를 자동으로 검색하고 로드합니다.
        2. 예: "번역해 줘", "이미지를 분석해 줘", "Microsoft 문서에서 검색해 줘"
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=chat_fn,
                    title="",
                    retry_btn=None,
                    undo_btn=None,
                )
            
            with gr.Column(scale=1):
                stats_output = gr.Markdown(get_stats_fn())
                refresh_btn = gr.Button("🔄 통계 새로고침")
                reset_btn = gr.Button("🗑️ 대화 초기화")
                status_output = gr.Textbox(label="상태", interactive=False)
        
        refresh_btn.click(fn=get_stats_fn, outputs=stats_output)
        reset_btn.click(fn=reset_fn, outputs=[chatbot, status_output])
    
    print("\n🌐 웹 인터페이스를 시작합니다...")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


def run_demo_mode():
    """데모 시나리오를 실행합니다."""
    from dynamic_mcp_agent import create_agent, registry
    
    print("\n" + "=" * 60)
    print("🎬 Dynamic MCP Agent - 데모 시나리오 (v1 Responses API)")
    print("=" * 60 + "\n")
    
    # 에이전트 생성
    agent = create_agent()
    
    print(f"📦 레지스트리에 {registry.count()}개의 도구가 등록되었습니다.\n")
    
    # 데모 시나리오
    demo_queries = [
        "Azure에서 문서 검색 도구가 있나요?",
        "azure_ai_search_tool을 로드해 주세요.",
        "텍스트를 영어로 번역하고 싶어요. 어떤 도구를 사용할 수 있나요?",
        "Microsoft Learn에서 Azure Functions 정보를 검색해 주세요.",
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*40}")
        print(f"📌 시나리오 {i}/{len(demo_queries)}")
        print(f"{'='*40}")
        print(f"\n👤 You: {query}")
        
        response = agent.chat_sync(query)
        print(f"\n🤖 Agent: {response}")
        
        # 현재 활성 도구 표시
        tools = agent.get_active_tools_list()
        if tools:
            print(f"\n🔧 현재 활성 도구: {', '.join(tools)}")
        
        print("\n" + "-" * 40)
        input("(Enter를 눌러 다음 시나리오로 계속...)")
    
    print("\n✅ 데모가 완료되었습니다!")
    
    # 최종 통계
    stats = agent.get_stats()
    print(f"\n📊 최종 통계:")
    print(f"   - 활성 도구 수: {stats['active_tools']}")
    print(f"   - 활성 도구: {', '.join(stats['active_tool_names'])}")


def run_stream_cli_mode():
    """스트리밍 CLI 모드로 에이전트를 실행합니다. (Responses API 스트리밍)"""
    from dynamic_mcp_agent import create_agent, registry
    
    print("\n" + "=" * 60)
    print("🤖 Dynamic MCP Agent - 스트리밍 모드 (v1 Responses API)")
    print("=" * 60)
    print("스트리밍 응답을 실시간으로 출력합니다.")
    print("'quit' 또는 'exit'를 입력하면 종료됩니다.")
    print("=" * 60 + "\n")
    
    agent = create_agent(enable_streaming=True)
    
    print(f"📦 레지스트리에 {registry.count()}개의 도구가 등록되었습니다.\n")
    
    async def _stream_response(msg: str):
        """스트리밍 응답 출력 (루프 바깥에 정의하여 매 턴 클로저 재생성 방지)"""
        async for chunk in agent.chat_stream(msg):
            print(chunk, end="", flush=True)
        print()
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("\n👋 안녕히 가세요!")
                break
            
            if user_input.lower() == 'reset':
                agent.reset_conversation()
                agent.reset_tools()
                print("\n🔄 대화와 도구가 초기화되었습니다.\n")
                continue
            
            # 스트리밍 응답
            print("\n🤖 Agent: ", end="", flush=True)
            asyncio.run(_stream_response(user_input))
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 안녕히 가세요!")
            break
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"\n⚠️ 오류가 발생했습니다: {e}\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Dynamic MCP Agent - Azure OpenAI v1 Responses API 기반 동적 도구 검색 에이전트"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Gradio 웹 인터페이스로 실행"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="데모 시나리오 실행"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="스트리밍 CLI 모드로 실행"
    )
    
    args = parser.parse_args()
    
    # 환경 변수 확인
    if not check_environment():
        sys.exit(1)
    
    if args.web:
        run_web_mode()
    elif args.demo:
        run_demo_mode()
    elif args.stream:
        run_stream_cli_mode()
    else:
        run_cli_mode()


if __name__ == "__main__":
    main()
