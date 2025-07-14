from langgraph.graph import END, START, StateGraph
# from langgraph.prebuilt import ToolNode

from nodes.planning.practical_planning import practical_planning_phase
from nodes.planning.role_selector import role_selector_phase
from nodes.research.research_phase import research_phase
from nodes.research.vector_store import vector_store_node
from nodes.writing.base import writing_phase
from states import ContentGenerationState
# from utils.tools import tools


def create_content_graph():
    graph_builder = StateGraph(ContentGenerationState)

    # tool_node = ToolNode(tools=tools)

    # graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("research_phase", research_phase)
    graph_builder.add_node("vector_store_node", vector_store_node)
    graph_builder.add_node("practical_planning_phase", practical_planning_phase)  # Используем практическое планирование
    graph_builder.add_node("role_selector_phase", role_selector_phase)
    graph_builder.add_node("writing_phase", writing_phase)

    graph_builder.add_edge(START, "research_phase")
    graph_builder.add_edge("research_phase", "vector_store_node")
    graph_builder.add_edge("vector_store_node", "practical_planning_phase")
    graph_builder.add_edge("practical_planning_phase", "role_selector_phase")
    graph_builder.add_edge("role_selector_phase", "writing_phase")
    graph_builder.add_edge("writing_phase", END)

    return graph_builder.compile()


graph = create_content_graph()


if __name__ == "__main__":
    import asyncio

    async def test_optimized():
        from models import Section

        sections = [
            Section(section_title="Основы ownership", content="Понятие владения в Rust"),
            Section(section_title="Borrowing", content="Заимствования и ссылки"),
            Section(section_title="Lifetimes", content="Время жизни переменных"),
        ]

        state = {
            "topic": "Система владения в Rust",
            "wishes": "Практические примеры с кодом",
            "sections": sections,
            "messages": [],
        }

        print("=== ТЕСТ ОПТИМИЗИРОВАННОГО ИССЛЕДОВАНИЯ ===")
        result = await research_phase(state)

        print(f"Секций обработано: {len(result['research_results'])}")
        for research in result["research_results"]:
            print(f"\nСекция: {research['section_title']}")
            print(f"Данные: {research['research_data'][:200]}...")

    asyncio.run(test_optimized())
