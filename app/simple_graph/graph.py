from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, add_messages
from langgraph.types import interrupt
from typing_extensions import Annotated, TypedDict


class SimpleState(TypedDict):
    thread_id: str
    messages: Annotated[list[str], add_messages]
    answer: str


async def start_node(state: SimpleState):
    user_answer = interrupt("Как ваше имя?")
    return {**state, "messages": user_answer}


async def next_node(state: SimpleState):
    answer: str = str(state["messages"][-1].content)  # type: ignore
    return {**state, "answer": f"Привет! {answer}"}


def build_graph():
    builder = StateGraph(SimpleState)
    builder.add_node("start_node", start_node)
    builder.add_node("next_node", next_node)

    builder.set_entry_point("start_node")
    builder.add_edge("start_node", "next_node")
    builder.set_finish_point("next_node")

    return builder.compile(checkpointer=MemorySaver())
