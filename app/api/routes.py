import uuid

from fastapi import APIRouter
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.api.schemas import StartResponse, UserReply
from app.simple_graph.graph import build_graph

router = APIRouter()
graph = build_graph()


@router.get("/chat/start")
async def start_chat():
    thread_id = str(uuid.uuid4())
    config = RunnableConfig(configurable={"thread_id": thread_id})

    async for chunk in graph.astream(
        {"thread_id": thread_id}, config=config, stream_mode="updates"
    ):
        if "__interrupt__" in chunk:
            message = chunk["__interrupt__"][0].value
            break

    return StartResponse(thread_id=thread_id, ai_message=message)


@router.post("/chat/continue")
async def continue_chat(data: UserReply):
    config = RunnableConfig(configurable={"thread_id": data.thread_id})

    state = await graph.ainvoke(Command(resume=data.user_input), config=config)

    return {"message": state["answer"]}
