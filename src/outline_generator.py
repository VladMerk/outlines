from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain as as_runnable

from llms import llm
from models import Outline
from states import ArticleState

outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы – эксперт в информационных технологиях, языках программирования и преподавании иностранных языков."
            "\n\n### 📝 Ваша задача:"
            " Написать **полную структурированную статью**, строго следуя заголовкам подтем, которые вам переданы."
            "\n\n### 🔍 Основные требования:"
            "- **Статья должна четко следовать структуре подтем** (заголовки обязательны!)."
            "- **Не пишите лишних разделов, не добавляйте вводных фраз**, если они не запрашивались."
            "- **Формат Markdown:** четкая структура, списки, заголовки, примеры кода (если уместно)."
            "- **Если тема связана с программированием, используйте примеры кода с пояснениями.**"
            "\n\n### 🎯 Как составить статью:"
            "1. **Каждая подтема – это отдельный раздел**."
            "2. **Заголовки подтем обязательны** (они уже согласованы)."
            "3. **Не добавляйте новые секции**."
            "\n\nТеперь напишите **полную статью по данной теме**, следуя предоставленным подтемам.",
        ),
        (
            "user",
            "📝 **Тема статьи:** {topic}\n\n📌 **Список подтем (разделов статьи):**\n{sections}",
        ),
    ]
)


@as_runnable
async def get_outline(state: ArticleState):
    outline_chain = outline_prompt | llm.with_structured_output(Outline)
    outline = Outline.model_validate(
        await outline_chain.ainvoke({"topic": state["topic"], "sections": state["sections"]})
        )  # type: ignore

    return {**state, "sections": outline.sections}
