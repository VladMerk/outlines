from langchain.prompts import ChatPromptTemplate

from llms import llm
from states import ContentGenerationState


async def role_selector_phase(state: ContentGenerationState):
    topic = state["topic"]
    wishes = state["wishes"]

    prompt = ChatPromptTemplate.from_template(
        """
        На основе темы и пожеланий пользователя (если они есть) определите, кто должен быть автором текста.

        Формат ответа:
        Должность

        Пример:
            Тема: Алгоритм быстрой сортировки
            Пожелания: подробные объснения как работает алогоритм и примеры кода на Python.

            Ответ: Python-программист, преподаватель университета.
            ---

            Тема: Развитие сюрреализма в цифровом искусстве
            Пожелания: рассказать об истории и влиянии

            Ответ: Историк современного искусства.
            ---

            Тема: История США конца XVIII века
            Пожелания: развитие сельского хозяйства в США в этот период

            Ответ: Историк, преподаватель истории США

        Теперь, пожалуйста, сформулируйте должность для следующей темы:
        - Тема: {topic}
        - Пожелания: {wishes}

        нужно вернуть только должность, без дополнений и посторонних слов.
        """
    )

    result = await llm.ainvoke(prompt.format(topic=topic, wishes=wishes))

    return {**state, "writer_role": result.content}
