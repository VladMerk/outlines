import tiktoken
from langchain.prompts import ChatPromptTemplate

from llms import llm
from models import SubSection
from states import ContentGenerationState
from loggers import create_logger, SafeLogger

writing_logger: SafeLogger = create_logger("writing_phase", "WRITING_PHASE")


async def writing_phase(state: ContentGenerationState):
    try:
        writing_logger.log_function_start("writing_phase")

        writing_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Вы - {role}.
                    Ваша задача - написать раздел статьи, посвященную теме и подразделу предоставленную пользователем.
                    Редактором был составлен план для написания статьи, поэтому четко следуйте инструкциям:
                    - Четко следуйте этому плану, учитывайте описание для подтем. Они согласованы с пользователем.
                    - Объясните понятно, последовательно, с примерами и пояснениями.
                    - Учитывайте предыдущий контекст, если он добавлен, в нем предыдущие написанные подраздели,
                    чтобы избежать повторов и сделать плавные переходы между темами.
                    - Используйте подготовленные исследовательские данные.
                    - Используйте Markdown для форматирования текста
                    - Испльзуйте Mermaid для Markdown для построения схем
                    - Не нужно добавлять "Введение" и "Заключение" к подсекции - нужны только ответы на описываемые темы
                    для секций статьи.

                    ВАЖНО ПРО СТРУКТУРУ ПРОЕКТА:
                    - Если это проектная статья, ОБЯЗАТЕЛЬНО указывайте названия файлов и структуру проекта
                    - Показывайте, в какой файл помещать код: src/main.rs, app/models.py, etc.
                    - Создавайте четкую структуру папок и файлов
                    - Объясняйте, как организовать код в проекте

                    - Обязательно нужно добавить секцию с рекомендациями для чтения/просмотра с различными полезными рессурсами,
                    которые могут помочь расширить знания только по указанному разделу статьи. Учитывайте рекомендации из предыдущего
                    контекста, чтобы избежать повторов. Если к разделу нет хороших рекомендаций какого то типа,
                    тогда оставить рекомендации пустыми.
                        - книги: автор, название, дополнительная информация о книги для упрощения поиска:
                        isbn номер, ссылка в интернет-магазине и т.д. Но если вы не уверены в существовании такой книги,
                        тогда не пишите ничего. Оставьте поле пустым.
                        - сслылки на рессурсы в интернете
                        - документация
                        - качественные запросы в поисковые системы по теме
                        - и другое, что посчитаете нужным.

                    Цель: сделать сложную тему понятной и практичной.
                    """,
                ),
                (
                    "user",
                    """
                        Тема статьи: {topic}
                        Подтема: {title}
                        Описание: {description}
                        Предыдущий контекст: {context}
                        План раздела: {plan}
                        Исследовательские данные: {research_data}

                        Напишите полный текст для этого раздела статьи.
                    """,
                ),
            ]
        )

        topic = state["topic"]
        plans = state["plans"]

        research_results = state["research_results"]
        role = state["writer_role"]
        final_sections: list[str] = []
        llm.temperature = 0.3

        writing_llm = writing_prompt | llm.with_structured_output(SubSection)

        writing_logger.local_state["custom_data"]["research_dates"] = list()

        for i, plan in enumerate(plans):
            research_data = research_results[i]["research_data"]
            writing_logger.local_state["custom_data"]["research_dates"].append(writing_logger.count_tokens(research_data))
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")

            if i == 0:
                context = ""
            else:
                # TODO
                N = min(20, len(final_sections))
                context = "\n".join([str(item) for item in final_sections[-N:]])
                len_context = writing_logger.count_tokens(context)
                writing_logger.log_info(f"Номер плана: {i}\nКоличество секций: {N}\nКоличество токенов: {len_context}")
                while len_context > 2000 and N > 1:
                    N -= 1
                    context = "\n".join([str(item) for item in final_sections[-N:]])

            plan_content = plan["plan"]
            if len(encoding.encode(plan_content)) > 1000:
                tokens = encoding.encode(plan_content)
                truncated_tokens = tokens[:500]
                plan_content = encoding.decode(truncated_tokens)
                plan_content += "\n\n[ПЛАН ОБРЕЗАН]"

            with writing_logger.safe_llm_call(
                "writing_phase",
                model="gpt-4o-mini",
                context=context,
                plan_content=plan_content,
                description=research_results[i]["description"],
                research_data=research_data,
            ):
                result = SubSection.model_validate(
                    await writing_llm.ainvoke(
                        {
                            "topic": topic,
                            "title": plan["section_title"],
                            "description": research_results[i]["description"],
                            "context": context,
                            "plan": plan_content,  # Используем обрезанный план
                            "role": role,
                            "research_data": research_data,  # Используем обрезанные данные
                        }
                    )
                )

                writing_logger.log_llm_response("writing_phase", str(result))

            final_sections.append(str(result))

        writing_logger.log_function_end("writing_phase")

        return {**state, "sections": final_sections}
    except Exception as e:
        writing_logger.log_error("writing_phase", e)
        raise
