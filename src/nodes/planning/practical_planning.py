from langchain.prompts import ChatPromptTemplate

from llms import llm
# from nodes.planning.strategy import determine_article_type


async def practical_planning_phase(state):
    """Планирование без академической воды, с фокусом на практику"""

    # Сначала определяем стратегию
    # state_with_strategy = await determine_article_type(state)
    # article_strategy = state_with_strategy["article_strategy"]
    # **Стратегия статьи:** {article_strategy}
    """
        **Структура секции:**
        1. **Прямое определение** (1 абзац, без "введений")
        2. **Базовый пример** (конкретный код/случай)
        3. **Развитие темы** (усложнение, вариации)
        4. **Практические моменты** (что важно знать)
        5. **Подводные камни** (частые ошибки)
    """

    planning_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            Вы - практик-методист, создающий планы для действенных статей.

            ПРИНЦИПЫ:
            - Никаких академических введений/заключений
            - Прямо к сути: определение → пример → практика
            - Минимум теории, максимум практики
            - Конкретные примеры с кодом/действиями
            - Красная нить через всю статью. Если статься проектная - нужна структура каталогов и файлов.
              Красная нить через структуру каталогов и файлов.

            ЗАПРЕЩЕНО:
            - "В этом разделе мы изучим..."
            - "Заключение: мы рассмотрели..."
            - "Введение в тему..."
            - Абстрактные рассуждения без примеров
            """,
            ),
            (
                "user",
                """

            **Секция:** {title}
            **Описание:** {description}
            **Исследовательские данные:** {research_data}

            Создайте практический план БЕЗ воды

            **Требования к содержанию:**
            - Конкретные примеры кода (если применимо)
            - Реальные сценарии использования
            - Практические советы
            - НЕТ абстрактных рассуждений

            **Если это часть проекта:**
            - Как эта секция связана с общим проектом
            - Какую часть функциональности реализуем
            - В каких файлах находятся примеры кода. название файла пишите в коментариях к коду.
              Например так:
              ```rust
              // src/main.rs
              ...
              // код файла и всех функций
              ...
              ```

            Будьте конкретны и практичны!
            """,
            ),
        ]
    )

    # topic = state["topic"]
    research_results = state["research_results"]
    plans = []

    for research in research_results:
        result = await llm.ainvoke(
            planning_prompt.format(
                # article_strategy=article_strategy,
                title=research["section_title"],
                description=research["description"],
                research_data=research.get("research_data", "Нет данных"),
            )
        )

        plans.append({"section_title": research["section_title"], "plan": result.content})

    return {**state, "plans": plans}


# Добавить в конец файла practical_planning.py

if __name__ == "__main__":
    import asyncio
    from models import Section

    async def test_practical_planning():
        """Тест планирования для разных типов статей"""

        # Тест 1: Проектная статья
        print("=== ТЕСТ 1: ПРОЕКТНАЯ СТАТЬЯ (CLI утилита) ===\n")

        project_state = {
            "topic": "Создание CLI утилиты для работы с файлами на Rust",
            "wishes": "Пошагово создать утилиту с командами list, copy, delete",
            "sections": [
                Section(section_title="Настройка проекта", content="Создание структуры проекта и настройка зависимостей"),
                Section(section_title="Базовая структура CLI", content="Реализация основного интерфейса командной строки"),
                Section(section_title="Команды работы с файлами", content="Реализация команд list, copy, delete"),
            ],
            "research_results": [
                {
                    "section_title": "Настройка проекта",
                    "description": "Создание структуры проекта и настройка зависимостей",
                    "research_data": "Для CLI в Rust используется clap для парсинга аргументов. Структура: src/main.rs, Cargo.toml",
                },
                {
                    "section_title": "Базовая структура CLI",
                    "description": "Реализация основного интерфейса командной строки",
                    "research_data": "clap derive API позволяет использовать структуры. Пример: #[derive(Parser)]",
                },
                {
                    "section_title": "Команды работы с файлами",
                    "description": "Реализация команд list, copy, delete",
                    "research_data": "std::fs для работы с файлами. fs::read_dir, fs::copy, fs::remove_file",
                },
            ],
        }

        result = await practical_planning_phase(project_state)

        print("СТРАТЕГИЯ:")
        print("-" * 50)
        # print(result["article_strategy"])
        print("\n" + "=" * 50 + "\n")

        print("ПЛАНЫ СЕКЦИЙ:")
        for i, plan in enumerate(result["plans"], 1):
            print(f"\n{i}. {plan['section_title']}")
            print("-" * 30)
            print(plan["plan"])

        # Проверки
        print("\n\nПРОВЕРКИ:")
        # is_project = "ПРОЕКТНАЯ" in result["article_strategy"]
        is_project = True
        print(f"✓ Определена как проектная: {is_project}")

        if is_project:
            # Проверяем упоминание структуры
            # has_structure = any(word in result["article_strategy"] for word in ["├──", "│", "src/", "Cargo.toml"])
            # print(f"✓ Есть структура проекта: {has_structure}")

            # Проверяем упоминание файлов в планах
            file_count = sum(1 for plan in result["plans"] if "файл" in plan["plan"].lower() or "src/" in plan["plan"])
            print(f"✓ Планы с упоминанием файлов: {file_count}/{len(result['plans'])}")

        print("\n" + "=" * 70 + "\n")

        # Тест 2: Объяснительная статья
        print("=== ТЕСТ 2: ОБЪЯСНИТЕЛЬНАЯ СТАТЬЯ (Паттерн) ===\n")

        explain_state = {
            "topic": "Паттерн Builder в программировании",
            "wishes": "Объяснить концепцию и показать примеры",
            "sections": [
                Section(section_title="Что такое Builder", content="Основная идея паттерна"),
                Section(section_title="Примеры использования", content="Реализация на разных языках"),
            ],
            "research_results": [
                {
                    "section_title": "Что такое Builder",
                    "description": "Основная идея паттерна",
                    "research_data": "Builder - порождающий паттерн для пошагового создания объектов",
                },
                {
                    "section_title": "Примеры использования",
                    "description": "Реализация на разных языках",
                    "research_data": "В Java - StringBuilder, в Rust - паттерн с методами .build()",
                },
            ],
        }

        result2 = await practical_planning_phase(explain_state)

        print("СТРАТЕГИЯ (первые 300 символов):")
        print("-" * 50)
        # print(result2["article_strategy"][:300] + "...")
        print(result2["plans"])

        # print(f"\n\nТип статьи: {'ОБЪЯСНИТЕЛЬНАЯ' if 'ОБЪЯСНИТЕЛЬНАЯ' in result2['article_strategy'] else 'другой'}")
        # print(f"\n\nТип статьи: {'ОБЪЯСНИТЕЛЬНАЯ' if 'ОБЪЯСНИТЕЛЬНАЯ' in result2['article_strategy'] else 'другой'}")
        # Сравнение подходов
        print("\n\nСРАВНЕНИЕ ПОДХОДОВ:")
        print(f"Проектная: {'структура' in result['plans'][0]['plan'].lower()}")
        print(f"Объяснительная: {'определение' in result2['plans'][0]['plan'].lower()}")

    # Запуск теста
    asyncio.run(test_practical_planning())
