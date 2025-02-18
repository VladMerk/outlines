from langgraph.types import Command

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}
state = {"some_text": "Original text"}
while True:
    for chunk in graph.stream(state, config=thread_config):
        print(chunk)
        if "__interrupt__" in chunk:
            # Получаем информацию от пользователя (например, через интерфейс)
            user_input = input("Please provide input: ")
            state = Command(resume=user_input)
            break
    else:
        # Граф завершил выполнение без остановок
        break
