import logging
import os
import sys
import json
import tiktoken
from datetime import datetime
from pathlib import Path
from typing import Any


class SafeLogger:
    """Простой логгер для вывода в консоль и сохранения в файл"""

    def __init__(self, name: str = "logger", phase_name: str = "Unknown", model_name: str = "gpt-4o-mini") -> None:
        self.name = name
        self.phase_name = phase_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.local_state = {
            "phase_name": self.phase_name,
            "current_function": None,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "warnings_count": 0,  # ДОБАВЛЕНО: инициализация счетчика предупреждений
            "session_start": datetime.now().isoformat(),
            "custom_data": {},  # Для пользовательских данных из функции.
        }

        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        self.log_file = self.log_dir / f"{name}_{self.session_id}.log"
        self.tokens_file = self.log_dir / f"tokens_{name}_{self.session_id}.json"

        self.setup_logging()

        self.logger.info(f"🚀 Логгер '{name}' создан для фазы '{phase_name}'")
        self.save_tokens_stats()

    def setup_logging(self) -> None:
        """Инициализация и настройка логгера"""

        # Создание логгера
        self.logger = logging.getLogger(f"{self.name}_{self.session_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # Форматтеры - ИСПРАВЛЕНО: добавлены недостающие скобки
        file_formatter = logging.Formatter(
            f"[%(asctime)s][%(levelname)s][{self.phase_name}] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_formatter = logging.Formatter(f"[%(levelname)s][{self.phase_name}] - %(message)s")

        # Файловый хэндлер с немедленной записью
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Отключаем буферизацию
        file_handler.stream.reconfigure(line_buffering=True)
        self.logger.addHandler(file_handler)

        # Консольный хэндлер
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def count_tokens(self, text: Any) -> int:  # ИЗМЕНЕНО: Any вместо str
        """Подсчет токенов в тексте"""
        try:
            if text is None:
                return 0
            text_str = str(text)
            return len(self.encoding.encode(text_str))
        except Exception:
            return len(str(text)) // 4

    def save_tokens_stats(self):
        """Сохранение статистики токенов в файл"""
        try:
            stats = {
                "session_id": self.session_id,
                "phase_name": self.phase_name,
                "logger_name": self.name,
                "last_update": datetime.now().isoformat(),
                "stats": {
                    "total_input_tokens": self.local_state["total_input_tokens"],
                    "total_output_tokens": self.local_state["total_output_tokens"],
                    "total_tokens": self.local_state["total_input_tokens"] + self.local_state["total_output_tokens"],
                    "warnings_count": self.local_state.get("warnings_count", 0),  # ДОБАВЛЕНО
                },
                "local_state": self.local_state,
            }

            with open(self.tokens_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

        except Exception as e:
            self.logger.warning(f"⚠️  Не удалось сохранить статистику токенов: {e}")

    def log_function_start(self, function_name: str, **input_data):
        """Логирование начала функции"""
        self.local_state["current_function"] = function_name
        self.local_state["function_start_time"] = datetime.now()  # ИСПРАВЛЕНО: datetime объект вместо строки

        self.logger.info(f"🎯 НАЧАЛО: {function_name}")

        input_tokens = 0
        for key, value in input_data.items():
            if value is not None:
                tokens = self.count_tokens(value)
                input_tokens += tokens

                # ИСПРАВЛЕНО: логика сравнения токенов
                if tokens > 1000:
                    self.logger.debug(f"📥 {key}: {tokens} токенов (большой)")
                elif tokens > 100:
                    self.logger.debug(f"📥 {key}: {tokens} токенов")
                else:
                    self.logger.debug(f"📥 {key}: {tokens} токенов (малый)")

        if input_tokens > 0:
            self.logger.info(f"📊 Общий размер входных данных: {input_tokens} токенов")

    def log_function_end(self, function_name: str, **output_data):
        """Логирование окончания функции"""
        # Вычисляем время выполнения
        duration = None
        if "function_start_time" in self.local_state:
            start_time = self.local_state["function_start_time"]
            duration = (datetime.now() - start_time).total_seconds()

        self.logger.info(f"✅ ЗАВЕРШЕНИЕ: {function_name}" + (f" за {duration:.2f}с" if duration else ""))

        # Логируем выходные данные
        output_tokens = 0
        for key, value in output_data.items():
            if value is not None:
                tokens = self.count_tokens(value)
                output_tokens += tokens

                if tokens > 1000:
                    self.logger.debug(f"📤 {key}: {tokens} токенов (большой)")
                elif tokens > 100:
                    self.logger.debug(f"📤 {key}: {tokens} токенов")
                else:
                    self.logger.debug(f"📤 {key}: {tokens} токенов (малый)")

        if output_tokens > 0:
            self.logger.info(f"📊 Общий размер выходных данных: {output_tokens} токенов")

        # Сохраняем статистику
        self.save_tokens_stats()

    def safe_llm_call(self, function_name: str, model: str = "", **prompt_components):
        """Контекстный менеджер для безопасного вызова LLM"""
        return LLMCallContext(self, function_name, model, prompt_components)

    def log_llm_response(self, function_name: str, response_data: Any):
        """Логирование ответа LLM (вызывается вручную после получения ответа)"""
        if response_data is None:
            self.logger.warning(f"⚠️  Пустой ответ от LLM в {function_name}")
            return

        # Подсчитываем токены ответа
        response_str = getattr(response_data, "content", str(response_data))
        output_tokens = self.count_tokens(response_str)

        # Обновляем статистику
        self.local_state["total_output_tokens"] += output_tokens

        self.logger.info(f"📤 LLM ОТВЕТ [{function_name}]: {output_tokens} токенов")

        # Если ответ очень большой - логируем превью
        if output_tokens > 5000:
            preview = response_str[:200] + "..." if len(response_str) > 200 else response_str
            self.logger.debug(f"📝 Превью ответа: {preview}")

        # Сохраняем статистику
        self.save_tokens_stats()

    def log_error(self, function_name: str, error: Exception, **context):
        """Логирование ошибки"""

        self.logger.error(f"❌ ОШИБКА в {function_name}: {type(error).__name__}: {str(error)}")

        # Логируем контекст ошибки
        if context:
            self.logger.debug(f"🔍 Контекст ошибки: {context}")

        # Логируем трейсбек в debug режиме
        self.logger.debug("📋 Трейсбек:", exc_info=True)

        # Если это ошибка токенов - особое внимание
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ["token", "context", "length", "limit", "maximum"]):
            self.logger.critical(f"🚨 ОШИБКА ТОКЕНОВ: {str(error)}")
            self.logger.critical("📊 Текущая статистика токенов:")
            self.logger.critical(f"  📥 Входящие: {self.local_state['total_input_tokens']}")
            self.logger.critical(f"  📤 Исходящие: {self.local_state['total_output_tokens']}")
            self.logger.critical(
                f"  🔢 Всего: {self.local_state['total_input_tokens'] + self.local_state['total_output_tokens']}"
            )

        # Сохраняем статистику
        self.save_tokens_stats()

    def log_warning(self, message: str):
        """Логирование предупреждения"""
        # ИСПРАВЛЕНО: инициализация счетчика если его нет
        if "warnings_count" not in self.local_state:
            self.local_state["warnings_count"] = 0
        self.local_state["warnings_count"] += 1
        self.logger.warning(f"⚠️  {message}")

    def log_info(self, message: str):
        """Логирование информации"""
        self.logger.info(f"ℹ️  {message}")

    def log_debug(self, message: str):
        """Логирование отладочной информации"""
        self.logger.debug(f"🔧 {message}")

    def get_stats(self) -> dict[str, Any]:
        """Получение текущей статистики"""
        return {
            "session_id": self.session_id,
            "phase_name": self.phase_name,
            "logger_name": self.name,
            "input_tokens": self.local_state["total_input_tokens"],
            "output_tokens": self.local_state["total_output_tokens"],
            "total_tokens": self.local_state["total_input_tokens"] + self.local_state["total_output_tokens"],
            "warnings": self.local_state.get("warnings_count", 0),  # ИСПРАВЛЕНО: get с default
            "local_state": self.local_state.copy(),
        }


class LLMCallContext:
    """Контекстный менеджер для LLM вызовов"""

    def __init__(self, logger_instance: SafeLogger, function_name: str, model: str, prompt_components: dict[str, Any]):
        self.logger = logger_instance
        self.function_name = function_name
        self.model = model
        self.prompt_components = prompt_components
        self.start_time = None
        self.input_tokens = 0

    def __enter__(self):
        """Вход в контекст - логируем перед LLM вызовом"""
        self.start_time = datetime.now()

        # Подсчитываем входящие токены
        self.input_tokens = 0
        prompt_details = {}

        for key, value in self.prompt_components.items():
            if value is not None:
                tokens = self.logger.count_tokens(value)
                self.input_tokens += tokens
                prompt_details[key] = {"tokens": tokens, "size_chars": len(str(value))}

        # Обновляем статистику
        self.logger.local_state["total_input_tokens"] += self.input_tokens

        # Логируем вызов
        self.logger.logger.info(f"🤖 LLM ВЫЗОВ [{self.function_name}]:")
        self.logger.logger.info(f"  📊 Модель: {self.model}")
        self.logger.logger.info(f"  📥 Входящие токены: {self.input_tokens}")

        # Предупреждения о размере промпта
        if self.input_tokens > 20000:
            self.logger.logger.critical(f"🚨 КРИТИЧЕСКИЙ РАЗМЕР ПРОМПТА: {self.input_tokens} токенов!")
        elif self.input_tokens > 15000:
            self.logger.logger.warning(f"⚠️  БОЛЬШОЙ ПРОМПТ: {self.input_tokens} токенов!")
        elif self.input_tokens > 10000:
            self.logger.logger.warning(f"🟡 Крупный промпт: {self.input_tokens} токенов")

        # Детали компонентов промпта
        self.logger.logger.debug("📝 Компоненты промпта:")
        for key, details in prompt_details.items():
            self.logger.logger.debug(f"  {key}: {details['tokens']} токенов ({details['size_chars']} символов)")

        # Принудительно сохраняем состояние перед LLM вызовом
        self.logger.save_tokens_stats()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекста - логируем результат"""
        if self.start_time is not None:
            duration = (datetime.now() - self.start_time).total_seconds()

            if exc_type is None:
                # Успешное выполнение
                self.logger.logger.info(f"✅ LLM вызов [{self.function_name}] завершён за {duration:.2f}с")
                self.logger.logger.info(f"  📥 Потрачено токенов на промпт: {self.input_tokens}")
                self.logger.logger.info(f"  ⏱️  Время выполнения: {duration:.2f}с")

            else:
                # Ошибка во время вызова
                self.logger.log_error(
                    self.function_name,
                    exc_val,
                    model=self.model,
                    input_tokens=self.input_tokens,
                    duration=duration,
                    prompt_components=list(self.prompt_components.keys()),
                )

        # Сохраняем статистику в любом случае
        self.logger.save_tokens_stats()

        # Не подавляем исключения
        return False


def create_logger(name: str = "main", phase_name: str = "MAIN") -> SafeLogger:
    """Создание простого логгера"""
    return SafeLogger(name, phase_name)
