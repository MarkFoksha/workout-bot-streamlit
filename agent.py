# agent.py — LangGraph-агент для Фітнес-тренера «WorkoutBot»

from __future__ import annotations

# Стандартні бібліотеки Python для типізації та роботи з системою
import json
from typing import Annotated, Any
from typing_extensions import TypedDict

# LangChain: Модель Gemini, декоратор інструментів та типи повідомлень
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage, 
    ToolMessage
)

# LangGraph: Побудова графа, керування станом та пам'ять (Checkpointing)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# Інструмент для безпечного впорскування стану в інструменти
from langgraph.prebuilt import InjectedState

# ============================================================
# КОНСТАНТИ
# ============================================================
MODEL_NAME = "gemini-2.5-flash"

# Базова системна роль тренера
BASE_TRAINER_PROMPT = (
    "Ти — професійний ШІ-тренер та експерт з фітнесу «WorkoutBot». "
    "Твоя мета — допомагати користувачу складати та оптимізувати тижневий план тренувань. "
    "Правила:\n"
    "1) Коли користувач просить запланувати вправи на конкретний день, обов'язково "
    "викликай інструмент `add_workout_day`.\n"
    "2) Для перегляду конкретного дня використовуй `get_workout_for_day`.\n"
    "3) Для підсумку всього тижня використовуй `weekly_summary`.\n"
    "Будь мотивуючим, давай чіткі інструкції щодо виконання вправ."
)

# ============================================================
# СХЕМА СТАНУ
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    workout_plan: list[dict]  # Сховище плану: [{"day": "Пн", "exercises": [...]}, ...]

# ============================================================
# ІНСТРУМЕНТИ ДЛЯ АГЕНТА
# ============================================================
@tool
def add_workout_day(day: str, exercises: list[str]) -> str:
    """
    Додає або оновлює тренування на конкретний день тижня.
    Викликай цей інструмент, коли користувач каже фрази на кшталт: 'заплануй на понеділок: присідання, біг...'
    
    Args:
        day: День тижня (наприклад: 'Понеділок', 'Вівторок')
        exercises: Список вправ у вигляді масиву рядків
    """
    if not exercises:
        return f"На день '{day}' не передано жодної вправи. Уточни список."
    
    ex_str = ", ".join(exercises)
    return f"Успішно заплановано тренування на {day}: {ex_str}"

@tool
def get_workout_for_day(day: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Повертає список вправ на обраний день тижня з поточної бази даних.
    
    Args:
        day: День тижня, який цікавить користувача
    """
    workout_plan = state.get("workout_plan", [])
    if not workout_plan:
        return "План тренувань на тиждень зараз абсолютно порожній."

    for w in workout_plan:
        if w.get("day", "").lower() == day.lower():
            ex = w.get("exercises", [])
            if not ex:
                return f"На день '{day}' заплановано відпочинок або вправи не вказані."
            return f"Ось твій план на {day}: {', '.join(ex)}"

    return f"На день '{day}' у твоєму поточному розкладі нічого немає."

@tool
def weekly_summary(state: Annotated[dict, InjectedState]) -> str:
    """
    Генерує повний підсумок тижневого плану тренувань із бази даних користувача.
    """
    workout_plan = state.get("workout_plan", [])
    if not workout_plan:
        return "Тижневий план тренувань порожній. Давай складемо його разом!"

    lines = []
    days_count = 0
    for w in workout_plan:
        day = w.get("day", "Невідомий день")
        ex = w.get("exercises", [])
        if ex:
            days_count += 1
            lines.append(f"- **{day}**: {', '.join(ex)}")
        else:
            lines.append(f"- **{day}**: Відпочинок / План відсутній")

    return f"Твій поточний розклад містить тренування на {days_count} дн.:\n" + "\n".join(lines)

TOOLS = [add_workout_day, get_workout_for_day, weekly_summary]

# ============================================================
# ПОБУДОВА ГРАФА
# ============================================================
def create_agent(api_key: str, model_name: str = MODEL_NAME):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2, # Низька температура гарантує точний виклик функцій
        api_key=api_key,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        # Формуємо динамічний контекст для моделі, додаючи поточну структуру плану в промпт
        current_plan_json = json.dumps(state.get("workout_plan", []), ensure_ascii=False)
        dynamic_prompt = (
            f"{BASE_TRAINER_PROMPT}\n\n"
            f"[АКТУАЛЬНА БАЗА ДАНИХ ТРЕНУВАНЬ НА СЬОГОДНІ]:\n{current_plan_json}"
        )
        
        # Шукаємо кастомний системний повідомлення, якщо користувач змінив його в UI
        sys_content = dynamic_prompt
        for msg in state["messages"]:
            if getattr(msg, "id", "") == "user_system_prompt":
                sys_content = f"{msg.content}\n\n[АКТУАЛЬНА БАЗА ДАНИХ ТРЕНУВАНЬ]:\n{current_plan_json}"

        msgs = [SystemMessage(content=sys_content, id="user_system_prompt")] + state["messages"]
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    # Кастомний вузол для виконання інструментів та оновлення структурованого стейту
    standard_tool_node = ToolNode(TOOLS)

    def custom_tool_executor(state: AgentState) -> dict:
        # Виконуємо стандартний запуск інструменту
        output = standard_tool_node.invoke(state)
        
        # Перевіряємо, чи викликався інструмент додавання плану
        last_msg = state["messages"][-1]
        updated_plan = list(state.get("workout_plan", []))
        
        if hasattr(last_msg, "tool_calls"):
            for tool_call in last_msg.tool_calls:
                if tool_call["name"] == "add_workout_day":
                    day = tool_call["args"].get("day")
                    exercises = tool_call["args"].get("exercises", [])
                    
                    if day and isinstance(exercises, list):
                        # Видаляємо старий запис цього дня, якщо він існував
                        updated_plan = [w for w in updated_plan if w.get("day", "").lower() != day.lower()]
                        # Додаємо оновлений варіант
                        updated_plan.append({"day": day, "exercises": exercises})
        
        output["workout_plan"] = updated_plan
        return output

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", custom_tool_executor)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)

# ============================================================
# ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ UI
# ============================================================
def extract_response_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([part if isinstance(part, str) else part.get("text", "") for part in content])
    return str(content)

def extract_tools_debug(messages: list[Any]) -> list[dict]:
    debug = []
    for m in messages:
        if isinstance(m, ToolMessage):
            debug.append({"type": "tool_result", "name": m.name, "content": m.content})
        else:
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    debug.append({"type": "tool_call", "name": tc.get("name"), "args": tc.get("args")})
    return debug