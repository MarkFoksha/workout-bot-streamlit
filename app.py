# app.py — Головний інтерфейс Streamlit для Фітнес-асистента WorkoutBot

import uuid
import pandas as pd
import streamlit as st

from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage, SystemMessage

# Імпорт логіки нашого LangGraph-агента
from agent import (
    create_agent,
    extract_response_text,
    extract_tools_debug,
    MODEL_NAME,
)

# ============================================================
# НАЛАШТУВАННЯ СТОРІНКИ
# ============================================================
st.set_page_config(
    page_title="WorkoutBot — AI Тренер",
    page_icon="🏋️‍♂️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# ІНІЦІАЛІЗАЦІЯ КЛІЄНТІВ ТА API
# ============================================================
@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

@st.cache_resource
def get_langgraph_agent(api_key: str, model_name: str):
    return create_agent(api_key, model_name)

api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Не знайдено API-ключа у файлі .streamlit/secrets.toml")
    st.stop()

# ============================================================
# СТАН СЕСІЇ STREAMLIT
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# Стартові дані для твого дизайну
if "workout_plan" not in st.session_state:
    st.session_state.workout_plan = [
        {"day": "Понеділок", "exercises": "Присідання, Випади"},
        {"day": "Вівторок", "exercises": "Відпочинок"}
    ]

if "workouts_completed" not in st.session_state:
    st.session_state.workouts_completed = 0

if "streak" not in st.session_state:
    st.session_state.streak = 0

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "Ти професійний фітнес-інструктор «WorkoutBot». Спілкуйся виключно українською мовою."

# ============================================================
# UI: БІЧНА ПАНЕЛЬ (ТВІЙ ДИЗАЙН)
# ============================================================
with st.sidebar:
    st.header("💪 Мої налаштування")
    
    fitness_level = st.radio(
        "Рівень підготовки",
        ["🌱 Початківець", "💪 Середній", "🔥 Просунутий"]
    )
    
    workout_duration = st.slider(
        "Тривалість тренування (хв)",
        min_value=15,
        max_value=90,
        value=45,
        step=15
    )

    st.session_state.fitness_settings = {
        "level": fitness_level,
        "duration": workout_duration
    }
    
    # ============================================================
    # СТАТИСТИКА ТА ТРЕКІНГ УСПІХУ
    # ============================================================
    st.divider()
    st.header("📊 Статистика")
    
    workouts_done = st.session_state.get("workouts_completed", 0)
    streak = st.session_state.get("streak", 0)
    
    # Виводимо метрики
    col1, col2 = st.columns(2)
    col1.metric("🏋️‍♂️ Тренувань", workouts_done)
    col2.metric("🔥 Streak", f"{streak} дн.")
    
    # КНОПКА НАСТУПНОГО ДНЯ / ЗАВЕРШЕННЯ ТРЕНУВАННЯ
    if st.button("✅ Я виконав сьогоднішнє тренування!", use_container_width=True):
        st.session_state.workouts_completed += 1
        st.session_state.streak += 1
        st.toast("💪 Супер! Зараховано наступний день дисципліни!")
        st.rerun() # Перезапускаємо інтерфейс, щоб метрики і мотиваційне повідомлення миттєво оновилися
    
    # Тижневий план як таблиця
    st.subheader("📅 План на тиждень")
    workout_plan = st.session_state.get("workout_plan", [])
    
    if workout_plan:
        df = pd.DataFrame(workout_plan)
        
        # Робимо список вправ зручним рядком для відображення, якщо раптом агент поверне список
        if "exercises" in df.columns:
            df["exercises"] = df["exercises"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            
        # Editable dataframe
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True
        )
        if st.button("💾 Зберегти зміни"):
            # Конвертуємо назад для агента
            new_plan = []
            for _, row in edited_df.iterrows():
                ex_list = [e.strip() for e in str(row["exercises"]).split(",") if e.strip()]
                new_plan.append({"day": row["day"], "exercises": ex_list})
            
            st.session_state.workout_plan = new_plan
            st.success("План оновлено!")
    else:
        st.info("План ще не створено. Попроси тренера скласти програму!")
        
    # Мотиваційне повідомлення
    if streak >= 7:
        st.success(f"🎉 Чудово! {streak} днів поспіль — це справжня дисципліна!")
    elif streak >= 3:
        st.info(f"💪 Гарна робота! {streak} дні поспіль, продовжуй!")

    st.divider()
    
    # Додаткові системні налаштування для чат-бота
    st.subheader("⚙️ Керування ботом")
    mode = st.radio("Режим роботи ШІ:", ["🛠️ Інтелектуальний Агент", "💬 Звичайний чат"], index=0)
    
    if st.button("🗑️ Очистити історію чату", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ============================================================
# ГОЛОВНИЙ ЕКРАН ЧАТУ
# ============================================================
st.title("🏋️‍♂️ Ваш Персональний ШІ-Тренер")

# Функція стрімінгу для стандартної моделі (без агента)
def stream_normal_gemini(prompt: str, history: list, sys_instruction: str):
    client = get_gemini_client(api_key)
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
    
    full_prompt = f"[Параметри: {fitness_level}, {workout_duration}хв]. {sys_instruction}"
    
    stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=full_prompt, temperature=0.7)
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text

# Обробка логіки інтелектуального агента
def run_agent_workflow(prompt: str):
    agent = get_langgraph_agent(api_key, MODEL_NAME)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    full_sys_prompt = (
        f"{st.session_state.system_prompt}\n"
        f"Параметри: Рівень: {fitness_level}, Тривалість: {workout_duration} хв."
    )
    sys_msg = SystemMessage(content=full_sys_prompt, id="user_system_prompt")
    
    result = agent.invoke(
        {
            "messages": [sys_msg, HumanMessage(content=prompt)],
            "workout_plan": st.session_state.workout_plan
        },
        config
    )
    
    # Синхронізація плану
    if "workout_plan" in result:
        st.session_state.workout_plan = result["workout_plan"]
        
    return extract_response_text(result["messages"][-1]), extract_tools_debug(result["messages"])

# Відображення історії
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("👋 Привіт! Я твій тренер. Бачу твої налаштування зліва. Давай складемо план тренувань!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Введення користувача
if user_query := st.chat_input("Напиши тренеру (наприклад: 'Додай на середу прес')..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        if "Агент" in mode:
            with st.spinner("Тренер формує план..."):
                ans_text, debug_info = run_agent_workflow(user_query)
            st.markdown(ans_text)
            
            if debug_info:
                with st.expander("🛠️ Лог роботи інструментів (Debug)", expanded=False):
                    st.json(debug_info)
            final_response = ans_text
        else:
            final_response = st.write_stream(
                stream_normal_gemini(user_query, st.session_state.messages[:-1], st.session_state.system_prompt)
            )
            
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    
    # Перезапуск інтерфейсу для миттєвого оновлення таблиці, якщо працював агент
    if "Агент" in mode:
        st.rerun()