import streamlit as st
import analysis_and_model as am
import presentation as pres

# Список страниц
pages = {
    "Анализ и модель": am.analysis_and_model_page,
    "Детальный анализ данных": am.detailed_data_analysis,
    "Презентация": pres.presentation_page
}

# Создание навигации в боковой панели
selected_page = st.sidebar.selectbox(
    "Выберите страницу", options=list(pages.keys()))

# Отображение выбранной страницы
pages[selected_page]()
