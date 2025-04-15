import pandas as pd
import optuna
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import requests
import xgboost as xgb

from matplotlib.patches import Patch
from sklearn.model_selection import KFold, train_test_split
from ucimlrepo import fetch_ucirepo
from clearml import Task, OutputModel, Dataset, Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score


def analysis_and_model_page() -> None:
    """
    Страница анализа данных и обучения модели в Streamlit.
    Включает предобработку, подбор гиперпараметров, кросс-валидацию,
    сохранение модели и интерактивное предсказание.
    """
    # FIXME: Надо как-то попробовать разделить UI Streamlit-приложение
    # и обучение модели
    st.title("Анализ данных и модель")

    # Инициализация ClearML задачи
    task = Task.init(
        project_name="Predictive Maintenance",
        task_name="Model Training"
    )

    # Загрузка датасета: если ресурс недоступен - загружаем локально
    try:
        dataset = fetch_ucirepo(id=601)
        data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    except ConnectionError:
        data = pd.read_csv("data/predictive_maintenance.csv")

    # Загрузка датасета в ClearML
    dataset_clearml = Dataset.create(
        dataset_name="Predictive Maintenance Data",
        dataset_project="Predictive Maintenance"
    )
    data.to_csv("predictive_maintenance.csv", index=False)
    dataset_clearml.add_files("predictive_maintenance.csv")
    dataset_clearml.upload()
    dataset_clearml.finalize()

    # Предобработка: удаление неинформативных признаков
    data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
              errors='ignore',
              inplace=True)

    print(data.head)

    # Кодирование категориального признака
    encoder = LabelEncoder()
    data['Type'] = encoder.fit_transform(data['Type'])

    # Масштабирование числовых признаков
    scaler = StandardScaler()
    numerical_features = [
        'Air temperature',
        'Process temperature',
        'Rotational speed',
        'Torque',
        'Tool wear'
    ]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Сохранение артефактов
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    task.upload_artifact(name='encoder', artifact_object='encoder.pkl')
    task.upload_artifact(name='scaler', artifact_object='scaler.pkl')

    # Разделение данных
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Оптимизация гиперпараметров
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_xgb(trial, X_train, X_test, y_train, y_test),
        n_trials=50
    )
    best_params = study.best_params
    best_params.update(
        {
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
    )
    task.connect(best_params)

    st.write("Лучшие параметры XGBoost:", best_params)

    # Кросс-валидация и логирование
    best_model = None
    best_roc_auc = -1
    best_metrics = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        st.write(f"Обучение на фолде {fold+1}/5")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_val_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]

        roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
        st.write(f"Fold {fold+1}: ROC-AUC = {roc_auc:.4f}")
        task.get_logger().report_scalar(
            "ROC-AUC", f"Fold {fold+1}", roc_auc, iteration=fold
        )

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            best_metrics = {
                "accuracy": accuracy_score(y_val_fold, y_pred),
                "confusion_matrix": confusion_matrix(y_val_fold, y_pred),
                "classification_report": classification_report(
                    y_val_fold, y_pred)
            }

    # Сохранение модели
    best_model.save_model("best_model.json")
    output_model = OutputModel(task=task, framework="XGBoost")
    output_model.update_weights("best_model.json")
    task.update_output_model(
        model_path="best_model.json", name="Best XGBoost Fold Model"
    )

    # Загрузка модели
    model = Model(model_id=output_model.id)
    model_path = model.get_local_copy()
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(model_path)

    # Отображение результатов
    st.write(f"Загружена лучшая модель с ROC-AUC: {best_roc_auc:.4f}")
    st.header("Результаты обучения модели")
    st.subheader("XGBoost")
    st.write(
        f"Accuracy: {best_metrics['accuracy']:.2f}, " +
        f"ROC-AUC: {best_roc_auc:.2f}"
    )

    # Визуализация confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(
        best_metrics['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )
    st.pyplot(fig)

    # Интерфейс Streamlit
    st.title("Прогнозирование отказов оборудования")

    with st.form("prediction_form"):  # Ввод данных пользователем
        prod_type = st.selectbox("Идентификатор продукта", ["L", "M", "H"])
        air_temp = st.number_input("Температура окружающей среды [K]")
        process_temp = st.number_input("Рабочая температура [K]")
        rotational_speed = st.number_input("Скорость вращения [rpm]")
        torque = st.number_input("Крутящий момент [Nm]")
        tool_wear = st.number_input("Износ инструмента [мин]")

        submit_button = st.form_submit_button("Предсказать")

    if submit_button:
        # Формирование запроса для API
        response = requests.post(
            "http://127.0.0.1:8080/serve/predictive_maintenance",
            json={
                "Product ID": prod_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear
            }
        )
        # Извлекаем вероятность отказа
        prediction_proba = response.json()["prediction"][0]
        # Отображение результатов предсказания
        st.write(f"Предсказание (вероятность отказа): {prediction_proba:.2f}")


def objective_xgb(trial: optuna.trial.Trial,
                  X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_test: pd.Series) -> float:
    """
    Оптимизационная функция для Optuna для подбора гиперпараметров XGBoost.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.1, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)


def detailed_data_analysis():
    """
    Метод предоставления детального анализа данных.
    """
    st.title("Детальный анализ данных")

    # Загрузка данных
    data = pd.read_csv("predictive_maintenance.csv")

    # Удаление неинформативных колонок
    columns_to_drop = ['UDI', 'Product ID']
    data = data.drop(columns=[col for col in columns_to_drop if
                              col in data.columns])

    st.header("Общее описание данных")
    st.write(data.describe())

    # Выбор типа анализа
    analysis_type = st.sidebar.selectbox(
        "Выберите анализ",
        ["Распределение признаков",
         "Корреляционная матрица",
         "Зависимости признаков",
         "Распределение отказов"])

    if analysis_type == "Корреляционная матрица":
        st.subheader("Корреляционная матрица")
        # Исключаем категориальные признаки
        corr_data = data.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif analysis_type == "Распределение признаков":
        st.subheader("Гистограммы распределения признаков")

        failure_columns = [
            'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        for col in data.select_dtypes(include=np.number).columns:
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=False, ax=ax, bins=30)

            # Добавление кастомной легенды
            if col in failure_columns:
                counts = data[col].value_counts().to_dict()
                legend_labels = [f"{int(k)}: {v}" for k, v in sorted(
                    counts.items())]
                legend_patches = [
                    Patch(color="blue", label=label) for label
                    in legend_labels
                ]
                ax.legend(handles=legend_patches, title="Значения")

            ax.set_title(f"График распределения параметра {col}")
            st.pyplot(fig)

    elif analysis_type == "Зависимости признаков":
        st.subheader("Scatter Plot зависимостей")
        x_feature = st.selectbox("Выберите ось X", data.columns)
        y_feature = st.selectbox("Выберите ось Y", data.columns)
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_feature, y=y_feature,
                        hue='Machine failure', ax=ax)
        st.pyplot(fig)

    elif analysis_type == "Распределение отказов":
        st.subheader("Распределение отказов по типам")
        fig, ax = plt.subplots()
        sns.countplot(x='Machine failure', data=data, ax=ax)
        st.pyplot(fig)

        st.subheader("Зависимость отказов от параметров")
        for col in ['Air temperature', 'Process temperature',
                    'Rotational speed', 'Torque', 'Tool wear']:
            fig, ax = plt.subplots()
            sns.boxplot(x='Machine failure', y=col, data=data, ax=ax)
            st.pyplot(fig)
