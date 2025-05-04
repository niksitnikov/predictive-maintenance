from clearml import Task
import pandas as pd
import xgboost as xgb
import joblib
import logging


# Отображение названий признаков с единицами -> стандартные названия
feature_rename_map = {
    'Air temperature [K]': 'Air temperature',
    'Process temperature [K]': 'Process temperature',
    'Rotational speed [rpm]': 'Rotational speed',
    'Torque [Nm]': 'Torque',
    'Tool wear [min]': 'Tool wear'
}

numerical_features = list(feature_rename_map.values())


class Preprocess:
    def __init__(self):
        task = Task.get_task(task_id="<service_id>")

        logging.basicConfig(level=logging.INFO)
        logging.info("[PREPROCESS] Current task: %s", task)

        encoder_path = task.artifacts['encoder'].get_local_copy()
        scaler_path = task.artifacts['scaler'].get_local_copy()

        self.encoder = joblib.load(encoder_path)
        self.scaler = joblib.load(scaler_path)

    def preprocess(self, data, *args, **kwargs):
        logging.info("[PREPROCESS] Received data: %s", data)

        if not isinstance(data, list):
            data = [data]
        df = pd.DataFrame(data)

        # Переименуем признаки с единицами измерения
        df.rename(columns=feature_rename_map, inplace=True)

        # Добавим 'Type' со значением по умолчанию, если отсутствует
        if 'Type' not in df.columns:
            df['Type'] = 'L'
            logging.info("[PREPROCESS] 'Type' not in input, defaulting to 'L'")

        # Кодируем 'Type'
        df['Type'] = self.encoder.transform(df['Type'])

        # Масштабируем числовые признаки
        df[numerical_features] = self.scaler.transform(df[numerical_features])

        expected_columns = ['Type'] + numerical_features
        df = df[expected_columns].astype(float)

        logging.info("[PREPROCESS] Final DataFrame:\n%s", df.head())

        dmatrix = xgb.DMatrix(df.values, feature_names=expected_columns)
        return dmatrix

    def postprocess(self, data, *args, **kwargs):
        logging.info("[POSTPROCESS] Type of 'data': %s", type(data))
        logging.info("[POSTPROCESS] Value of 'data': %s", data)
        prediction_proba = data[0]
        formatted_prediction = f"{prediction_proba:.2f}"
        return {
            "prediction": data.tolist(),
            "comment": f"Вероятность отказа: {formatted_prediction}"
        }
