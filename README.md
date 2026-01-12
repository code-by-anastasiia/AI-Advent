# Мой проект: Предсказание рисков

Это проект для предсказания сердечно-сосудистых рисков.

## Основные функции

### predict_risk()
Функция для предсказания риска на основе параметров пациента.

Параметры:
- age: возраст (лет)
- cholesterol: холестерин (мг/дл)
- blood_pressure: давление (мм рт.ст.)

Возвращает: вероятность риска (0-1)

### load_model()
Загружает обученную модель из файла model.pkl

## Установка
```bash
pip install -r requirements.txt
```

## Использование
```python
from main import predict_risk

risk = predict_risk(age=45, cholesterol=220, blood_pressure=140)
print(f"Риск: {risk:.2%}")
```
