"""
Главный модуль проекта предсказания рисков
"""

def predict_risk(age, cholesterol, blood_pressure):
    """
    Предсказывает сердечно-сосудистый риск
    
    Args:
        age: возраст пациента (лет)
        cholesterol: уровень холестерина (мг/дл)
        blood_pressure: систолическое давление (мм рт.ст.)
    
    Returns:
        float: вероятность риска от 0 до 1
    
    Example:
        >>> risk = predict_risk(45, 220, 140)
        >>> print(f"Риск: {risk:.2%}")
    """
    risk_score = (age / 100) * 0.3 + (cholesterol / 300) * 0.4 + (blood_pressure / 200) * 0.3
    return min(risk_score, 1.0)


def load_model(model_path="model.pkl"):
    """
    Загружает обученную модель
    
    Args:
        model_path: путь к файлу модели
    
    Returns:
        Объект модели
    """
    return None


if __name__ == "__main__":
    risk = predict_risk(age=45, cholesterol=220, blood_pressure=140)
    print(f"Риск: {risk:.2%}")