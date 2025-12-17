"""
Ana Weather - Summary
Каждые 10 минут читает историю из JSON и отправляет summary в Telegram
"""

import time
from datetime import datetime
from day13_weather_mcp import create_agent
from telegram_notifier import TelegramNotifier


def generate_summary(agent, notifier: TelegramNotifier, city: str = "Warsaw"):
    """Читает JSON и генерирует summary"""
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Генерация summary...")
    
    try:
        # Просим Claude проанализировать историю из JSON
        response = agent.chat(
            f"Проанализируй историю погоды в городе {city}. "
            f"Используй инструмент weather_history с action='stats' чтобы получить статистику. "
            f"Сделай краткое summary на русском языке: "
            f"- за какой период данные, "
            f"- как менялась температура (мин/макс/средняя), "
            f"- как менялся ветер, "
            f"- какие тренды. "
            f"Отвечай кратко, максимум 5-6 строк.",
            silent=True
        )
        
        print(response)
        
        # Формируем сообщение для Telegram
        current_time = datetime.now().strftime('%H:%M, %d.%m.%Y')
        
        telegram_message = f"""
<b>Ana Weather - Summary</b>

{current_time}

{response}
"""
        
        notifier.send_message(telegram_message)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Отправлено в Telegram")
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ошибка: {e}")
        notifier.send_message(f"<b>Ana Weather - Ошибка</b>\n\n{e}")


def main():
    """Главная функция daemon"""
    
    print("\n" + "="*60)
    print("Ana Weather - Summary")
    print("="*60)
    
    agent = create_agent()
    if not agent:
        print("Ошибка: не удалось создать агента")
        return
    
    notifier = TelegramNotifier()
    if not notifier.enabled:
        print("Ошибка: Telegram не настроен")
        return
    
    CITY = "Warsaw"
    INTERVAL_MINUTES = 10
    
    print(f"Город: {CITY}")
    print(f"Интервал: {INTERVAL_MINUTES} минут")
    print(f"Читает из: weather_history.json")
    print("="*60 + "\n")
    
    # Приветственное сообщение
    welcome = f"""
<b>Ana Weather Summary запущен</b>

Город: {CITY}
Интервал: каждые {INTERVAL_MINUTES} минут
Запущен: {datetime.now().strftime('%H:%M, %d.%m.%Y')}
"""
    
    notifier.send_message(welcome)
    
    # Первый summary сразу
    print("[Summary #1]")
    generate_summary(agent, notifier, CITY)
    
    summary_count = 1
    
    try:
        while True:
            # Ждём интервал
            for i in range(INTERVAL_MINUTES):
                remaining = INTERVAL_MINUTES - i
                if i % 10 == 0 and i > 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Следующий summary через {remaining} минут")
                time.sleep(60)
            
            # Генерируем summary
            summary_count += 1
            print(f"\n[Summary #{summary_count}]")
            generate_summary(agent, notifier, CITY)
    
    except KeyboardInterrupt:
        print(f"\n\nSummary Daemon остановлен")
        print(f"Всего summary: {summary_count}\n")
        
        goodbye = f"""
<b>Ana Weather Summary остановлен</b>

Всего summary: {summary_count}
Остановлен: {datetime.now().strftime('%H:%M, %d.%m.%Y')}
"""
        
        notifier.send_message(goodbye)


if __name__ == "__main__":
    main()
