import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)  # Создание директории, если её нет
LOG_FILE = LOG_DIR / "app.log"

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Проверяем, не были ли уже добавлены хендлеры (чтобы избежать дублирования)
    if not logger.hasHandlers():
        # Хендлер для записи в файл
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Хендлер для вывода на консоль
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
