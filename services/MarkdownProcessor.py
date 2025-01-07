# services/markdown_service.py

import re


class MarkdownProcessor:
    """
    Класс для обработки Markdown-разметки в тексте.
    """

    @staticmethod
    async def strip_markdown(text):
        """
        Удаляет Markdown-разметку из текста.
        """
        if not text:
            return text

        # Удаление заголовков (например, ### Заголовок)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # Удаление жирного текста (**текст** или __текст__)
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)

        # Удаление курсива (*текст* или _текст_)
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)

        # Удаление списков (- или *)
        text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)

        # Удаление ссылок [текст](URL)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Удаление изображений ![alt](URL)
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)

        # Удаление горизонтальных линий
        text = re.sub(r'^---$', '', text, flags=re.MULTILINE)

        # Удаление обратных кавычек (для кода)
        text = re.sub(r'`', '', text)

        # Удаление лишних пустых строк
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()
