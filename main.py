import cv2
import easyocr
import gradio as gr
import numpy as np


def preprocess_image(image):
    """Предобработка изображения для улучшения качества распознавания."""
    # Переводим в черно-белый формат
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Метод Otsu автоматически подбирает контрастность для текста
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def recognize_document(input_image):
    if input_image is None:
        return "Пожалуйста, загрузите изображение文档."

    try:
        # 1. Очищаем картинку от шумов
        processed_img = preprocess_image(input_image)

        # 2. Инициализируем нейросеть
        reader = easyocr.Reader(["ru", "en"], gpu=False)

        # 3. Распознаем текст
        result = reader.readtext(processed_img)

        # 4. Собираем текст в одну строку
        recognized_lines = []
        for bbox, text, confidence in result:
            if confidence > 0.20:
                recognized_lines.append(text)

        if not recognized_lines:
            return "Текст на документе не найден или изображение слишком размыто."

        return "\n".join(recognized_lines)

    except Exception as e:
        return f"Произошла ошибка при обработке: {str(e)}"


# Создаем веб-интерфейс программы
interface = gr.Interface(
    fn=recognize_document,
    inputs=gr.Image(label="Загрузите скан или фото документа"),
    outputs=gr.Textbox(label="Распознанный текст документа", lines=10),
    title="Нейросеть распознавания документов (Гостиничное дело)",
    description="Система для автоматического считывания паспортных данных и текстов договоров.",
)

if __name__ == "__main__":
    # Запуск локального веб-сервера
    interface.launch()


