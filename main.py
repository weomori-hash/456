import cv2
import easyocr
import numpy as np


def preprocess_image(image_path):
    """Предобработка скана для улучшения качества распознавания."""
    # Загружаем изображение в черно-белом формате
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Метод Otsu автоматически подбирает порог контрастности для текста
    _, thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def recognize_document(image_path):
    # 1. Очищаем картинку от шумов
    processed_img = preprocess_image(image_path)

    # 2. Инициализируем нейросеть (подгружаем русский и английский)
    reader = easyocr.Reader(["ru", "en"], gpu=False)

    # 3. Распознаем текст
    print("Сканирование и распознавание текста...")
    result = reader.readtext(processed_img)

    # 4. Выводим результат
    print("\n--- Результат распознавания ---")
    for bbox, text, confidence in result:
        # Отсекаем неуверенные результаты (точность ниже 20%)
        if confidence > 0.20:
            print(f"Текст: {text} (Уверенность: {confidence:.2f})")


if __name__ == "__main__":
    # Имя файла картинки, которую вы положите в папку с кодом
    IMAGE_PATH = "scan_document.jpg"

    try:
        recognize_document(IMAGE_PATH)
    except Exception as e:
        print(f"Произошла ошибка: {e}")
