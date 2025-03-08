import cv2
import pytesseract
from pytesseract import Output

def tesseract_detect_orientation(image):
    # Используем pytesseract для определения ориентации
    osd = pytesseract.image_to_osd(image, output_type=Output.DICT)

    # Выводим информацию о ориентации
    return osd['orientation']
