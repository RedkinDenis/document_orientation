import cv2
import numpy as np
from skimage import morphology

def is_line_upside_down(line):
    """
    Определяет, перевернута ли строка.
    
    :param line: Изображение строки.
    :return: True, если строка перевернута, иначе False.
    """
    # Адаптивная бинаризация
    binary = cv2.adaptiveThreshold(
        line, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Вертикальная проекция (сумма пикселей по строкам)
    vertical_projection = np.sum(binary, axis=1)
    
    # Разделяем проекцию на верхнюю и нижнюю части
    height = binary.shape[0]
    upper_half = vertical_projection[:height // 2]
    lower_half = vertical_projection[height // 2:]
    
    # Нормализация гистограммы
    upper_sum = np.sum(upper_half) / len(upper_half)
    lower_sum = np.sum(lower_half) / len(lower_half)
    
    # Сравниваем нормализованные суммы
    if upper_sum > lower_sum:
        return False  # Строка в нормальной ориентации
    else:
        return True  # Строка перевернута
