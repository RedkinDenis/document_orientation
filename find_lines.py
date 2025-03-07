import cv2
import numpy as np
from skimage import morphology

from image_processing import *

def find_most_dense_region(image, region_height=50):
    """
    Находит участок с наибольшей плотностью пикселей (одну строку).
    
    :param image: Изображение в градациях серого.
    :param region_height: Высота участка для анализа.
    :return: Координаты участка (y_start, y_end).
    """
    # Адаптивная бинаризация
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Вертикальная проекция (сумма пикселей по строкам)
    vertical_projection = np.sum(binary, axis=1)
    
    # Находим участок с наибольшей плотностью
    max_density = 0
    best_row = 0
    
    for row in range(len(vertical_projection) - region_height):
        density = np.sum(vertical_projection[row:row + region_height])
        if density > max_density:
            max_density = density
            best_row = row
    
    return best_row, best_row + region_height

def find_and_ignore_first_lines(image, num_lines_to_ignore=2, region_height=50):
    """
    Находит и закрашивает первые несколько строк.
    
    :param image: Изображение в градациях серого.
    :param num_lines_to_ignore: Количество строк для игнорирования.
    :param region_height: Высота участка для анализа.
    :return: Изображение с закрашенными строками.
    """
    working_image = image.copy()
    
    for _ in range(num_lines_to_ignore):
        # Находим участок с наибольшей плотностью белых пикселей
        y_start, y_end = find_most_dense_region(working_image, region_height)
        
        # Закрашиваем найденную строку
        working_image[y_start:y_end, :] = 0  # Закрашиваем черным цветом
    
    return working_image

def is_mostly_black_line(line, threshold=0.02):

    _, line = cv2.threshold(line, 127, 255, cv2.THRESH_BINARY)
    """
    Проверяет, является ли строка "почти черной" (менее 10% белых пикселей).
    
    :param line: Изображение строки.
    :param threshold: Порог для процента белых пикселей (по умолчанию 10%).
    :return: True, если строка "почти черная", иначе False.
    """
    # Подсчитываем количество белых пикселей
    white_pixels = np.sum(line == 255)
    
    # Общее количество пикселей в строке
    total_pixels = line.size
    
    # Вычисляем процент белых пикселей
    white_percentage = white_pixels / total_pixels
    
    # Проверяем, меньше ли процент белых пикселей порога
    # print(white_pixels, total_pixels, white_percentage)
    return white_percentage < threshold

def count_lines(image):
    # Удаление белых краев
    image = remove_white_margins(image)

    image = cv2.bitwise_not(image)
    # Копия изображения для закрашивания
    working_image = image.copy()

    count = 0

    while (True):
        # Находим участок с наибольшей плотностью белых пикселей
        y_start, y_end = find_most_dense_region(working_image, 60)
        
        # Закрашиваем найденную строку
        # print(1)
        line = working_image[y_start:y_end, :]
        if (is_mostly_black_line(line)):

            # cv2.imshow("image", line)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return count

        working_image[y_start:y_end, :] = 0  # Закрашиваем черным цветом
        count += 1