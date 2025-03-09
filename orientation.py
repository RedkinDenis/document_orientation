import cv2
import numpy as np

from image_processing import *
from lines_processing import *    

def orientation_detect(image):
    """
    это основная функция.

    image: входное изображение. должно быть открыто

    return: возвращает кортеж (orientation, angle)
    orientation: ориентация документа (портретная или альбомная)
    angle: угол на который необходимо повернуть документ для нормализации
    """

    if image is None:
        print("Ошибка: изображение не загружено.")
        return


    angle = 0
    orientation = ''
    working_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # проверяется правильность ориентации букв.
    # на данном этапе случаи поворота на 0 и 180 и на 90 и 270 градусов между собой не различимы
    if not is_text_orientation_right(working_image):
        # print('rotate')
        working_image = rotate_image_90(working_image)
        angle += 90
    # получаем документ который однозначно повернут на 0 или 180 градусов
    
    (w, h) = get_horizontal_bounding_box(working_image)

    # у текста с портретной ориентацией высота ограничивающей рамки больше чем ее ширина (и наоборот)
    if h > w:
        orientation = "portrait"
    else:
        orientation = "albumn"

    # если текст все же перевернут, добавляем 180 к углу поворота
    if (is_text_upside_down(working_image)):
        angle += 180

    return (orientation, angle)

def is_text_orientation_right(image):

    """
    Метод основан на том, что в английском и русском языках количество вертикальных составляющих
    букв значительно больше чем количество горизонтальных составляющих
    """

    # Размытие для уменьшения шума
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Ядра для вертикальных и горизонтальных линий (использованы стандартные ядра, взятые из интеренета)
    kernel_vertical = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])

    kernel_horizontal = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]])

    # Применение свертки
    vertical_lines = cv2.filter2D(blurred_image, -1, kernel_vertical)
    horizontal_lines = cv2.filter2D(blurred_image, -1, kernel_horizontal)

    # Пороговая обработка
    _, vertical_thresh = cv2.threshold(vertical_lines, 50, 255, cv2.THRESH_BINARY)
    _, horizontal_thresh = cv2.threshold(horizontal_lines, 50, 255, cv2.THRESH_BINARY)

    # Удаляем пересечение множеств, чтобы полностью отделить вертикальные линии от горизонтальных

    # Нахождение пересечения
    intersection = cv2.bitwise_and(vertical_thresh, horizontal_thresh)

    # Вычитание пересечения из каждого массива
    vertical_unique = cv2.bitwise_and(vertical_thresh, cv2.bitwise_not(intersection))
    horizontal_unique = cv2.bitwise_and(horizontal_thresh, cv2.bitwise_not(intersection))
    # (приятный бонус - мы сразу получаем инвертированные изображение, так как белый фон - тоже общий, и мы его отсекаем)

    # Подсчет белых пикселей
    white_vertical = cv2.countNonZero(vertical_unique)
    white_horizontal = cv2.countNonZero(horizontal_unique)

    # Определение ориентации
    if white_vertical < white_horizontal:
        return False
    else:
        return True

def get_horizontal_bounding_box(image):

    """
    Вспомогательная функция для определения ориентации
    Ограничивает весь документ рамкой и возвращает ее длину и ширину
    """

    # Применение адаптивной бинаризации
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Поиск контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Если контуры не найдены, возвращаем ошибку
    if not contours:
        print("Контуры не найдены. Возможно, документ пуст.")
        return

    # Объединяем все контуры в один
    all_contours = np.vstack(contours)

    # Вычисляем ограничивающую рамку
    x, y, w, h = cv2.boundingRect(all_contours)

    return (w, h)


def is_text_upside_down(image):

    """
    функция определяющая перевернут ли текст документа на 180 градусов

    принцип: текст разбивается на строки и каждая строка обрабатывается отдельно.
    в результате ориентация всего документа определяется по большинству
    """

    lines, lines_count = find_lines(image, detect_line_height(image))
    # lines, lines_count = find_lines(image)

    # print(f"lines count - {lines_count}")
    
    # Счетчик перевернутых строк
    upside_down_count = 0
    total_lines = 0
    
    # Анализируем строки, пока не будут обработаны все
    for line in lines:
        # Анализируем строку
        is_line_UD = is_line_upside_down(line)

        if is_line_UD == 1:
            # print('Line is upside-down')
            upside_down_count += 1
            total_lines += 1

        elif is_line_UD == 0: 
            # print('line is normal')
            total_lines += 1

        # show_image(line, name="line")

    # print(f"upside_down_count {upside_down_count} total_lines {total_lines}")
    # Если большинство строк перевернуто, считаем весь текст перевернутым
    if upside_down_count > total_lines / 2:
        return True  # Текст перевернут
    else:
        return False  # Текст в нормальной ориентации

# # # Загрузка изображения
# image = cv2.imread("test_data/original_5.png", cv2.IMREAD_GRAYSCALE)
# orientation = orientation_detect(image)

# print(f"Ориентация документа '{orientation[0]}'")
# print(f"Необходимо довернуть на угол {orientation[1]}° по часовой стрелке")

# detect_line_width(image)