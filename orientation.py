import cv2
import numpy as np

from image_processing import *
from lines_processing import *    


def orientation_detect(image):

    """
    это основная функция.
    return: возвращает кортеж (orientation, angle)
    orientation: ориентация документа (портретная или альбомная)
    angle: угол на который необходимо повернуть документ для нормализации
    """

    angle = 0
    orientation = ''

    # проверяется правильность ориентации букв.
    # на данном этапе случаи поворота на 0 и 180 и на 90 и 270 градусов между собой не различимы
    if not is_text_orientation_right(image):
        # print('rotate')
        image = rotate_image_90(image)
        angle += 90
    # получаем документ который однозначно повернут на 0 или 180 градусов
    
    (w, h) = get_horizontal_bounding_box(image)

    # у текста с портретной ориентацией высота ограничивающей рамки больше чем ее ширина (и наоборот)
    if h > w:
        orientation = "portrait"
    else:
        orientation = "albumn"

    # если текст все же перевернут, добавляем 180 к углу поворота
    if (is_text_upside_down(image)):
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

    if image is None:
        print("Ошибка: изображение не загружено.")
        return

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

    lines_count = count_lines(image)
    # print(f"lines count - {lines_count}")
    # Удаление белых краев
    image = remove_white_margins(image)

    image = cv2.bitwise_not(image)
    # Копия изображения для закрашивания
    working_image = image.copy()

    # working_image = find_and_ignore_first_lines(working_image, num_lines_to_ignore=2, region_height=60) # comm
    
    # Счетчик перевернутых строк
    upside_down_count = 0
    total_lines = 0
    
    # Анализируем строки, пока не будут обработаны все
    for i in range(lines_count):
        # Находим участок с наибольшей плотностью (одну строку)
        y_start, y_end = find_most_dense_region(working_image, region_height=60)
        
        # Если плотность слишком мала, завершаем цикл
        if y_end - y_start < 10:
            break
        
        # Вырезаем строку
        line = working_image[y_start:y_end, :]

        # Анализируем строку
        is_line_UD = is_line_upside_down(line)

        if is_line_UD == 1:
            # print('Line is upside-down')
            upside_down_count += 1
            total_lines += 1

        elif is_line_UD == 0:
            # print('line is normal')
            total_lines += 1

        # cv2.imshow("image", line)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Закрашиваем обработанный участок
        working_image[y_start:y_end, :] = 0

    # print(f"upside_down_count {upside_down_count} total_lines {total_lines}")
    # Если большинство строк перевернуто, считаем весь текст перевернутым
    if upside_down_count > total_lines / 2:
        return True  # Текст перевернут
    else:
        return False  # Текст в нормальной ориентации

# # Загрузка изображения
# image = cv2.imread("test_data/rotated_270_1.png", cv2.IMREAD_GRAYSCALE)
# orientation = orientation_detect(image)

# print(f"Ориентация документа '{orientation[0]}'")
# print(f"Необходимо довернуть на угол {orientation[1]}° по часовой стрелке")
