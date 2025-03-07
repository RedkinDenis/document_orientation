import cv2
import numpy as np
from skimage import morphology

from image_processing import *
from find_lines import *    


def is_text_orientation_right(image):
    # Размытие для уменьшения шума
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Ядра для вертикальных и горизонтальных линий
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

    # Вычисляем bounding box с нулевым углом поворота
    x, y, w, h = cv2.boundingRect(all_contours)

    return (w, h)

def rotate_image_90(image):
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image

def orientation_detect(image):
    if not is_text_orientation_right(image):
        print('rotate')
        image = rotate_image_90(image)
    
    (w, h) = get_horizontal_bounding_box(image)

    if h > w:
        print("portrait")
    else:
        print("albumn")

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(is_text_upside_down(image))


def is_text_upside_down(image):

    lines_count = count_lines(image)
    print(f"lines count - {lines_count}")
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

    print(f"upside_down_count {upside_down_count} total_lines {total_lines}")
    # Если большинство строк перевернуто, считаем весь текст перевернутым
    if upside_down_count > total_lines / 2:
        return True  # Текст перевернут
    else:
        return False  # Текст в нормальной ориентации


def find_dots_and_commas(image):
    """
    Находит точки и запятые в строке.
    
    :param image: Изображение строки.
    :return: Список координат точек и запятых (y-координаты).
    """


    # Бинаризация изображения (текст белый, фон черный)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # image = cv2.medianBlur(image, 5)

    # kernel = np.ones((3, 3), np.uint8)
    # morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # cv2.imshow("Contours", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Находим контуры
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем копию изображения для отрисовки контуров
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Преобразуем в цветное изображение

    # Фильтруем контуры по размеру и рисуем их
    min_area = 10  # Минимальная площадь контура (исключает шумы)
    max_area = 40     # Максимальная площадь контура (исключает слишком большие объекты)

    dots_and_commas = []

    while (len(dots_and_commas) == 0 and max_area != 70):
        for contour in contours:
            # print(1)
            area = cv2.contourArea(contour)
            
            # Игнорируем слишком маленькие и слишком большие контуры
            if min_area < area < max_area:

                x, y, w, h = cv2.boundingRect(contour)
                simb = image[:, x:x+w]
                simb_cp = simb.copy()

                simb[y:y+h, :w] = 0
                # print(f'simb sum - {sum(sum(simb))}')
                # cv2.imshow("simb", output_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                #  and 
                # print(f"is_simb - {is_simb}")
                if (is_mostly_black_line(simb) and h // w <= 2 and h // w >= 1): # != - исключающее или. Знак препинания существует только в одной половине строки
                    # simb_low = simb_cp[0:y + h - h//2, :]
                    # simb_up = simb_cp[y + h//2:y + h, :]
                    # is_simb = (is_mostly_black_line(simb_up) != is_mostly_black_line(simb_low))
                    
                    # print(f"is_simb - {is_simb}")
                    # cv2.imshow("simb", simb_cp)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # if (is_simb):
                    dots_and_commas.append(y + h)  # Нижняя граница контура
                    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)  # Зеленый цвет, толщина линии 2
        max_area += 5

    # print(f"count of dots - {len(dots_and_commas)}")
    # cv2.imshow("Contours", output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return dots_and_commas

def is_line_upside_down(line):
    """
    Определяет, перевернут ли текст в строке.
    
    :param line: Изображение строки.
    :return: True, если текст перевернут, иначе False.
    """
    # Находим точки и запятые
    dots_and_commas = find_dots_and_commas(line)
    
    if len(dots_and_commas) == 0:
        # print("no dots and commas")
        return -1  # Эту строку не будем учитывать
    
    # Средняя позиция точек и запятых
    avg_position = np.mean(dots_and_commas)
    
    # Высота строки
    height = line.shape[0]
    
    # Если средняя позиция в верхней половине строки, текст перевернут
    if avg_position < height // 2:
        return 1 # строка перевернута
    else:
        return 0 # строка в нормальной ориентации


# Загрузка изображения
image = cv2.imread("test_data/rotated_90_5.png", cv2.IMREAD_GRAYSCALE)
orientation_detect(image)