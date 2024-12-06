import numpy as np
import cv2
import matplotlib.pyplot as plt


def project_points(world_points, K, R, t):
    """
    Проектує 3D точки на 2D з використанням моделі камери-стенопа.
    """
    # Конвертація world_points у вигляд гомогенних координат
    world_points_h = np.hstack((world_points, np.ones((world_points.shape[0], 1))))

    # Формуємо матрицю зовнішніх параметрів [R | t]
    Rt = np.hstack((R, t.reshape(-1, 1)))

    # Проєкція точок
    camera_matrix = K @ Rt
    image_points_h = camera_matrix @ world_points_h.T

    # Переходимо до декартових координат
    image_points = image_points_h[:2, :] / image_points_h[2, :]
    return image_points.T


def epipolar_condition(pL, pR, F):
    """
    Перевірка епіполярної умови: pR.T * F * pL = 0
    pL та pR - це координати точок на лівій та правій камерах
    F - фундаментальна матриця
    """
    # Додаємо одиничні координати для точок (перехід до гомогенних координат)
    pL_h = np.append(pL, 1)
    pR_h = np.append(pR, 1)

    # Перевіряємо епіполярну умову
    return np.isclose(np.dot(pR_h.T, np.dot(F, pL_h)), 0)


def main():
    # Внутрішні параметри камери
    fx, fy = 800, 800  # Фокусна відстань
    cx, cy = 640, 360  # Координати головної точки
    K1 = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0, 0, 1]])  # Камера 1

    K2 = np.array([[fx, 0, cx - 100],  # Камера 2 (зсув по x)
                   [0, fy, cy],
                   [0, 0, 1]])  # Камера 2

    # Зовнішні параметри камери
    R = np.eye(3)  # Ніякого обертання
    t = np.array([0, 0, -5])  # Перенесення для камери 1

    # Набір точок у світовій системі координат
    world_points = np.array([[1, 1, 0],
                             [2, 2, 0],
                             [3, 1, 0],
                             [4, 4, 0]])

    # Проєкція точок на обидві камери
    image_points_L = project_points(world_points, K1, R, t)  # Ліва камера
    image_points_R = project_points(world_points, K2, R, t)  # Права камера

    # Визначення фундаментальної матриці (F)
    # Приклад - можна використовувати F з попереднього кроку
    F = np.array([[-0.01601045, -0.01222168, 0.03970097],
                  [-0.01044193, -0.00797334, 0.02588434],
                  [0.040236, 0.03071371, -0.09977529]])

    # Перевірка епіполярної умови для кожної пари точок
    for i in range(len(image_points_L)):
        pL = image_points_L[i]
        pR = image_points_R[i]
        if not epipolar_condition(pL, pR, F):
            print(f"Епіполярна умова виконується для лівої точки: {pL} та правої точки: {pR}")
        else:
            print(f"Епіполярна умова не виконується для лівої точки: {pL} та правої точки: {pR}")

    # Візуалізація результатів
    plt.figure()
    plt.scatter(image_points_L[:, 0], image_points_L[:, 1], color='red', label='Left Camera Points')
    plt.scatter(image_points_R[:, 0], image_points_R[:, 1], color='blue', label='Right Camera Points')
    plt.title('Projection of 3D points onto 2D image planes')
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')
    plt.gca().invert_yaxis()  # Інвертуємо вісь Y для відповідності системі координат зображення
    plt.grid(True)
    plt.legend()
    plt.show()

    # Збереження результату
    output_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for point in image_points_L.astype(int):
        cv2.circle(output_image, tuple(point), 5, (0, 0, 255), -1)  # Ліва камера
    for point in image_points_R.astype(int):
        cv2.circle(output_image, tuple(point), 5, (0, 255, 0), -1)  # Права камера
    cv2.imwrite("output_projection_with_epipolar_check.png", output_image)
