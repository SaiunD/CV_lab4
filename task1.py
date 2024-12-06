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


def main():
    # Внутрішні параметри камери
    fx, fy = 800, 800  # Фокусна відстань
    cx, cy = 640, 360  # Координати головної точки
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Зовнішні параметри камери
    R = np.eye(3)  # Ніякого обертання
    t = np.array([0, 0, -5])  # Перенесення

    # Набір точок у світовій системі координат
    world_points = np.array([[1, 1, 0],
                             [2, 2, 0],
                             [3, 1, 0],
                             [4, 4, 0]])

    # Проєкція точок
    image_points = project_points(world_points, K, R, t)

    # Візуалізація результатів
    plt.figure()
    plt.scatter(image_points[:, 0], image_points[:, 1], color='red', label='Image Points')
    plt.title('Projection of 3D points onto 2D image plane')
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')
    plt.gca().invert_yaxis()  # Інвертуємо вісь Y для відповідності системі координат зображення
    plt.grid(True)
    plt.legend()
    plt.show()

    # Збереження результату
    output_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for point in image_points.astype(int):
        cv2.circle(output_image, tuple(point), 5, (0, 0, 255), -1)
    cv2.imwrite("output_projection.png", output_image)

