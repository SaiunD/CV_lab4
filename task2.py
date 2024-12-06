import cv2
import numpy as np
import matplotlib.pyplot as plt

# Функція для створення 3D точок на площині
def generate_3d_points(grid_size=(6, 8), square_size=1.0):
    points = []
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            points.append((j * square_size, i * square_size, 0))  # Z=0, лежать у площині XY
    return np.array(points, dtype=np.float32)

# Функція для симуляції проекції точок на зображення
def project_points(points_3d, camera_matrix, rvec, tvec):
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None)
    return points_2d.reshape(-1, 2)

# Основна функція
def main():
    # Параметри камери
    image_size = (640, 480)  # Розмір зображення
    focal_length = 800  # Фокусна відстань
    principal_point = (image_size[0] / 2, image_size[1] / 2)  # Головна точка
    camera_matrix = np.array([
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Генерація 3D точок
    grid_size = (6, 8)  # 6 стовпців і 8 рядків
    square_size = 30  # Розмір квадрата у міліметрах
    points_3d = generate_3d_points(grid_size, square_size)

    # Симуляція кадрів
    num_frames = 5
    all_projected_points = []
    for frame_idx in range(num_frames):
        # Генерація випадкового положення камери
        rvec = np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)
        tvec = np.random.uniform(-200, 200, size=3).astype(np.float32)

        # Проекція точок
        projected_points = project_points(points_3d, camera_matrix, rvec, tvec)

        # Фільтрація точок, що виходять за межі зображення
        valid_points = []
        for pt in projected_points:
            if 0 <= pt[0] < image_size[0] and 0 <= pt[1] < image_size[1]:
                valid_points.append(pt)
        valid_points = np.array(valid_points)

        all_projected_points.append(valid_points)

        # Візуалізація
        plt.figure()
        plt.scatter(valid_points[:, 0], valid_points[:, 1], color='blue')
        plt.title(f'Frame {frame_idx + 1} - Projected Points')
        plt.xlim(0, image_size[0])
        plt.ylim(image_size[1], 0)  # Зображення має верхнє походження
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.grid(True)
        plt.show()
