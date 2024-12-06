import numpy as np
import cv2
import matplotlib.pyplot as plt

# Параметри першої камери
camera_matrix1 = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
dist_coeffs1 = np.zeros(5, dtype=np.float64)

# Параметри другої камери
camera_matrix2 = np.array([[750, 0, 300], [0, 750, 200], [0, 0, 1]], dtype=np.float64)
dist_coeffs2 = np.zeros(5, dtype=np.float64)

# Розташування другої камери відносно першої
R = cv2.Rodrigues(np.array([0.05, -0.1, 0.15], dtype=np.float64))[0]  # Обертання
T = np.array([[0.2, 0.1, 1.0]], dtype=np.float64).T  # Зміщення

# Генерація точок на площині
rows, cols = 6, 9
square_size = 0.04
object_points = np.zeros((rows * cols, 3), np.float64)
object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

# Визначення нового розміру зображення (збільшуємо масштаб)
image_size = (1280, 960)  # Збільшено вдвічі для більшого зображення

# Функція для проектування точок
def project_points(points_3D, camera_matrix, dist_coeffs, R, T):
    return cv2.projectPoints(points_3D, R, T, camera_matrix, dist_coeffs)[0].reshape(-1, 2)

# Проекції для обох камер
image_points1 = project_points(object_points, camera_matrix1, dist_coeffs1, np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
image_points2 = project_points(object_points, camera_matrix2, dist_coeffs2, R, T)

# Перевірка мінімальних і максимальних точок для другої камери
print("Min point (camera 2):", image_points2.min(axis=0))
print("Max point (camera 2):", image_points2.max(axis=0))

# Функція для візуалізації
def visualize_projections(image_points, title, image_size):
    img = np.zeros(image_size, np.uint8)
    for point in image_points:
        cv2.circle(img, tuple(point.astype(int)), 10, (255, 255, 255), -1)  # Збільшено радіус точок
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Відображення проекцій
plt.figure(figsize=(20, 10))  # Ширше вікно для великих зображень
plt.subplot(1, 2, 1)
visualize_projections(image_points1, "Проекції на камеру 1", image_size)

plt.subplot(1, 2, 2)
visualize_projections(image_points2, "Проекції на камеру 2", image_size)
plt.show()
