import cv2
import numpy as np

# Функція для проекції 3D-точок на 2D площину
def project_points(object_points, camera_matrix, dist_coeffs, R, T):
    # Проекція 3D-точок на 2D за допомогою заданих параметрів камери
    image_points, _ = cv2.projectPoints(object_points, R, T, camera_matrix, dist_coeffs)
    return image_points.reshape(-1, 2)  # Перетворення в одномірний масив 2D-точок

# Параметри першої камери
camera_matrix1 = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs1 = np.zeros(5)

# Параметри другої камери
camera_matrix2 = np.array([[750, 0, 300], [0, 750, 200], [0, 0, 1]], dtype=np.float32)
dist_coeffs2 = np.zeros(5)

# Розташування другої камери відносно першої
R = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0]  # Обертання
T = np.array([[0.5, -0.2, 0.3]]).T  # Зміщення

# Генерація точок на площині
rows, cols = 6, 9
square_size = 0.04
object_points = np.zeros((rows * cols, 3), np.float32)
object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

# Список 2D точок для калібрування
image_points2 = []  # Тут ми зберігатимемо зображення точок

# Проекція точок на зображення для камери 2
image_points2 = project_points(object_points, camera_matrix2, dist_coeffs2, R, T)

# Створення списку точок для калібрування
object_points_all = [object_points]  # Список 3D точок
image_points_all = [image_points2]  # Список 2D точок на зображенні

# Розмір зображення
image_size = (640, 480)

# Калібрування камери
ret, camera_matrix2_calibrated, dist_coeffs2_calibrated, rvecs, tvecs = cv2.calibrateCamera(
    object_points_all, image_points_all, image_size, None, None
)

# Виведення результатів калібрування
print("Результати калібрування камери 2:")
print(f"Калібрована матриця камери:\n{camera_matrix2_calibrated}")
print(f"Калібровані коефіцієнти дисторсії:\n{dist_coeffs2_calibrated}")

# Порівняння з початковими параметрами камери 2
print("\nПорівняння з початковими параметрами:")
print(f"Початкова матриця камери 2:\n{camera_matrix2}")
print(f"Початкові коефіцієнти дисторсії камери 2:\n{dist_coeffs2}")

# Різниця між каліброваними і початковими параметрами
camera_matrix_diff = camera_matrix2_calibrated - camera_matrix2
dist_coeffs_diff = dist_coeffs2_calibrated - dist_coeffs2

print("\nРізниця між каліброваними і початковими параметрами:")
print(f"Різниця у матриці камери:\n{camera_matrix_diff}")
print(f"Різниця у коефіцієнтах дисторсії:\n{dist_coeffs_diff}")
