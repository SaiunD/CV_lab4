import numpy as np
import cv2


# Функція для генерації тестових точок
def generate_synthetic_points(num_points=50, noise=0.1):
    """Генерує набір 3D точок та проєкції на дві камери з шумом."""
    # Генеруємо випадкові 3D-точки
    object_points = np.random.uniform(-1, 1, (num_points, 3)).astype(np.float32)

    # Генеруємо внутрішні параметри камер
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    distortion_coeffs = np.zeros(5, dtype=np.float32)  # Без дисторсії

    # Параметри другої камери (поворот та зсув)
    R = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0]  # Поворот (як вектор)
    T = np.array([[0.5, -0.2, 0.3]], dtype=np.float32).T  # Зміщення

    # Проєкція точок на обидві камери
    image_points1, _ = cv2.projectPoints(object_points, np.zeros(3), np.zeros(3), camera_matrix, distortion_coeffs)
    image_points2, _ = cv2.projectPoints(object_points, R, T, camera_matrix, distortion_coeffs)

    # Додавання шуму
    image_points1 += np.random.normal(0, noise, image_points1.shape)
    image_points2 += np.random.normal(0, noise, image_points2.shape)

    return object_points, image_points1, image_points2, camera_matrix, distortion_coeffs


# Генерація тестових даних
num_points = 100  # Збільшуємо кількість точок для покращення результатів
object_points, image_points1, image_points2, camera_matrix, dist_coeffs = generate_synthetic_points(
    num_points=num_points)

# Перетворення у формат для OpenCV
object_points = [object_points]
image_points1 = [image_points1]
image_points2 = [image_points2]

# Визначення початкових параметрів
initial_camera_matrix1 = camera_matrix.copy()
initial_camera_matrix2 = camera_matrix.copy()
flags = cv2.CALIB_FIX_INTRINSIC  # Фіксуємо внутрішні параметри

# Стерео калібрування
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
ret_stereo, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = cv2.stereoCalibrate(
    object_points,
    image_points1,
    image_points2,
    initial_camera_matrix1,
    dist_coeffs,
    initial_camera_matrix2,
    dist_coeffs,
    (640, 480),  # Розмір зображення
    criteria=criteria,
    flags=flags
)

# Вивід результатів
print("Середньоквадратична помилка репроєкції:", ret_stereo)
print("\nМатриця обертання (R):\n", R)
print("\nВектор трансляції (T):\n", T)

# Оцінка різниці з початковими значеннями
initial_R = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0]
initial_T = np.array([[0.5, -0.2, 0.3]], dtype=np.float32).T

print("\nПорівняння з початковими параметрами:")
print("Початкове обертання (R):\n", initial_R)
print("Початкове зміщення (T):\n", initial_T)

print("\nРізниця у матриці обертання (R):\n", R - initial_R)
print("\nРізниця у векторі трансляції (T):\n", T - initial_T)
