import cv2
import numpy as np

# Функція для генерації 3D точок (шахова дошка)
def generate_3d_points(grid_size=(6, 8), square_size=1.0):
    points = []
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            points.append((j * square_size, i * square_size, 0))  # Z=0
    return np.array(points, dtype=np.float32)

# Функція для симуляції проектування 3D точок
def project_points(points_3d, camera_matrix, rvec, tvec):
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None)
    return points_2d.reshape(-1, 2)

# Основна функція
def main():
    # Початкові параметри камери
    image_size = (640, 480)  # Розмір зображення
    focal_length = 800  # Фокусна відстань
    principal_point = (image_size[0] / 2, image_size[1] / 2)  # Головна точка
    true_camera_matrix = np.array([
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    true_dist_coeffs = np.zeros(5, dtype=np.float32)  # Без дисторсії

    # Генерація 3D точок
    grid_size = (6, 8)  # 6x8 сітка
    square_size = 30  # Розмір квадрата
    points_3d = generate_3d_points(grid_size, square_size)

    # Симуляція кількох кадрів
    num_frames = 5
    object_points = []  # 3D точки для кожного кадру
    image_points = []  # 2D проекції для кожного кадру
    for _ in range(num_frames):
        # Генерація випадкового положення камери
        rvec = np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)
        tvec = np.random.uniform(-200, 200, size=3).astype(np.float32)

        # Проекція точок
        projected_points = project_points(points_3d, true_camera_matrix, rvec, tvec)

        # Збереження точок
        object_points.append(points_3d)
        image_points.append(projected_points)

    # Калібрування камери
    initial_camera_matrix = np.eye(3, dtype=np.float32)  # Початкове наближення
    initial_dist_coeffs = np.zeros(5, dtype=np.float32)
    retval, calibrated_camera_matrix, calibrated_dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=image_size,
        cameraMatrix=initial_camera_matrix,
        distCoeffs=initial_dist_coeffs,
        flags=cv2.CALIB_FIX_PRINCIPAL_POINT,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    # Вивід результатів
    print("=== Результати калібрування ===")
    print("Середня помилка калібрування:", retval)
    print("Матриця камери (калiбрована):\n", calibrated_camera_matrix)
    print("Матриця камери (істинна):\n", true_camera_matrix)
    print("Коефіцієнти дисторсії (калiбровані):\n", calibrated_dist_coeffs)
    print("Коефіцієнти дисторсії (істинні):\n", true_dist_coeffs)
