import numpy as np
import matplotlib.pyplot as plt

# Приклад матриці фундаментальної матриці F
F = np.array([[9.18296606e-05, 6.41392061e-05, -4.34614773e-01],
              [5.96751606e-05, 4.40291456e-05, -2.89649766e-01],
              [-2.25675170e-04, -1.59168159e-04, 1.08570672e+00]])

# Приклад проекцій на камери
points_L = np.array([[320, 240],
                     [480, 400],
                     [160, 400],
                     [480, 80]])

points_R = np.array([[375, 200],
                     [525, 350],
                     [225, 350],
                     [525, 50]])

# Ректифікація точок
points_L_rectified = np.array([[-0.39660938, -0.29523341],
                               [-0.39679947, -0.29496384],
                               [-0.39940871, -0.29126446],
                               [-0.39500646, -0.29750607]])

points_R_rectified = np.array([[-0.39607435, -0.29599199],
                               [-0.39641082, -0.29551489],
                               [-0.39816034, -0.29303443],
                               [-0.39476971, -0.29784173]])


# Функція для перевірки епіполярної умови
def check_epipolar_condition(L, R, F):
    for l, r in zip(L, R):
        # Додаємо координату 1 до кожної точки для отримання 3D координат
        l_homogeneous = np.append(l, 1)  # Ліва точка (додаємо 1)
        r_homogeneous = np.append(r, 1)  # Права точка (додаємо 1)

        print(f"Перевірка для точки: {l_homogeneous}, {r_homogeneous}")

        # Перевірка епіполярної умови: p_L^T F p_R
        result = np.dot(l_homogeneous, np.dot(F, r_homogeneous.T))  # Перевірка епіполярної умови
        print(f"p_L^T F p_R = {result}")


# Візуалізація
def visualize(points_L, points_R, points_L_rectified, points_R_rectified):
    plt.figure(figsize=(12, 6))

    # Оригінальні зображення
    plt.subplot(1, 2, 1)
    plt.scatter(points_L[:, 0], points_L[:, 1], c='blue', label='Точки на камері 1 (оригінальні)')
    plt.scatter(points_R[:, 0], points_R[:, 1], c='red', label='Точки на камері 2 (оригінальні)')
    plt.title('Оригінальні зображення')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Ректифіковані зображення
    plt.subplot(1, 2, 2)
    plt.scatter(points_L_rectified[:, 0], points_L_rectified[:, 1], c='blue', label='Точки на камері 1 (ректифіковані)')
    plt.scatter(points_R_rectified[:, 0], points_R_rectified[:, 1], c='red', label='Точки на камері 2 (ректифіковані)')
    plt.title('Ректифіковані зображення')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Перевірка епіполярної умови для ректифікованих точок
check_epipolar_condition(points_L_rectified, points_R_rectified, F)

# Візуалізація результатів
visualize(points_L, points_R, points_L_rectified, points_R_rectified)
