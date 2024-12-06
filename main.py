import os


def main():
    while True:
        print("\n=== Меню завдань ===")
        print("1. Проекція точок (камера-стеноп)")
        print("2. Симуляція даних для калібрування")
        print("3. Калібрування камери (calibrateCamera)")
        print("4. Додавання другої камери")
        print("5. Калібрування другої камери")
        print("6. Стерео калібрування камер")
        print("7. Розрахунок фундаментальної та істотної матриць")
        print("8. Перевірка умов фундаментальної матриці")
        print("9. Розрахунок матриць гомографії")
        print("10. Ректифікація стереозображень")
        print("0. Вихід")

        choice = input("Оберіть завдання (0-10): ").strip()
        if choice == "0":
            print("Вихід із програми.")
            break
        try:
            task_file = f"task{choice}.py"
            if os.path.exists(task_file):
                os.system(f"python {task_file}")
            else:
                print("Такого завдання не існує.")
        except Exception as e:
            print(f"Помилка під час виконання: {e}")


if __name__ == "__main__":
    main()
