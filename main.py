import math

def converge(xk, xkp, n, eps):
    norm = sum((xk[i] - xkp[i]) ** 2 for i in range(n))
    return math.sqrt(norm) < eps

def round_to_eps(x, eps):
    precision = -math.log10(eps)
    factor = 10 ** precision
    return round(x * factor) / factor

def has_diagonal_dominance(a, n):
    for i in range(n):
        sum_off_diag = sum(abs(a[i][j]) for j in range(n) if i != j)
        if abs(a[i][i]) <= sum_off_diag:
            return False
    return True

def determinant(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(n):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(minor)
    return det

def cramer_method(a, b, n):
    det_a = determinant(a)
    if det_a == 0:
        print("Система не имеет уникального решения (определитель равен нулю).")
        return

    solutions = []
    for i in range(n):
        modified_matrix = [row[:] for row in a]
        for j in range(n):
            modified_matrix[j][i] = b[j]
        det_modified = determinant(modified_matrix)
        solutions.append(det_modified / det_a)

    print("\nРешение методом Крамера:")
    for i, value in enumerate(solutions):
        print(f"x{i + 1} = {value}")

def gauss_method(a, b, n):
    x = [0.0] * n

    # Forward elimination
    for i in range(n):
        for k in range(i + 1, n):
            factor = a[k][i] / a[i][i]
            for j in range(i, n):
                a[k][j] -= factor * a[i][j]
            b[k] -= factor * b[i]

    # Back substitution
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= a[i][j] * x[j]
        x[i] /= a[i][i]

    print("\nРешение методом Гаусса:")
    for i in range(n):
        print(f"x{i + 1} = {x[i]}")

def seidel_method(a, b, n, eps):
    x = [0.0] * n
    prev_x = [0.0] * n

    if not has_diagonal_dominance(a, n):
        print("Диагональное преобладание не выполнено. Метод Зейделя может не сработать.")
        return

    iterations = 0
    while True:
        prev_x[:] = x[:]

        for i in range(n):
            sum_ = sum(a[i][j] * x[j] for j in range(n) if i != j)
            x[i] = (b[i] - sum_) / a[i][i]

        iterations += 1
        if converge(x, prev_x, n, eps):
            break

    print("\nРешение методом Зейделя:")
    for i in range(n):
        print(f"x{i + 1} = {round_to_eps(x[i], eps)}")
    print(f"Количество итераций: {iterations}")

def main():
    n = 3  # Размерность матрицы
    eps = 0.001  # Точность

    # Матрица и вектор из задания
    a = [
        [2.7, 3.3, 3.1],
        [3.5, 1.7, 2.8],
        [4.1, 5.8, -1.7]
    ]

    b = [2.7, 1.7, 0.8]

    print("Выберите метод решения:")
    print("1. Метод Зейделя")
    print("2. Метод Гаусса")
    print("3. Метод Крамера")

    method = int(input("Введите номер метода: "))

    if method == 1:
        seidel_method(a, b, n, eps)
    elif method == 2:
        gauss_method(a, b, n)
    elif method == 3:
        cramer_method(a, b, n)
    else:
        print("Некорректный выбор метода.")

if __name__ == "__main__":
    main()


#Решение методом Гаусса:
#x1 = -0.17740670890635932
#x2 = 0.42980653090631693
#x3 = 0.5679472780826853

#Решение методом Крамера:
#x1 = -0.1774067089063594
#x2 = 0.42980653090631704
#x3 = 0.5679472780826852