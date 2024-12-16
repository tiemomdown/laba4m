import math

def converge(xk, xkp, n, eps):
    norm = sum((xk[i] - xkp[i]) ** 2 for i in range(n))
    return math.sqrt(norm) < eps
def calculate_residual_vector(a, b, x, n):
    residual = [0.0] * n
    for i in range(n):
        ax = sum(a[i][j] * x[j] for j in range(n))
        residual[i] = b[i] - ax
    return residual

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

def to_diagonal_dominance_gauss_jordan(a, b, n):
    for i in range(n):
        if abs(a[i][i]) == 0:
            for k in range(i + 1, n):
                if abs(a[k][i]) > abs(a[i][i]):
                    a[i], a[k] = a[k], a[i]
                    b[i], b[k] = b[k], b[i]
                    break
        for j in range(n):
            if i != j:
                factor = a[j][i] / a[i][i]
                for k in range(n):
                    a[j][k] -= factor * a[i][k]
                b[j] -= factor * b[i]
    for i in range(n):
        factor = a[i][i]
        for j in range(n):
            a[i][j] /= factor
        b[i] /= factor
    print("Матрица приведена к диагональному преобладанию методом Гаусса-Жордана.")
    for row, bi in zip(a, b):
        print(row, "|", bi)
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
    free_variables = []
    for i in range(n):
        if abs(a[i][i]) < 1e-14:
            for k in range(i + 1, n):
                if abs(a[k][i]) > abs(a[i][i]):
                    a[i], a[k] = a[k], a[i]
                    b[i], b[k] = b[k], b[i]
                    break

        for k in range(i + 1, n):
            factor = a[k][i] / a[i][i]
            for j in range(i, n):
                a[k][j] -= factor * a[i][j]
            b[k] -= factor * b[i]
    print("Треугольная матрица: ")
    for i in a:
        for j in i:
            if j < 1e-14:
                j = 0
            print(j, end=' ')
        print()
    not_comparable = True
    for i in range(n):
        if a[i][i] == 0:
            not_comparable = False
            break
    if (not not_comparable):
        inconsistent = any(all(abs(a[i][j]) < 1e-9 for j in range(n)) and abs(b[i]) > 1e-9 for i in range(n))
        if inconsistent:
            print("Система несовместна (нет решений).")
            return
        else:
            print("Система имеет бесконечно много решений.")
            print("Пример одного из решений:")
            for i in range(n):
                x[i] = 0 if abs(a[i][i]) < 1e-9 else b[i] / a[i][i]
            for i in range(n):
                print(f"x{i + 1} = {x[i]}")
            return
    else:
        for i in range(n - 1, -1, -1):
            x[i] = b[i]
            for j in range(i + 1, n):
                x[i] -= a[i][j] * x[j]
            x[i] /= a[i][i]

        print("\nРешение методом Гаусса:")
        for i in range(n):
            print(f"x{i + 1} = {x[i]}")
        residual = calculate_residual_vector(a, b, x, n)
        print("\nВектор невязки:")
        for i in range(n):
            print(f"r{i + 1} = {residual[i]}")

def seidel_method(a, b, n, eps):
    x = [0.0] * n
    prev_x = [0.0] * n
    if not has_diagonal_dominance(a, n):
        print("Диагональное преобладание не выполнено. Пробуем привести матрицу к диагональному преобладанию.")
        to_diagonal_dominance_gauss_jordan(a, b, n)
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
    residual = calculate_residual_vector(a, b, x, n)
    print("\nВектор невязки:")
    for i in range(n):
        print(f"r{i + 1} = {residual[i]}")
def main():
    n = 3  # Размерность матрицы
    eps = 0.001  # Точность
    # Матрица и вектор из задания
    # a = [
    #     [9, 8, 7],
    #     [6, 5, 4],
    #     [3, 2, 1]
    # ]
    # b = [8, 5, 2]
    a = [
        [3.1, 2.8, 1.9],
        [1.9, 3.1, 2.1],
        [7.5, 3.8, 4.8]
    ]

    b = [0.2, 2.1, 5.6]

    print("1.",end ='')
    gauss_method(a, b, n)
    print("2.", end = '')
    cramer_method(a, b, n)
    print("3. Метод Зейделя")
    seidel_method(a, b, n, eps)

if __name__ == "__main__":
    main()
#Вывод ступенчатой матрицы, вектор невязки,
'''
Решение методом Зейделя:
x1 = -1.236
x2 = -1.431
x3 = 4.231
Количество итераций: 2
2. Метод Гаусса

Решение методом Гаусса:
x1 = -1.235969253448457
x2 = -1.4310835000526474
x3 = 4.230809729388226
3. Метод Крамера

Решение методом Крамера:
x1 = -1.235969253448457
x2 = -1.4310835000526474
x3 = 4.230809729388226
'''
