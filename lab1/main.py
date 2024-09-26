import matplotlib.pyplot as plt
import numpy as np
from sympy import solve, symbols

def f(x):
    return 0.5 * np.dot(np.dot(x.T, A), x) + np.dot(b, x)

def f_d(x):
    return 0.5 * np.dot((A.T + A), x) + b

lam = 10**-4 # λ
eps = 10**-6 # ε

tempA = np.array(
    [
        [ 11,17,19,13,17,19 ],
        [ 17,12,20,11,16,12 ],
        [ 11,18,14,10,15,13 ],
        [ 12,15,12,18,20,19 ],
        [ 20,12,11,20,13,14 ],
        [ 17,14,11,18,17,14 ],
    ]
)

A : np.ndarray = np.dot(tempA.T , tempA)

b : np.ndarray = np.array([15,14,16,13,20,18]).T
x_0 :np.ndarray = np.array([13,19,12,16,10,20]).T


# Собственные зачтения
eigen_values = np.linalg.eig(A)[0]

# Проверяем что матрица не вырожденная
tempA_det = np.linalg.det(tempA)


print(f"Матрица A:" , *tempA.tolist() , sep='\n\t', end='\n\n')
print(f"Собственные значения матрицы A:", *eigen_values.tolist(), sep='\n\t',  end='\n\n')
print(f"Вектор b:\n\t", "(", *b.tolist(), ")", sep=' ', end='\n\n' )
print(f"Вектор x0: \n\t" , "(", *x_0.tolist(), ")", sep=' ' , end='\n\n')


# Точное значение
x = symbols("x:6")
solution = list(solve(f_d(x),  x).values())

print()
print('---'*10)
print()

print(f"Точное значение x:\n\t", '(', *solution , ')' , sep=' ', end='\n\n')

x_ast = np.array(solution)
print(f"Точное значение f(x):\n\t", f(x_ast), end='\n\n')


# Градиент
x_k = x_0
x_k_1 = x_k - lam * f_d(x_k)

diff = np.linalg.norm(x_k_1 - x_k)

x_k_values = [x_k]
f_values = [f(x_k)]

while diff >= eps:
    x_k = x_k_1
    x_k_1 = x_k - lam * f_d(x_k)

    diff = np.linalg.norm(x_k_1 - x_k)

    x_k_values.append(x_k)
    f_values.append(f(x_k))

steps = len(x_k_values)

tempResult =  [
    (x_k_values[steps // 4], f(x_k_values[steps // 4 ])),
    (x_k_values[steps // 2], f(x_k_values[steps // 2 ])),
    (x_k_values[steps * 3 // 4], f(x_k_values[steps * 3 // 4 ])),
    (x_k_values[steps - 1], f(x_k_values[steps - 1 ])),
]

print()
print('---'*10)
print()


print('Кол-во шагов:\n\t', steps , end='\n\n')

print('Результаты и значения на 1/4, 1/2, 3/4 и последнем шаге:')

for i, (x, fun) in enumerate(tempResult):
    print(f'\tx = {x.tolist()} , | f(x) = {fun}')

print()
print('---'*10)
print()

x_a = -np.dot(np.linalg.inv(A), b)

delta_x = [abs(x_k[i] - solution[i]) for i in range(len(x_k))]
delta_f = abs(f(x_k) -  f(x_a))

print("Разница | Xm - Xточ |:")

for i in range(len(x_k)):
    print(f"\t| Xm{i + 1} - Xточ{i + 1} | = " , delta_x[i])

print()

print("Разница | Fm - Fточ | = " , delta_f)
print()


# График

plt.plot(range(len(x_k_values)), f_values)
plt.show()