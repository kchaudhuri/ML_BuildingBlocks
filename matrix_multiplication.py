"""
Sparse Matrix Multiplication
"""

def sparse_matrix_multiplication(matrix_a, matrix_b):

    result = [[]]

    m_a = len(matrix_a)
    y_b = len(matrix_b[0])
    if len(matrix_a[0]) == len(matrix_b):
        nx_ab = len(matrix_b)
    else:
        return result

    result = [[0 for _ in range(y_b)] for _ in range(m_a)]
    for m in range(m_a):
        for y in range(y_b):
            for nx in range(nx_ab):
                result[m][y] += matrix_a[m][nx] * matrix_b[nx][y]

    return result

