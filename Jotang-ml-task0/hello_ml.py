import numpy as np

# 用列表 + 字典记录姓名和成绩
students = [
    {"name": "张三", "score": 88},
    {"name": "李四", "score": 92},
    {"name": "王五", "score": 79},
    {"name": "赵六", "score": 95},
]


def analyze_scores(student_list):
    """计算平均分，并找出最高分及对应学生"""
    scores = [stu["score"] for stu in student_list]

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    top_students = [stu["name"] for stu in student_list if stu["score"] == max_score]

    return avg_score, max_score, top_students


if __name__ == "__main__":
    # 1. 成绩分析
    avg, max_score, top_students = analyze_scores(students)

    print("=== 成绩分析 ===")
    print(f"平均分：{avg:.2f}")
    print(f"最高分：{max_score}")
    print(f"最高分学生：{', '.join(top_students)}")

    # 2. NumPy 矩阵乘法
    # A 的形状：(2, 3)
    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    # B 的形状：(3, 2)
    B = np.array([
        [7, 8],
        [9, 10],
        [11, 12]
    ])

    # 矩阵乘法：C = A @ B，形状为 (2, 2)
    C = A @ B

    print("\n=== 矩阵乘法 ===")
    print("矩阵 A：")
    print(A)
    print("A 的形状：", A.shape)

    print("\n矩阵 B：")
    print(B)
    print("B 的形状：", B.shape)

    print("\n矩阵 C = A @ B：")
    print(C)
    print("C 的形状：", C.shape)