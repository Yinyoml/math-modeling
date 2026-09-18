# -*- coding: utf-8 -*-
"""
简短数模示例：两个经典模型
1. 线性规划 —— 生产计划问题（scipy.optimize.linprog）
2. 最小二乘拟合 —— 数据拟合模型（numpy.polyfit）
"""
import numpy as np
from scipy.optimize import linprog, curve_fit

print("=" * 50)
print("【1】线性规划：生产计划问题")
print("=" * 50)
# 某厂生产 A、B 两种产品，求最大利润：
#   利润：A 每件 40 元，B 每件 30 元
#   约束：原料 2A + B <= 100；工时 A + 2B <= 80；A <= 40
# 目标 max z = 40x1 + 30x2  ->  linprog 求 min，故取负
c = [-40, -30]
A_ub = [[2, 1], [1, 2], [1, 0]]
b_ub = [100, 80, 40]
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
x1, x2 = res.x
print(f"最优产量：A = {x1:.2f} 件, B = {x2:.2f} 件")
print(f"最大利润：{-res.fun:.2f} 元")

print()
print("=" * 50)
print("【2】最小二乘拟合：人口增长模型")
print("=" * 50)
# 已知若干年份人口数据，拟合 y = a * x + b（线性模型）
year = np.array([2015, 2016, 2017, 2018, 2019, 2020])
pop = np.array([13.7, 13.8, 13.9, 14.0, 14.1, 14.2])  # 亿人
k, b = np.polyfit(year, pop, 1)                       # 一次多项式拟合
pred = k * 2025 + b
print(f"拟合模型：y = {k:.4f} * year + {b:.4f}")
print(f"预测 2025 年人口：{pred:.2f} 亿人")
