a = [('Damage', (29, 30, 31, 32, 33, 34, 35)), ('Other', (10, 11, 12)), ('Another', (20, 21, 22, 23))]

b = [10, 11, 29, 30]  # 示例索引值列表

a = [elem for elem in a if not any(index in elem[1] for index in b)]

print(a)
