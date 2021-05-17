def yanghuisanjiao(n):
    print([1])
    line = [1, 1]
    print(line)
    for i in range(2, n):
        l = []
        for j in range(0, len(line) - 1):
            l.append(line[j] + line[j + 1])  # 除去首尾中间的数字
        line = [1] + l + [1]  # 列表的拼接， 加上首尾完整的一行
        print(line)


if __name__ == "__main__":
    yanghuisanjiao(5)

