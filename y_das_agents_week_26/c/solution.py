if __name__ == '__main__':
    target, platforms_num = tuple(map(int, input().split()))
    platforms = set(map(int, input().split()))

    prev_1, prev_2 = ((0, '')), ((0, ''))
    for platform in range(1, target+1):
        if platform in platforms or platform == target:
            if prev_1 < prev_2:
                current = (prev_1[0] + 1, prev_1[1] + '2')
            else:
                current = (prev_2[0] + 1, prev_2[1] + '1')
        else:
            current = (float('inf'), '')

        prev_1, prev_2 = prev_2, current

    if current[0] == float('inf'):
        print(-1)
    else:
        print(current[0])
        print(current[1])
