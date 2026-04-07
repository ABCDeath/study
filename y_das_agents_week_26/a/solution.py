if __name__ == '__main__':
    problems_num, queries_num = tuple(map(int, input().split(' ')))
    problem_values = list(map(int, input().split(' ')))

    queries = [None] * queries_num
    for i in range(queries_num):
        queries[i] = tuple(map(int, input().split(' ')))

    open_idx, close_idx = [0] * problems_num, [0] * problems_num
    for start, stop in queries:
        open_idx[start-1] += 1
        close_idx[stop-1] += 1

    count = [0] * problems_num
    prev_opened = 0
    for i in range(problems_num):
        open_idx[i] += prev_opened
        count[i] = open_idx[i]
        prev_opened = open_idx[i] - close_idx[i]

    problem_values.sort(reverse=True)
    count.sort(reverse=True)

    print(sum(v * c for v, c in zip(problem_values, count)))
