def check_connectitvity(n_node, edges):
    f = [i for i in range(n_node)]

    def find(x):
        if f[x] == x:
            return x
        else:
            return find(f[x])

    def union(x, y):
        f[find(x)] = find(y)

    for st, ed in edges:
        union(st, ed)
    for i in range(1, n_node):
        if f[i] != f[0]:
            return False
    return True


def dijkstra(adj_mat, src, tgt):
    n_nodes = len(adj_mat)
    dist = {i: adj_mat[src][i] for i in range(n_nodes)}
    visit, prev = {src}, [None if adj_mat[src][i] >= 0xffff else src for i in range(n_nodes)]
    while len(visit) < n_nodes:
        min_d, sel_node = 0xff, -1
        for node in range(n_nodes):
            if node not in visit and dist[node] < min_d:
                min_d, sel_node = dist[node], node
        if sel_node == -1:
            # print('Warning: Unconnect graph')
            break
        visit.add(sel_node)
        for node in range(n_nodes):
            if dist[node] > dist[sel_node] + adj_mat[sel_node][node]:
                dist[node] = dist[sel_node] + adj_mat[sel_node][node]
                prev[node] = sel_node
        if sel_node == tgt:
            break
    edges = [(prev[tgt], tgt)]
    cur_prev, cur = edges[-1]
    while cur_prev != src:
        cur_prev, cur = prev[cur_prev], cur_prev
        edges.append((cur_prev, cur))
    return edges[::-1]


def steiner_tree(adj_mat, sub_nodes):
    inf = 0xffff
    n_node = len(adj_mat)
    n_sub_node = len(sub_nodes)
    n_space = 1 << n_sub_node

    f = [[inf for _ in range(n_node)] for _ in range(n_space)]
    record = [[[] for _ in range(n_node)] for _ in range(n_space)]
    for i in range(n_sub_node):
        f[1 << i][sub_nodes[i]] = 0

    queue = []

    def spfa(s):
        while queue:
            cur = queue.pop(0)
            for ed in range(n_node):
                if adj_mat[cur][ed] != inf and f[s][ed] > f[s][cur] + adj_mat[cur][ed]:
                    f[s][ed] = f[s][cur] + adj_mat[cur][ed]
                    record[s][ed] = record[s][cur] + [(cur, ed)]
                    if ed not in queue:
                        queue.append(ed)

    for s in range(1, n_space):
        for i in range(n_node):
            ss = s & (s - 1)
            while ss:
                if f[s][i] > f[ss][i] + f[s ^ ss][i]:
                    f[s][i] = f[ss][i] + f[s ^ ss][i]  # combine ss with s ^ ss w.r.t root i
                    record[s][i] = record[ss][i] + record[s ^ ss][i]
                ss = s & (ss - 1)
            if f[s][i] != inf and s not in queue:
                queue.append(i)
        spfa(s)

    min_edge, min_record = inf, None
    for i in range(n_node):
        if f[n_space - 1][i] < min_edge:
            min_edge = f[n_space - 1][i]
            min_record = record[n_space - 1][i]

    if min_record is None:
        raise Exception('No steiner tree found')
    return min_record


if __name__ == '__main__':
    edges = [(0, 4), (1, 4), (1, 2), (2, 5), (5, 3), (5, 6), (1, 7), (7, 8)]
    adj_mat = [[0xffff for _ in range(9)] for _ in range(9)]
    for x, y in edges:
        adj_mat[x][y] = adj_mat[y][x] = 1
    tree = steiner_tree(adj_mat, [1, 7, 0])
    print(tree)