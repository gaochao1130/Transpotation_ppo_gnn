import numpy as np

INF = 9999.0  # 不可达标记

def load_distance_matrix(filepath, delimiter=",", inf_value=INF):
    """加载距离矩阵，将 inf_value 转换为 np.inf"""
    dist = np.loadtxt(filepath, delimiter=delimiter)
    dist[dist >= inf_value - 1e-6] = np.inf
    return dist

def simplify_graph(dist_matrix, epsilon=1e-6, keep_nearest=True):
    """
    基于三角不等式移除冗余边，处理不可达距离（np.inf）。
    参数:
        dist_matrix : numpy.ndarray, shape (N,N)
        epsilon     : 浮点误差容忍度
        keep_nearest: bool, 是否强制保留每个节点的最近可达邻居
    返回:
        edges : list of tuple (i, j, distance)
    """
    N = dist_matrix.shape[0]
    # 可达性掩码
    reachable = np.isfinite(dist_matrix)
    # 记录是否保留边 (i,j) , i<j
    keep = np.zeros((N, N), dtype=bool)

    # 初始只保留可达边
    for i in range(N):
        for j in range(i+1, N):
            if reachable[i, j]:
                keep[i, j] = True
                keep[j, i] = True

    # 冗余检查
    for i in range(N):
        for j in range(i+1, N):
            if not keep[i, j]:
                continue
            # 检查是否存在中间节点 k 使得 i->k + k->j <= i->j
            for k in range(N):
                if k == i or k == j:
                    continue
                if not (reachable[i, k] and reachable[k, j]):
                    continue
                if dist_matrix[i, k] + dist_matrix[k, j] <= dist_matrix[i, j] + epsilon:
                    keep[i, j] = False
                    keep[j, i] = False
                    break

    # 强制保留每个节点的最近邻（可达）
    if keep_nearest:
        for i in range(N):
            # 所有可达邻居（不包括自身）
            neighbors = [j for j in range(N) if j != i and reachable[i, j]]
            if not neighbors:
                continue
            # 找距离最小的邻居
            min_dist = np.inf
            nearest = -1
            for j in neighbors:
                d = dist_matrix[i, j]
                if d < min_dist:
                    min_dist = d
                    nearest = j
            if nearest != -1:
                keep[i, nearest] = True
                keep[nearest, i] = True

    # 收集保留的边
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if keep[i, j]:
                edges.append((i, j, dist_matrix[i, j]))
    return edges

def save_edges(edges, filename_edges="edges.txt", filename_index="edge_index.txt"):
    """保存边列表及PyG格式的边索引"""
    with open(filename_edges, 'w') as f:
        f.write("src\tdst\tdistance\n")
        for i, j, d in edges:
            f.write(f"{i}\t{j}\t{d:.6f}\n")
    with open(filename_index, 'w') as f:
        f.write("src\tdst\n")
        for i, j, _ in edges:
            f.write(f"{i}\t{j}\n")
            f.write(f"{j}\t{i}\n")   # 无向图需两条有向边
    print(f"边列表已保存至 {filename_edges} (共 {len(edges)} 条无向边)")
    print(f"边索引已保存至 {filename_index} (共 {2*len(edges)} 条有向边)")

if __name__ == "__main__":
    # 加载距离矩阵（请根据实际情况修改路径）
    dist = load_distance_matrix("distance_matrix.csv")
    print(f"距离矩阵大小: {dist.shape}")

    # 统计可达边数（原始，无自环）
    original_edges = np.sum(np.isfinite(dist) & ~np.eye(dist.shape[0], dtype=bool)) // 2
    print(f"原始可达边数: {original_edges}")

    # 简化
    edges = simplify_graph(dist, epsilon=1e-6, keep_nearest=True)
    print(f"简化后边数: {len(edges)}")

    # 检查孤立节点
    # 构建可达矩阵（基于保留的边）
    N = dist.shape[0]
    reachable_from_edges = np.zeros((N, N), dtype=bool)
    for i, j, _ in edges:
        reachable_from_edges[i, j] = True
        reachable_from_edges[j, i] = True
    degrees = np.sum(reachable_from_edges, axis=1)
    if np.any(degrees == 0):
        print("警告：存在孤立节点（没有保留任何边）！")

    # 保存
    save_edges(edges, "simplified_edges.txt", "基础数据/simplified_edge_index.txt")

