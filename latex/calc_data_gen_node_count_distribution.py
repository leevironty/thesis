import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

def get_all_paths() -> list[list[tuple[int, int]]]:
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (2, 5),
        (5, 6),
        (5, 7),
        (2, 8),
    ]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    nodes = list(range(1, 9))
    all_paths: list[list[tuple[int, int]]] = []

    while nodes:
        source = nodes.pop(0)
        all_paths += list(
            path for path in nx.all_simple_edge_paths(graph, source, nodes)
            if len(path) >= 2
        )
    return all_paths


def get_unique_counts(sample_count) -> tuple[np.ndarray, np.ndarray]:
    max_attempts = 100
    rng = random.Random(123)
    all_paths = get_all_paths()
    lens: list[int] = []

    for i in range(sample_count):
        got_connected = False
        for _ in range(max_attempts):
            num_lines = rng.randint(2, 4)
            paths = rng.sample(all_paths, k=num_lines)
            g = nx.Graph()
            for line in paths:
                g.add_edges_from(line)
            if nx.is_connected(g):
                got_connected = True
                break
        if not got_connected:
            raise RuntimeError(
                f'Did not find a connected line plan in {max_attempts} attempts.'
            )
        
        lens.append(len(list(g.nodes)))
    
    out = np.array(lens)
    unique, counts = np.unique(out, return_counts=True)
    return unique, counts


def main():
    sample_count = 10000
    unique, counts = get_unique_counts(sample_count)

    for u, c in zip(unique, counts):
        print(f'value: {u}, share: {c / sample_count:.4f}')
    
    fig, ax = plt.subplots()

    ax.bar(unique, counts / sample_count)
    ax.set_xlabel('$|S|$')
    ax.set_ylabel('share of problem instances')

    fig.tight_layout()
    fig.savefig('figures/data_node_count_shares.pdf')


if __name__ == '__main__':
    main()
