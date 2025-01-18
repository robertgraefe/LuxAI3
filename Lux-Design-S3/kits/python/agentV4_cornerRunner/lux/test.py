from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np

graph = np.array([[0,1,1,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0],
                  [1,0,0,0,1,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,1,0,0,1,0,0],
                  [0,0,0,0,1,0,0,1],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,1,0,0],])

csr = csr_matrix(graph)

print(csr)

dist_matrix, pred = shortest_path(graph, return_predecessors=True)

print(dist_matrix)
print(pred)

def get_path(Pr, i, j):
    path = [np.int32(j)]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]

print(get_path(pred, 0, 7))

