import time
import numpy as np
from scipy.linalg import null_space
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import eig
from sklearn.cluster import KMeans
from collections import defaultdict


def build_matrices(graph_name):  # cleaned_dataset/highschool_2011/t10.txt
    num_nodes = 0
    unique_labels = set()
    id2label = {}
    label2volume = {}
    label2members = defaultdict(list)

    attributes = open(graph_name.split('/t')[0] + '/attributes.txt', 'r')
    for line in attributes.readlines():
        num_nodes += 1
        items = line.strip().split(',')
        id = int(items[0])

        group_label = int
        if items[1] == 'M':
            group_label = 1-1  # make index of groups start from 0
        else:
            if items[1] == 'F':
                group_label = 2-1
            else:
                group_label = int(items[1])-1
        unique_labels.add(group_label)
        id2label[id] = group_label  # index of nodes already start from 0
        if group_label not in label2volume.keys():
            label2volume[group_label] = 1
        else:
            label2volume[group_label] += 1
        label2members[group_label].append(id)
    attributes.close()

    A = np.zeros((num_nodes, num_nodes), dtype=int)
    W = np.zeros((num_nodes, num_nodes), dtype=int)
    h = len(unique_labels)
    F = np.zeros((num_nodes, h-1), dtype=np.float32)

    # build A at t0
    graph = open(graph_name, 'r')
    for line in graph.readlines():
        items = line.strip().split(',')
        id_0 = int(items[0])  # index of nodes already start from 0
        id_1 = int(items[1])
        A[id_0, id_1] = 1
        A[id_1, id_0] = 1  # undirected
    graph.close()

    # build W at t0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if A[i, j] == 1:
                for k in range(j+1, num_nodes):
                    if A[j, k] == 1 and A[k, i] == 1:
                        W[i, j] += 1
                        W[j, i] += 1
                        W[j, k] += 1
                        W[k, j] += 1
                        W[i, k] += 1
                        W[k, i] += 1

    # build F, Z which are consistent
    e = np.ones((num_nodes, ), dtype=int)
    D = np.diag(np.dot(W,e))  # clique-weighted degree matrix
    for i in range(num_nodes):
        if D[i, i] == 0:
            D[i, i] = 1

    for i in range(num_nodes):
        label = id2label[i]
        if label < h-1:
            F[i, label] = 1 - (label2volume[label] / num_nodes)

    Z = null_space(np.transpose(F))
    Z = Z.real
    Q = sqrtm(np.dot(np.dot(np.transpose(Z), D),Z))
    Qinv = inv(Q).real

    L = D - W
    M = np.dot(np.dot(np.dot(np.dot(Qinv,np.transpose(Z)),L),Z),Qinv)
    M = (M + np.transpose(M))/2

    lamda, X = eig(M)

    # sort eigenvalues and eigenvectors
    idx = lamda.argsort()
    lamda = lamda[idx]
    X = X[:, idx]

    return label2members, D, W, A, Z, M, lamda.real, X.real


def fast_update_laplacian(edge_list, D, W, A):
    updated_edges = open(edge_list, 'r')
    for line in updated_edges.readlines():
        items = line.strip().split(',')
        id_0 = int(items[0])
        id_1 = int(items[1])

        if A[id_0, id_1] == 0:  # new edge
            A[id_0, id_1] = 1
            A[id_1, id_0] = 1

            id_0_one_hop = np.nonzero(A[id_0, :])[0]
            id_1_one_hop = np.nonzero(A[id_1, :])[0]

            involving_nodes = set(id_0_one_hop) & set(id_1_one_hop)

            for u in involving_nodes:
                W[id_0, id_1] += 1
                W[id_1, id_0] += 1
                W[id_0, u] += 1
                W[u, id_0] += 1
                W[id_1, u] += 1
                W[u, id_1] += 1

                D[id_0, id_0] += 1
                D[id_1, id_1] += 1
                D[u, u] += 1
        updated_edges.close()
    return D, W, A


def fast_track_eigens(M_last, M, lamda_last, U_last):
    U = U_last
    lamda = lamda_last
    q = lamda_last.shape[0]
    num_nodes = M.shape[0]

    delta_M = M - M_last

    X = np.dot(np.dot(np.transpose(U_last), delta_M), U_last)
    delta_lamda = np.diag(X)

    # delta_lamda = np.zeros((num_nodes,), np.float32)
    # for j in range(q):
    #     delta_lamda[j] = np.dot(np.dot(np.transpose(U_last[:, j]), delta_M), U_last[:, j])
    # X = np.diag(delta_lamda)

    if min(delta_lamda) > 0:  # with high-order tracking
        # update eigen-value
        lamda = lamda_last + delta_lamda

        # update eigen-vector
        for j in range(q):
            v = np.full((q,), lamda_last[j]) + np.full((q,), delta_lamda[j]) - lamda_last
            D = np.diag(v)
            alpha_j = np.dot(inv(D - X).real, X[:,j])
            delta_u_j = np.sum(np.multiply(alpha_j.T, U_last), axis=1)
            U[:, j] = U_last[:, j] + delta_u_j

    else:  # with low order tracking
        for j in range(q):
            # update eigen vector
            delta_u_j = np.zeros((num_nodes,), np.float32)
            for k in range(q):
                if k != j:
                    # temp = min(lamda_last[j] - lamda_last[k], 0.01)
                    temp = lamda_last[j] - lamda_last[k]
                    delta_u_j += np.dot(np.dot(np.transpose(U_last[:, k]), delta_M), U_last[:, j]) / (
                                temp) * U_last[:, k]
            U[:, j] = U_last[:, j] + delta_u_j
            # update eigen value
            delta_lamda_j = np.dot(np.dot(np.transpose(U_last[:, j]), delta_M), U_last[:, j])
            lamda[j] = lamda_last[j] + delta_lamda_j

    return lamda, U


def clustering(label2members, A, W, Z, Qinv, X, k):
    H = np.dot(np.dot(Z, Qinv), X)
    # print(H)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(H)

    # --- metrics --- #
    num_nodes = A.shape[0]
    V = np.array(range(num_nodes))
    num_groups = max(label2members.keys()) + 1  # i.e., h

    ncut = 0
    cancut = 0
    sum_balance_score = 0

    for l in range(k):
        cluster = np.where(kmeans.labels_ == l)[0]
        # print(cluster)
        cluster_bar = np.array(list(set(V).difference(set(cluster))))

        # 1. ncut
        cut = np.sum(A[cluster, :][:, cluster_bar])
        mu = np.sum(A[:, cluster])
        if mu != 0:
            ncut += cut / mu

        # 2. cancut
        cut = np.sum(W[cluster, :][:, cluster_bar])
        mu = np.sum(W[:, cluster])
        if mu != 0:
            cancut += cut / mu

        # 3. balance score
        score_list = []
        for s in range(num_groups):
            for s_prime in range(s + 1, num_groups):
                numerator = len(set(label2members[s]).intersection(set(cluster)))
                denominator = len(set(label2members[s_prime]).intersection(set(cluster)))

                if numerator == 0 or denominator == 0:
                    balance_score = 0
                else:
                    balance_score = numerator / denominator

                if balance_score > 1:
                    balance_score = 1 / balance_score

                score_list.append(balance_score)
        sum_balance_score += min(score_list)  # get the balance score of the cluster l

    avg_balance_score = sum_balance_score / k
    return ncut, cancut, avg_balance_score


if __name__ == '__main__':
    q_list = [5,6,7, 10,11,12, 14,15,16]
    q_list = [5,6,7]
    for q in q_list:
        print('------------ number of clusters : ' + str(q) + ' ------------')

        # static algorithm at t0
        graph_name = 'cleaned_dataset/highschool_2011/t0.txt'
        print("start building initial graph...")
        label2members, D, W, A, Z, M, lamda, X = build_matrices(graph_name)

        print("start solving initial solution...")
        Q = sqrtm(np.dot(np.dot(np.transpose(Z), D), Z))
        Qinv = inv(Q).real

        X = X[:,:q]
        lamda = lamda[:q]

        print("start tracking...")
        # start tracking from t1 to t10
        start_time = time.time()
        for t in range(10):
            print("--- start building graph t" + str(t) + "...")
            M_last = M
            lamda_last = lamda
            X_last = X

            # update L
            D, W, A = fast_update_laplacian(graph_name.split('/t')[0] + '/d' + str(t) + '.txt', D, W, A)

            # update M, obtain perturbation matrix
            Q = sqrtm(np.dot(np.dot(np.transpose(Z), D), Z))
            Qinv = inv(Q).real
            L = D - W
            M = np.dot(np.dot(np.dot(np.dot(Qinv, np.transpose(Z)), L), Z), Qinv)
            M = (M + np.transpose(M)) / 2

            print("--- start tracking solution t" + str(t) + "...")
            # track eigen-pairs
            lamda, X = fast_track_eigens(M_last, M, lamda_last, X_last)

        # finish tracking from t1 to t10, report performance
        time_consumption = time.time() - start_time
        ncut, cancut, avg_balance_score = clustering(label2members, A, W, Z, Qinv, X, q)
        print(ncut, cancut, avg_balance_score, time_consumption/10)



