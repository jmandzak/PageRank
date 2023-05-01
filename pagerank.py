# Inspiration taken from https://allendowney.github.io/DSIRP/pagerank.html

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def pageRank(graph, d, max_iter, epsilon):
    # Initialize PageRank value of each node
    page_rank = []
    for i in range(graph[0].size):
        page_rank.append(1 / graph[0].size)
    page_rank = np.array(page_rank)

    # compute the new page rank of each node
    num_iterations = 0
    for _ in range(max_iter):
        num_iterations += 1
        old_pr = page_rank.copy()
        for i in range(graph[0].size):
            # compute the sum of page rank/outgoing_links of each node
            sum_pr_divided_outgoing_links = 0
            for j, links in enumerate(graph):
                # if the given node does not connect to the current node or i and j are the same node, skip
                if j == i or links[i] == 0:
                    continue

                # this is the PR(node) / L(node) part of the formula
                outgoing_links = np.sum(links)
                sum_pr_divided_outgoing_links += page_rank[j] / outgoing_links

            # multiply the sum by the dampening factor, add to the base. Set new page_rank
            sum_pr_divided_outgoing_links *= d
            page_rank[i] = ((1 - d) / graph[0].size) + sum_pr_divided_outgoing_links

        # check to see if we need to loop again or if we're done
        difference = 0.0
        for val, old_val in zip(page_rank, old_pr):
            difference += abs(val - old_val)

        if difference < epsilon:
            break

    
    print(f'Number of iterations to converge: {num_iterations}')
    return page_rank


def main():
    # Create random graph, get adjacency matrix from it
    graph = nx.random_graphs.fast_gnp_random_graph(100, 0.75, directed=True)
    adj_matrix = nx.adjacency_matrix(graph).todense()
    adj_matrix = adj_matrix.astype(np.float64)

    # nx.draw_networkx(graph, with_labels=True)#, labels={0: 1, 1: 2, 2: 3, 3: 4, 4: 80})
    # plt.savefig('graph.png')

    # Calculate and print networkx implementation of pagerank
    page_rank = nx.pagerank(graph)
    print('Networkx Page Rank Values')
    for val in page_rank.values():
        print(round(val, 4), end=' ')
    print()

    # Calculate and print custom implementation of pagerank
    print('\nOur Page Rank Values')
    page_rank = pageRank(adj_matrix, 0.85, 10000, 0.000001)
    for val in page_rank:
        print(round(val, 4), end=' ')
    print()

if __name__ == '__main__':
    main()