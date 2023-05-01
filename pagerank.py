# Inspiration taken from https://allendowney.github.io/DSIRP/pagerank.html

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time


def pageRank(graph, d, max_iter, epsilon):
    # Initialize PageRank value of each node
    page_rank = []
    for i in range(graph[0].size):
        page_rank.append(1 / graph[0].size)
    page_rank = np.array(page_rank)

    # compute the new page rank of each node
    # num_iterations = 0
    for _ in range(max_iter):
        # num_iterations += 1
        old_pr = page_rank.copy()
        for i in range(graph[0].size):
            # compute the sum of page rank/outgoing_links of each node
            sum_pr_divided_outgoing_links = 0
            for j, links in enumerate(graph):
                # if the given node does not connect to the current node, skip
                if links[i] == 0:
                    continue

                # this is the PR(node) / L(node) part of the formula
                outgoing_links = np.sum(links)
                sum_pr_divided_outgoing_links += page_rank[j] / outgoing_links

            # multiply the sum by the dampening factor, add to the base. Set new page_rank
            sum_pr_divided_outgoing_links *= d
            page_rank[i] = ((1 - d) / graph[0].size) + sum_pr_divided_outgoing_links

        # normalize pagerank, helps with nodes with no outgoing edges
        page_rank = page_rank / np.linalg.norm(page_rank, ord=1)

        # check to see if we need to loop again or if we're done
        difference = 0.0
        for val, old_val in zip(page_rank, old_pr):
            difference += abs(val - old_val)

        if difference < epsilon:
            break

    
    # print(f'Number of iterations to converge: {num_iterations}')
    return page_rank

def fastPageRank(graph, d, max_iter, epsilon):
    # initialize page_rank vector
    page_rank = []
    for _ in range(graph[0].size):
        page_rank.append(1.0 / graph[0].size)

    page_rank = np.array(page_rank)

    for _ in range(max_iter):
        # initialize old page rank vector, and sum
        old_pr = page_rank.copy()
        summation_terms = []
        for i, row in enumerate(graph):
            summation_terms.append(old_pr[i] / np.sum(row))

        # now calc each new page rank
        for i, row in enumerate(graph):
            new_pr = np.dot(summation_terms, graph[:, i])
            new_pr *= d
            new_pr += ((1 - d) / row.size)
            page_rank[i] = new_pr

        # normalize
        page_rank = page_rank / np.linalg.norm(page_rank, ord=1)

        # check to see if we can exit early
        total_difference = 0
        for old_val, val in zip(old_pr, page_rank):
            total_difference += abs(old_val - val)
        if total_difference < epsilon:
            break

    return page_rank


def randomWalk(graph, max_iter):
    # initialize the visited values to 0 for each node
    visited = []
    for _ in range(graph[0].size):
        visited.append(0)

    current_node = 0
    for _ in range(max_iter):
        row = graph[current_node]
        choice = np.random.random()
        
        # if choice is less than 0.15 (which is the probability it will jump to a random node), jump to a random node
        if choice < 0.15:
            current_node = np.random.choice(graph[0].size)
        else:
            # construct possible options
            connected_nodes = []
            for i, edge in enumerate(row):
                if edge == 1:
                    connected_nodes.append(i)

            # special case if there's no outgoing edges
            if len(connected_nodes) == 0:
                current_node = np.random.choice(graph[0].size)
            else:
                # jump to next node
                current_node = np.random.choice(connected_nodes)
        
        visited[current_node] += 1
    
    # normalize visited
    visited = np.divide(visited, max_iter)
    return visited


def main():
    # Create random graph with numpy random function
    adj_matrix = np.random.randint(0, 2, size=(5,5))
    graph = nx.convert_matrix.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # nx.draw_networkx(graph, with_labels=True)#, labels={0: 1, 1: 2, 2: 3, 3: 4, 4: 80})
    # plt.savefig('graph.png')

    # Calculate and print networkx implementation of pagerank
    old_time = time.time()
    page_rank = nx.pagerank(graph)
    new_time = time.time()
    print('Networkx Page Rank')
    print(f'Time: {round(new_time-old_time, 4)}')
    for val in page_rank.values():
        print(round(val, 4), end=' ')
    print()

    # Calculate and print custom implementation of pagerank
    custom_adj_matrix = adj_matrix.copy()
    # if there are nodes with no outgoing connections, add outgoing edge to all nodes
    for i, row in enumerate(custom_adj_matrix):
        if(np.sum(row) == 0):
            for j in range(row.size):
                custom_adj_matrix[i][j] = 1

    print('\nOur Page Rank')
    old_time = time.time()
    page_rank = pageRank(custom_adj_matrix, 0.85, 100, 0.000001)
    new_time = time.time()
    print(f'Time: {round(new_time-old_time, 4)}')
    for val in page_rank:
        print(round(val, 4), end=' ')
    print()

    print('\nOur Fast Page Rank')
    old_time = time.time()
    page_rank = fastPageRank(custom_adj_matrix, 0.85, 100, 0.000001)
    new_time = time.time()
    print(f'Time: {round(new_time-old_time, 4)}')
    for val in page_rank:
        print(round(val, 4), end=' ')
    print()

    # Create a random walk with max_iter iterations to see how close it is to PageRank
    print('\nRandom Walk')
    random_walk_vals = randomWalk(adj_matrix, 10000)
    for val in random_walk_vals:
        print(round(val, 4), end=' ')
    print()

if __name__ == '__main__':
    main()