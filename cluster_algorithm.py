"""
Apply clustering after:
"Community detection for correlation matrices", Mel MacMahon, Diego Garlaschelli, 2015
http://arxiv.org/abs/1311.1924

Based on a modified Louvain method. Original Louvain method is from:
"Fast unfolding of communities in large networks", Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte,
Etienne Lefebvre, 2008
http://arxiv.org/pdf/0803.0476

The code is inspired by the Matlab implementation of the Louvain method written by Antoine Scherrer. It can be found at:
https://perso.uclouvain.be/vincent.blondel/research/louvain.html

The additional possibility, brought by this thesis, for a node to move to an empty community is implemented.
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

__author__ = 'Benjamin'


def cluster(m, verbose=True, shuffle=True, recursive=True, m_init=pd.DataFrame()):
    """ Detects communities by maximizing the modularity of the partition, using the modified Louvain method.
    :param m: Matrix of similarity (between days or stocks). Modified to account for noise and possibly the global mode.
    :type m: pandas.DataFrame
    :param verbose: If 1, the algorithm will print what it is doing.
    :type verbose: bool
    :param shuffle: Randomize the order in which the nodes are moved. This can yield different partitions.
    :type shuffle: bool
    :param recursive: 0, gives the first pass communities. 1, returns the partition with maximal modularity.
    :type recursive: bool
    :param m_init: Initial (non-modified) matrix. Used to compute the modularity. Does not impact the clustering.
    It only impacts the modularity as a scaling factor.
    :type m_init: pandas.DataFrame
    :return: List of dictionaries. The elements of the list contain a dictionary for one pass. Each dictionary contains:
    'COM': the community partition, 'SIZE': the size of each community, 'MOD': the modularity of the given partition,
    'Niter': the number of iterations of the pass, 'end': handles the case that m not possible to decompose.
    :type: list
    """

    assert isinstance(m, pd.DataFrame), "m must be a pandas.DataFrame"
    index = list(m.columns.values)
    if m_init.empty:
        m_init = m
    c_norm = float(np.abs(m_init).sum().sum())
    n = m.shape[0]

    ending = 0
    if float(c_norm) == 0. or n == 1:
        print "NO MORE DECOMPOSITION"
        ending = 1
        community = [{"end": ending}]
        return community

    n_iter = 0

    com = np.arange(n)  # Each node is put in an individual community at the beginning.
    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- Start of the first phase -----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    gain = True
    same_partition = True
    range_n = range(n)
    extra_community = range(n)  # extra community identifiers
    while gain:
        gain = False
        # randomize the order in which the nodes are moved. This can yield different partitions.
        if shuffle:
            np.random.shuffle(range_n)
        for i in range_n:
            com_i = com[i]  # community of stock i
            g = np.zeros(n) - 1  # gain vector
            best_increase = 0
            com_new = com_i
            com[i] = -1
            com_i_nodes = np.where(com == com_i)[0]  # array of the indices of the stocks in community i

            i_col = m.iloc[:, i].values

            c_ij1 = i_col[com_i_nodes]  # strength of the links of i in its previous community
            q_removed = - c_ij1.sum()/c_norm  # modularity gain from removing i from its previous community
            # if removing a node from its community increases the modularity,
            # then we consider the possibility of a move to an empty community
            extra = False
            if q_removed > 0:
                best_increase = q_removed
                # find empty community identifiers, and choose one at random
                free_com_id = np.array(extra_community)[np.in1d(extra_community, com, invert=True)]
                c_new_t = np.random.choice(free_com_id)
                extra = True

            nb = np.where(i_col)[0]  # take the neighbours of i (nodes with a non-zero link)
            if shuffle:
                np.random.shuffle(nb)  # it does not have to be shuffled. The result would only change if two target
            # communities yield the same modularity increase.
            for j in nb:
                com_j = com[j]
                if com_i != com_j and j != i:  # check all neighbours j of i except those in its own community (faster)
                    if g[com_j] == -1:  # we do not consider several times the same community (faster)
                        com_j_nodes = np.where(com == com_j)[0]  # array of the indices of the stocks in community j

                        c_ij2 = i_col[com_j_nodes]  # strength of the links of i in the community of j
                        q_added = c_ij2.sum()/c_norm  # modularity gain from adding i in the community of j
                        q_increase = q_removed + q_added  # modularity gain from this operation
                        g[com_j] = q_increase

                        if q_increase > best_increase:
                            best_increase = q_increase
                            c_new_t = com_j  # temporary new community
                            extra = False
            # print str(i) + " best increase: " + str(best_increase)
            if best_increase > 0:  # if the maximum gain of modularity is positive, we change the community if i
                com_new = c_new_t  # new community is the one with the best increase in modularity
                if verbose:
                    print "moved %d to %d" % (i, com_new)
                    if extra:
                        print 'MOVED TO AN EXTRA COMMUNITY!'

            com[i] = com_new  # i is placed in the community for which the gain is maximum.
            # If no positive gain is possible, i stays in its original community.

            if com_new != com_i:  # If no community changed, we get out of the while loop.
                gain = True  # If any of the stock changed community, we stay in the loop.
                same_partition = False

        n_iter += 1
        if verbose:
            if gain:
                print "END OF ITERATION: %d" % n_iter
            else:
                print 'ITERATIONS ARE COMPLETED'

    n_iter -= 1
    com, com_size = reindex_communities(com)
    community = [{"COM": pd.DataFrame(com, index), "SIZE": com_size, "MOD": modularity(com, m, c_norm), "Niter": n_iter,
                  "end": ending, "same partition": same_partition}]
    # community is a list of dictionaries. Each element of the list contains the information of one pass.
    if verbose:
        print "MODULARITY IS: %.9f" % community[0]["MOD"]

    if not recursive:
        return community
    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Start of the second phase -----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    else:
        m_new = m  # m_new will contain the strength of the links between the hyper-nodes,
        # ie. self-loops and renormalized interactions.
        com_current = com  # Community of the hyper-nodes.
        com_full = com  # Community of the single name nodes.
        k = 2
        if k == 2 and verbose:
            print "PASS NO.1 IS COMPLETED"

        while True:
            m_old = m_new
            n_node = m_old.shape[0]  # Number of hyper-nodes.

            com_unique = np.unique(com_current)

            n_com = len(com_unique)  # number of communities
            ind_com = np.zeros((n_com, n_node)) - 1  # The lines will contain the indices of the hyper-nodes in comm. i
            ind_com_full = np.zeros((n_com, n)) - 1  # The lines will contain the indices of the nodes in comm. i

            for i in xrange(n_com):
                ind = np.where(com_current == i)[0]
                ind_com[i, range(len(ind))] = ind
            for i in xrange(n_com):
                ind = np.where(com_full == i)[0]
                ind_com_full[i, range(len(ind))] = ind

            m_new = np.zeros((n_com, n_com)) - 1
            m_new = pd.DataFrame(m_new)
            for i in xrange(n_com):
                for j in xrange(i, n_com):
                    ind1 = ind_com[i, :]
                    ind2 = ind_com[j, :]
                    ind1 = ind1[np.where(ind1 >= 0)].astype(int)  # nodes in community m
                    ind2 = ind2[np.where(ind2 >= 0)].astype(int)  # nodes in community n
                    m_new.ix[i, j] = m_new.ix[j, i] = m_old.ix[ind1, ind2].sum().sum()  # renormalized interactions

            # new communities of the hyper-nodes
            community_t = cluster(m_new, verbose, shuffle, recursive=False, m_init=m_init)
            end = community_t[0]["end"]

            if not end:
                same_partition = community_t[0]["same partition"]
                com_full = np.zeros(n) - 1
                com_current = community_t[0]["COM"].T.values[0]
                for i in xrange(n_com):
                    ind1 = ind_com_full[i, :]
                    ind1 = ind1[np.where(ind1 >= 0)].astype(int)
                    com_full[ind1] = com_current[i]  # new communities of the single-name nodes

                com_full, com_size = reindex_communities(com_full)
                community_new = {"COM": pd.DataFrame(com_full, index), "SIZE": com_size,
                                 "MOD": modularity(com_full, m, c_norm),
                                 "Niter": community_t[0]["Niter"], "end": end, "same partition": same_partition}

                if same_partition:
                    if verbose:
                        print "identical segmentation => END\nTOTAL PASSES: %d" % (k - 1)
                    return community
                else:
                    community.append(community_new)
                    if verbose:
                        print "PASS NO. %d IS COMPLETED" % k
            else:
                if verbose:
                    print "EMPTY MATRIX OR NO MORE MERGING POSSIBLE => END"
                return community
            k += 1


def reindex_communities(com):
    """ Reindexes the communities according to their size. 0 is now the larger community.
    :param com: Array of communities.
    :return: com_reindex : Re-indexed array of communities.
    :return: com_size : Size of the communities, in decreasing order.
    """
    com_reindex = np.zeros(len(com)) - 1
    com_unique = np.unique(com)  # the sorted unique elements of com
    size = np.zeros(len(com_unique)) - 1
    for i in range(0, len(com_unique)):
        size[i] = len(com[np.where(com == com_unique[i])])  # number of elements in each community

    com_size = -np.sort(-size)  # sort the sizes in decreasing order
    idx = np.argsort(-size)  # indices that sort size to get com_sizes: size[idx] = com_size

    for i in range(0, len(com_unique)):
        com_reindex[np.where(com == com_unique[idx[i]])] = i  # assign the new identification number to communities

    com_reindex = com_reindex.astype(int)
    com_size = com_size.astype(int)

    return com_reindex, com_size


def modularity(com, m, c_norm):
    """ Computes the modularity of the partition.
    :param com: Array of communities.
    :param m: Matrix of link weights (matrix modified for noise and possibly global mode).
    :param c_norm: Sum of all the elements in the original matrix.
    :return: Modularity of the partition.
    """
    com_unique = np.unique(com)
    modularity_ = 0.
    for i in range(len(com_unique)):
        com_i = np.where(com == com_unique[i])[0]
        modularity_ += m.ix[com_i, com_i].sum().sum()
    modularity_ /= c_norm
    return modularity_


def cluster_iter(m, iterations, debug=1, shuffle=1, verbose=0, jobs=-1, recursive=True):
    """ The Louvain algorithm is sensitive to the order in which we pick the nodes to move.
    This can give different partitions.
    We iterate the algorithm and select the partition that give the largest modularity.
    :param m: Matrix (=network) to cluster.
    :type m: pandas.DataFrame
    :param iterations: Number of desired iterations.
    :type iterations: int
    :param shuffle: Randomize the order in which the nodes are moved. This can yield different partitions.
    :type shuffle: bool
    :param verbose: (bool)
    :param debug: (bool)
    :param jobs: Number of multiprocessors used. If you want to use other programs while running, set it to 1.
    :type jobs: int
    :return: Same as the cluster function.
    """

    param = [(m, verbose, shuffle, recursive) for _ in range(iterations)]

    cs = Parallel(n_jobs=jobs)(delayed(wrapper)(p) for p in param)
    mod = [cs[i][-1]["MOD"] for i in range(iterations)]
    t = np.where(mod == max(mod))[0]  # select largest modularity
    t = t[0]
    if debug:
        print "Modularities: " + str(mod)
        print "Position of max modularity: " + str(t)
    return cs[t]


def wrapper(param):
    return cluster(*param)

