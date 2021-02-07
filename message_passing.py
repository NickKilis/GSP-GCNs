import numpy as np
from scipy.linalg import sqrtm 
from scipy.special import softmax
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import matplotlib.pyplot as plt
from matplotlib import animation
#----------------------------------------------------------------------------#
#=================  Message Passing as Matrix Multiplication  ===============#
# adjacency matrix
A = np.array([[0, 1, 0, 0, 0], 
              [1, 0, 1, 0, 0],
              [0, 1, 0, 1, 1], 
              [0, 0, 1, 0, 0], 
              [0, 0, 1, 0, 0]])

# feature vector
feats = np.arange(A.shape[0]).reshape((-1,1))+1

# matrix multiplication between the adjacency matrix and the feature vector
H = A @ feats
''' 
A) The feature vector (Graph Signal) supported by a Graph structure:
    
                         (5)
                          |
                          |
                 (4)-----(3)     
                          |
                          |
             (1)---------(2)         

    The multiplication of the adjacency matrix and the feature vector is :
        - a way to mask out all the values, 
        - except the ones that node i has a connection with.
    
    e.g. First row (corresponding to node 1): 
        - Every element of the feature vector is multiplied by zero, 
        - except the value corresponding to node 2, which is node's 1 only graph connection.
    
        [0 1 0 0 0]   [1]     [(0)(1) + (1)(2) + (0)(3) + (0)(4) + (0)(5)]     [2]  
        [1 0 1 0 0]   [2]     [(1)(1) + (0)(2) + (1)(3) + (0)(4) + (0)(5)]     [4]
        [0 1 0 1 1] * [3]  =  [(0)(1) + (1)(2) + (0)(3) + (1)(4) + (1)(5)]  =  [11]
        [0 0 1 0 0]   [4]     [(0)(1) + (0)(2) + (1)(3) + (0)(4) + (0)(5)]     [3]
        [0 0 1 0 0]   [5]     [(0)(1) + (0)(2) + (1)(3) + (0)(4) + (0)(5)]     [3]
    
    The final result of this operation:
        - a new feature vector (with the same shape as the original)
        - but now each value represents the sum of the connected neighborhoods of each node.
    
    This is a simple form of message passing:
        - message              : feature vectors,
        - aggregation function : sum.


B) Alternative aggregation function: Average
    - Divide each element of the result by the neighborhood size of the node.
    - Implementation by using the Degree matrix D:
        
        [1, 0, 0, 0, 0] 
        [0, 2, 0, 0, 0]
        [0, 0, 3, 0, 0]
        [0, 0, 0, 1, 0] 
        [0, 0, 0, 0, 1]
    
    1. Take the inverse of matrix D:
            
        [1, 0,   0,   0, 0] 
        [0, 0.5, 0,   0, 0]
        [0, 0,   0.3, 0, 0]
        [0, 0,   0,   1, 0] 
        [0, 0,   0,   0, 1]
            
    2. multiply the adjacency matrix with the inverse of D  (D^-1 A = Aavg):
            
        [0, 1, 0, 0, 0]     [1,   0,   0, 0, 0]    [  0,   1,   0,   0,   0]
        [1, 0, 1, 0, 0]     [0, 0.5,   0, 0, 0]    [0.5,   0, 0.5,   0,   0]
        [0, 1, 0, 1, 1]  *  [0,   0, 0.3, 0, 0]  = [  0, 0.3,   0, 0.3, 0.3]
        [0, 0, 1, 0, 0]     [0,   0,   0, 1, 0]    [  0,   0,   1,   0,   0] 
        [0, 0, 1, 0, 0]     [0,   0,   0, 0, 1]    [  0,   0,   1,   0,   0]
        
    In this way we effectively assign a weight to each edge in the adjacency matrix,
    such that each value in a row is divided by the neighborghood size of that row.
    
    If we use this matrix to multiply the feature vector: 
        - we still mask out the non-connections,
        - but also scale each value by the neighborhood size.
        - Thus we compute the average rather than the sum.
    
    [  0,   1,   0,   0,   0]   [1]     [(0)(1)   + (1)(2)   + (0)(3)   + (0)(4)   + (0)(5)]      [2]  
    [0.5,   0, 0.5,   0,   0]   [2]     [(0.5)(1) + (0)(2)   + (0.5)(3) + (0)(4)   + (0)(5)]      [2]
    [  0, 0.3,   0, 0.3, 0.3] * [3]  =  [(0)(1)   + (0.3)(2) + (0)(3)   + (0.3)(4) + (0.3)(5)]  = [3.3]
    [  0,   0,   1,   0,   0]   [4]     [(0)(1)   + (0)(2)   + (1)(3)   + (0)(4)   + (0)(5)]      [3]
    [  0,   0,   1,   0,   0]   [5]     [(0)(1)   + (0)(2)   + (1)(3)   + (0)(4)   + (0)(5)]      [3]

C) Self connections (relevant to GCNs):
    - Add edges from each node to itself, forming a new modified adjacency matrix.
    - The implication of this procedure is, when we perform the neighborhood 
      aggregation of messages we include each node's own value in that aggregation.
    - Self connections derive directly from the convolution operation in which 
      we also aggregate information from the central element.
    - The modified adjacency matrix is : 
        A_mod = A + I , where I is the identity matrix.
    - The scaling in this case is performed:
        D^1/2 A_mod D^1/2
        - by putting D^1/2 on the right side of A_mod we scale by the 
          neighborhood size of the destination node rather than the source node.
        - in other words:
            1. we just normalize the connections in the adjacency matrix by using the neighborhood sizes of both the
              source and the destination nodes 
            2. each message is scaled by the neighborhood size that the message is coming from (dj) in a weighted sum
               and then this overall weighted sum is scaled by 1 over the square root of the degree of node source i (di).
               
               A_mod(i,j) = A(i,j) 1/ sqrt(di dj)
               
'''
#==== Scale neighborhood sum by neighborhood size (i.e. average values) =====#
D = np.zeros(A.shape)
np.fill_diagonal(D, A.sum(axis=0))

D_inv = np.linalg.inv(D)

D_inv @ A

H_avg = D_inv @ A @ feats

#========================= Normalized Adjacency Matrix ======================#

g = nx.from_numpy_array(A)
A_mod = A + np.eye(g.number_of_nodes())

# D for A_mod:
D_mod = np.zeros_like(A_mod)
np.fill_diagonal(D_mod, A_mod.sum(axis=1).flatten())

# Inverse square root of D:
D_mod_invroot = np.linalg.inv(sqrtm(D_mod))


node_labels = {i: i+1 for i in range(g.number_of_nodes())}
pos = nx.planar_layout(g)

fig, ax = plt.subplots(figsize=(10,10))
nx.draw(g, pos, with_labels=True, 
        labels=node_labels, 
        node_color='#83C167', 
        ax=ax, edge_color='gray', node_size=1500, font_size=30, font_family='serif')
plt.savefig('simple_graph.png', bbox_inches='tight', transparent=True)

A_hat = D_mod_invroot @ A_mod @ D_mod_invroot

#========================== Diffusion mechanism =============================#
#=============== Application the adjacency matrix repeatedly ================#
'''
    Allow the Graph Signal to travel to more distant neighborhoods in the Graph,
    until it eventually reaches a steady state.
'''
H = np.zeros((g.number_of_nodes(), 1))
H[0,0] = 1 
iters = 10
results = [H.flatten()]
for i in range(iters):
    H = A_hat @ H
    results.append(H.flatten())

print(f"Initial signal input: {results[0]}")
print(f"Final signal output after running {iters} steps of message-passing:  {results[-1]}")

fig, ax = plt.subplots(figsize=(10, 10))
kwargs = {'cmap': 'hot', 'node_size': 1500, 'edge_color': 'gray', 'vmin': np.array(results).min(), 'vmax': np.array(results).max()*1.1}

def update(idx):
    ax.clear()
    colors = results[idx]
    nx.draw(g, pos, node_color=colors, ax=ax, **kwargs)
    ax.set_title(f"Iter={idx}", fontsize=20)

# writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
anim = animation.FuncAnimation(fig, update, frames=len(results), interval=1000, repeat=True)
# anim.save('diffusion.mp4',writer=writer, dpi=600, bitrate=-1,savefig_kwargs={'transparent': True, 'facecolor': 'none'})
