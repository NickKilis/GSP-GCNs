'''
    GCN from scratch with numpy:
        1. Message passing (multiplication with the Adjacency matrix of the Graph)
        2. Computation of a linear projection with W and then passes that through the activation function
        3. Backpropagation : independent of the Graph (exactly the same as in other NNs)
'''
#============================================================================#
#----------------------------- IMPORT LIBRARIES -----------------------------#
import numpy as np
from scipy.linalg import sqrtm 
from scipy.special import softmax
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import matplotlib.pyplot as plt
from matplotlib import animation

#------------------------- DEFINE UTILITY FUNCTIONS -------------------------#
def draw_kkl(nx_G, label_map, node_color, pos=None, **kwargs):
    fig, ax = plt.subplots(figsize=(10,10))
    if pos is None:
        pos = nx.spring_layout(nx_G, k=5/np.sqrt(nx_G.number_of_nodes()))
    nx.draw(nx_G, pos, with_labels=label_map is not None, labels=label_map, node_color=node_color, ax=ax, **kwargs)


def glorot_init(nin, nout):
    sd = np.sqrt(6.0 / (nin + nout))
    return np.random.uniform(-sd, sd, size=(nin, nout))


def xent(pred, labels):
    return -np.log(pred)[np.arange(pred.shape[0]), np.argmax(labels, axis=1)]


def norm_diff(dW, dW_approx):
    return np.linalg.norm(dW - dW_approx) / (np.linalg.norm(dW) + np.linalg.norm(dW_approx))

#----------------- GET LABELS VIA COMMUNITY DETECTION -----------------------#
g = nx.karate_club_graph()
g.number_of_nodes(), g.number_of_edges()
communities = greedy_modularity_communities(g)
colors = np.zeros(g.number_of_nodes())
for i, com in enumerate(communities):
    colors[list(com)] = i
n_classes = np.unique(colors).shape[0]
labels = np.eye(n_classes)[colors.astype(int)]
club_labels = nx.get_node_attributes(g,'club')

#------------------------ VISUALIZE DATA WITH LABELS ------------------------#
#_ = draw_kkl(g, club_labels, colors, cmap='gist_rainbow', edge_color='gray')
_ = draw_kkl(g, None, colors, cmap='gist_rainbow', edge_color='gray')

fig, ax = plt.subplots(figsize=(10,10))
pos = nx.spring_layout(g, k=5/np.sqrt(g.number_of_nodes()))
kwargs = {"cmap": 'gist_rainbow', "edge_color":'gray'}
nx.draw(g, pos, with_labels=False, node_color=colors, ax=ax, **kwargs)
#plt.savefig('karate_club_graph.png', bbox_inches='tight', transparent=True)

#----------------------- NORMALIZED ADJACENCY MATRIX ------------------------#
A = nx.to_numpy_matrix(g)
A_mod = A + np.eye(g.number_of_nodes()) # add self-connections
D_mod = np.zeros_like(A_mod)
np.fill_diagonal(D_mod, np.asarray(A_mod.sum(axis=1)).flatten())
D_mod_invroot = np.linalg.inv(sqrtm(D_mod))
A_hat = D_mod_invroot @ A_mod @ D_mod_invroot

#------------------------------ INPUT FEATURES ------------------------------#
'''
Since we dont have node features, we use the identity matrix in their place.
This can effectively:
    - map each node in the graph to a column of learnable parameters 
      in the first layer resulting in fully learnable node embeddings
'''
X = np.eye(g.number_of_nodes())

#-------------------------- GCN IMPLEMENTATION ------------------------------#
class GradDescentOptim():
    '''
    The parameter update process after the gradients were calculated.
    It's only role is to flow the gradients down the layers during backpropagation.
    '''
    def __init__(self, lr, wd):
        self.lr = lr
        self.wd = wd
        self._y_pred = None
        self._y_true = None
        self._out = None
        self.bs = None
        self.train_nodes = None
        
    def __call__(self, y_pred, y_true, train_nodes=None):
        self.y_pred = y_pred
        self.y_true = y_true
        
        if train_nodes is None:
            self.train_nodes = np.arange(y_pred.shape[0])
        else:
            self.train_nodes = train_nodes
            
        self.bs = self.train_nodes.shape[0]
        
    @property
    def out(self):
        return self._out
    
    @out.setter
    def out(self, y):
        self._out = y
    

class GCNLayer():
    def __init__(self, n_inputs, n_outputs, activation=None, name=''):
        '''
        GCN layer instanciation:
            - Define input and output feature dimensions, 
              which construct our learnable parameter matrix W.
        
        GCN equation:      H(l+1) = σ (W A_hat H(l)) 
            - l is the layer number.
            - l=0 : refers to the input features
        
        A GCN updates the node embeddings from H(l) to H(l+1).
        The update :
            - uses A_hat to multiply the incoming feature vector (message passing),
            - multiplies that result by a learned matrix of parameters W and
            - applies a non-linear activation function σ
        
        The GCN is basically a dense neural network layer, with the exception
        that initially it does the message passing on the graph, 
        by multiplying the input with the normalized adjacency matrix.
        
        Aside from the activation function, the only difference is the pre-computed
        inputs from the forward pass stored in _X. The actual gradient updates
        have nothing to do with the graph. It's just a standard feed-forward neural 
        network procedure where the incoming gradient is differentiated, 
        with respect to the layer's input, to get the partial derivatives on the W parameters. 
        '''
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(self.n_outputs, self.n_inputs)
        self.activation = activation
        self.name = name
        
    def __repr__(self):
        return f"GCN: W{'_'+self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"
        
    def forward(self, A, X, W=None):
        """
        Assumes:
            - A is (bs, bs) adjacency matrix and 
            - X is (bs, D), 
            where: 
            - bs = batch size and 
            - D = input feature length
        """
        # 1.STEP : Message passing
        self._X = (A @ X).T # the underscore is used later for calculating gradients.  (D, bs)
        
        if W is None:
            W = self.W
        
        # 2.STEP: Computation of a linear projection with W and then passes that through the activation function
        H = W @ self._X # (h, D)*(D, bs) -> (h, bs)
        if self.activation is not None:
            H = self.activation(H)
        self._H = H # (h, bs)
        return self._H.T # (bs, h)
    
    def backward(self, optim, update=True):
        # dtanh is the derivative of tanh activation function
        dtanh = 1 - np.asarray(self._H.T)**2 # (bs, out_dim)
        
        # d2 extracts the incoming gradients from GradDescentOptim out attribute,
        # which were computed by the downstream layer and passed them through the 
        # activation function by multiplying with dtanh.
        d2 = np.multiply(optim.out, dtanh)  # (bs, out_dim) *element_wise* (bs, out_dim)
        
        # Next we update GradDescentOptim out attribute, to represent the gradients 
        # that will pass to the next layer, which is jsut d2 projected backward by this layer's 
        # weights (which is the same as we did in the softmax layer)
        optim.out = d2 @ self.W # (bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)
        
        # also the same as in softmax these intermediate gradients are passed to the W matrix
        # by matrix multiplying with the input (_X from the forward pass).
        dW = np.asarray(d2.T @ self._X.T) / optim.bs  # (out_dim, bs)*(bs, D) -> (out_dim, D)
        
        dW_wd = self.W * optim.wd / optim.bs # weight decay update
        
        if update:
            self.W -= (dW + dW_wd) * optim.lr 
        
        return dW + dW_wd

    
class SoftmaxLayer():
    '''
    A Linear layer with softmax activation:
    - The forward pass does not perform the message passing step. It is just 
    applying the layer to the already computed node embeddings that constitute
    the output of the GCN layer.
    - After that it applies the softmax activation, with an additional trick
      about subtracting the max, for numerical stability. 
    '''
    
    def __init__(self, n_inputs, n_outputs, name=''):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(self.n_outputs, self.n_inputs)
        self.b = np.zeros((self.n_outputs, 1))
        self.name = name
        self._X = None # Used to calculate gradients
        
    def __repr__(self):
        return f"Softmax: W{'_'+self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"
    
    def shift(self, proj):
        shiftx = proj - np.max(proj, axis=0, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=0, keepdims=True)
        
    def forward(self, X, W=None, b=None):
        """Compute the softmax of vector x in a numerically stable way.
           X is assumed to be (bs, h)
        """
        self._X = X.T
        if W is None:
            W = self.W
        if b is None:
            b = self.b

        proj = np.asarray(W @ self._X) + b # (out, h)*(h, bs) = (out, bs)
        return self.shift(proj).T # (bs, out)
    
    def backward(self, optim, update=True):
        # should take in optimizer, update its own parameters and update the optimizer's "out"
        # Build mask that zeroes out the loss values for nodes that were not ised during training
        # This list of nodes is kept in the GradDescentOptim object
        train_mask = np.zeros(optim.y_pred.shape[0])
        train_mask[optim.train_nodes] = 1
        train_mask = train_mask.reshape((-1, 1))
        
        # d1 is the common equation for the partial derivative the cross entropy loss with respect to the pre-softmax values
        # In other words it flows the gradients of the loss function through the softmax activation
        # derivative of loss w.r.t. activation (pre-softmax)
        d1 = np.asarray((optim.y_pred - optim.y_true)) # (bs, out_dim)
        d1 = np.multiply(d1, train_mask) # (bs, out_dim) with loss of non-train nodes set to zero
        
        # When d1 is matrix multiplied with W, 
        # it gives the gradients flowing through the weights of this layer to the next layer.
        # The result is stored in the GradDescentOptim out attribute,
        # which will be used by the next layer to calculate its gradients.
        optim.out = d1 @ self.W # (bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)
        
        # To calculate the final gradients for this layer's parameteters we use d1.
        # The bias term db is just d1 directly but averaged accross the training batch.
        # For dW we have to matrix multiply with the input (_X from the forward pass).
        dW = (d1.T @ self._X.T) / optim.bs  # (out_dim, bs)*(bs, in_dim) -> (out_dim, in_dim)
        db = d1.T.sum(axis=1, keepdims=True) / optim.bs # (out_dim, 1)
        
        # option of L2 weight decay for the W parameters
        dW_wd = self.W * optim.wd / optim.bs # weight decay update
        
        # during the actual update process we multiply these gradients by the learning rate
        # and take a step in the negative direction, so as to minimize the loss function
        if update:   
            self.W -= (dW + dW_wd) * optim.lr
            self.b -= db.reshape(self.b.shape) * optim.lr
        
        return dW + dW_wd, db.reshape(self.b.shape)


gcn1 = GCNLayer(g.number_of_nodes(), 2, activation=np.tanh, name='1')
sm1 = SoftmaxLayer(2, n_classes, "SM")
opt = GradDescentOptim(lr=0, wd=1.)

gcn1_out = gcn1.forward(A_hat, X)
opt(sm1.forward(gcn1_out), labels)

#============= Gradient checking on Softmax layer ===========================#
def get_grads(inputs, layer, argname, labels, eps=1e-4, wd=0):
    cp = getattr(layer, argname).copy()
    cp_flat = np.asarray(cp).flatten()
    grads = np.zeros_like(cp_flat)
    n_parms = cp_flat.shape[0]
    for i, theta in enumerate(cp_flat):
        #print(f"Parm {argname}_{i}")
        theta_cp = theta
        
        # J(theta + eps)
        cp_flat[i] = theta + eps
        cp_tmp = cp_flat.reshape(cp.shape)
        predp = layer.forward(*inputs, **{argname: cp_tmp})
        wd_term = wd/2*(cp_flat**2).sum() / labels.shape[0]
        #print(wd_term)
        Jp = xent(predp, labels).mean() + wd_term
        
        # J(theta - eps)
        cp_flat[i] = theta - eps
        cp_tmp = cp_flat.reshape(cp.shape)
        predm = layer.forward(*inputs, **{argname: cp_tmp})
        wd_term = wd/2*(cp_flat**2).sum() / labels.shape[0]
        #print(wd_term)
        Jm = xent(predm, labels).mean() + wd_term
        
        # grad
        grads[i] = ((Jp - Jm) / (2*eps))
        
        # Back to normal
        cp_flat[i] = theta

    return grads.reshape(cp.shape)


dW_approx = get_grads((gcn1_out,), sm1, "W", labels, eps=1e-4, wd=opt.wd)
db_approx = get_grads((gcn1_out,), sm1, "b", labels, eps=1e-4, wd=opt.wd)


dW, db = sm1.backward(opt, update=False)

assert norm_diff(dW, dW_approx) < 1e-7
assert norm_diff(db, db_approx) < 1e-7

#================= Gradient checking on GCN layer ===========================#
def get_gcn_grads(inputs, gcn, sm_layer, labels, eps=1e-4, wd=0):
    cp = gcn.W.copy()
    cp_flat = np.asarray(cp).flatten()
    grads = np.zeros_like(cp_flat)
    n_parms = cp_flat.shape[0]
    for i, theta in enumerate(cp_flat):
        theta_cp = theta
        
        # J(theta + eps)
        cp_flat[i] = theta + eps
        cp_tmp = cp_flat.reshape(cp.shape)
        pred = sm_layer.forward(gcn.forward(*inputs, W=cp_tmp))
        w2 = (cp_flat**2).sum()+(sm_layer.W.flatten()**2).sum()
        Jp = xent(pred, labels).mean() + wd/(2*labels.shape[0])*w2
        
        # J(theta - eps)
        cp_flat[i] = theta - eps
        cp_tmp = cp_flat.reshape(cp.shape)
        pred = sm_layer.forward(gcn.forward(*inputs, W=cp_tmp))
        w2 = (cp_flat**2).sum()+(sm_layer.W.flatten()**2).sum()
        Jm = xent(pred, labels).mean() + wd/(2*labels.shape[0])*w2
        
        # grad
        grads[i] = ((Jp - Jm) / (2*eps))
        
        # Back to normal
        cp_flat[i] = theta

    return grads.reshape(cp.shape)

dW2 = gcn1.backward(opt, update=False)

dW2_approx = get_gcn_grads((A_hat, X), gcn1, sm1, labels, eps=1e-4, wd=opt.wd)

assert norm_diff(dW2, dW2_approx) < 1e-7

#============================= The GCN Model ================================#
class GCNNetwork():
    def __init__(self, n_inputs, n_outputs, n_layers, hidden_sizes, activation, seed=0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        
        np.random.seed(seed)
        
        self.layers = list()
        # Input layer
        gcn_in = GCNLayer(n_inputs, hidden_sizes[0], activation, name='in')
        self.layers.append(gcn_in)
        
        # Hidden layers
        for layer in range(n_layers):
            gcn = GCNLayer(self.layers[-1].W.shape[0], hidden_sizes[layer], activation, name=f'h{layer}')
            self.layers.append(gcn)
            
        # Output layer
        sm_out = SoftmaxLayer(hidden_sizes[-1], n_outputs, name='sm')
        self.layers.append(sm_out)
        
    def __repr__(self):
        return '\n'.join([str(l) for l in self.layers])
    
    def embedding(self, A, X):
        # Loop through all GCN layers
        H = X
        for layer in self.layers[:-1]:
            H = layer.forward(A, H)
        return np.asarray(H)
    
    def forward(self, A, X):
        # GCN layers
        H = self.embedding(A, X)
        
        # Softmax
        p = self.layers[-1].forward(H)
        
        return np.asarray(p)


gcn_model = GCNNetwork(
    n_inputs=g.number_of_nodes(), 
    n_outputs=n_classes, 
    n_layers=2,
    hidden_sizes=[16, 2], 
    activation=np.tanh,
    seed=100)

gcn_model

y_pred = gcn_model.forward(A_hat, X)
embed = gcn_model.embedding(A_hat, X)
xent(y_pred, labels).mean()

pos = {i: embed[i,:] for i in range(embed.shape[0])}
_ = draw_kkl(g, None, colors, pos=pos, cmap='gist_rainbow', edge_color='gray')


#================================ Training ==================================#
train_nodes = np.array([0, 1, 8])
test_nodes = np.array([i for i in range(labels.shape[0]) if i not in train_nodes])
opt2 = GradDescentOptim(lr=2e-2, wd=2.5e-2)

embeds = list()
accs = list()
train_losses = list()
test_losses = list()

loss_min = 1e6
es_iters = 0
es_steps = 50
# lr_rate_ramp = 0 #-0.05
# lr_ramp_steps = 1000

for epoch in range(15000):
    
    y_pred = gcn_model.forward(A_hat, X)

    opt2(y_pred, labels, train_nodes)
    
#     if ((epoch+1) % lr_ramp_steps) == 0:
#         opt2.lr *= 1+lr_rate_ramp
#         print(f"LR set to {opt2.lr:.4f}")

    for layer in reversed(gcn_model.layers):
        layer.backward(opt2, update=True)
        
    embeds.append(gcn_model.embedding(A_hat, X))
    # Accuracy for non-training nodes
    acc = (np.argmax(y_pred, axis=1) == np.argmax(labels, axis=1))[
        [i for i in range(labels.shape[0]) if i not in train_nodes]
    ]
    accs.append(acc.mean())
    
    loss = xent(y_pred, labels)
    loss_train = loss[train_nodes].mean()
    loss_test = loss[test_nodes].mean()
    
    train_losses.append(loss_train)
    test_losses.append(loss_test)
    
    if loss_test < loss_min:
        loss_min = loss_test
        es_iters = 0
    else:
        es_iters += 1
        
    if es_iters > es_steps:
        print("Early stopping!")
        break
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch+1}, Train Loss: {loss_train:.3f}, Test Loss: {loss_test:.3f}")
        
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)


fig, ax = plt.subplots()
ax.plot(np.log10(train_losses), label='Train')
ax.plot(np.log10(test_losses), label='Test')
ax.legend()
ax.grid()

accs[-1]

pos = {i: embeds[-1][i,:] for i in range(embeds[-1].shape[0])}
_ = draw_kkl(g, None, colors, pos=pos, cmap='gist_rainbow', edge_color='gray')


fig, ax = plt.subplots()
_ = ax.plot(accs, marker='o')
ax.grid()
_ = ax.set(ylim=[0,1])

N = 500
snapshots = np.linspace(0, len(embeds)-1, N).astype(int)

fig, ax = plt.subplots(figsize=(10, 10))
kwargs = {'cmap': 'gist_rainbow', 'edge_color': 'gray', }#'node_size': 55}

def update(idx):
    ax.clear()
    embed = embeds[snapshots[idx]]
    pos = {i: embed[i,:] for i in range(embed.shape[0])}
    nx.draw(g, pos, node_color=colors, ax=ax, **kwargs)

anim = animation.FuncAnimation(fig, update, frames=snapshots.shape[0], interval=10, repeat=False)
anim.save('embed_anim.mp4', dpi=300)
