#http://peterroelants.github.io/posts/rnn_implementation_part01/
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm


# Create dataset
nb_of_samples = 1
sequence_len = 2
# Create the sequences
X = np.zeros((nb_of_samples, sequence_len))
for row_idx in range(nb_of_samples):
    #X[row_idx,:] = np.around(np.random.rand(sequence_len)).astype(int)
    X[row_idx, :] = np.ones(sequence_len)

# Create the targets for each sequence
t = np.sum(X, axis=1)


print t
print X
print X.shape
#print t*X[1]
# Define the forward step functions
def update_state(xk, sk, wx, wRec):
    """
    Compute state k from the previous state (sk) and current input (xk),
    by use of the input weights (wx) and recursive weights (wRec).
    """
    return xk * wx + sk * wRec

def forward_states(X, wx, wRec):
    """
    Unfold the network and compute all state activations given the input X,
    and input weights (wx) and recursive weights (wRec).
    Return the state activations in a matrix, the last column S[:,-1] contains the
    final activations.
    """
    # Initialise the matrix that holds all states for all input sequences.
    # The initial state s0 is set to 0.
    S = np.zeros((X.shape[0], X.shape[1]+1))
    # Use the recurrence relation defined by update_state to update the
    #  states trough time.
    for k in range(0, X.shape[1]):
        # S[k] = S[k-1] * wRec + X[k] * wx
        S[:,k+1] = update_state(X[:,k], S[:,k], wx, wRec)
    return S

def cost(y, t):
    """
    Return the MSE between the targets t and the outputs y.
    """
    return ((t - y)**2).sum() / nb_of_samples

def output_gradient(y, t):
    """
    Compute the gradient of the MSE cost function with respect to the output y.
    """
    return 2.0 * (y - t) / nb_of_samples

def backward_gradient(X, S, grad_out, wRec,wx):
    """
    Backpropagate the gradient computed at the output (grad_out) through the network.
    Accumulate the parameter gradients for wX and wRec by for each layer by addition.
    Return the parameter gradients as a tuple, and the gradients at the output of each layer.
    """
    # Initialise the array that stores the gradients of the cost with respect to the states.
    grad_over_time = np.zeros((X.shape[0], X.shape[1]+1))
    grad_over_time[:,-1] = grad_out
    # Set the gradient accumulations to 0
    wx_grad = 0
    wRec_grad = 0
    for k in range(X.shape[1], 0, -1):
        # Compute the parameter gradients and accumulate the results.
        wx_grad += np.sum(grad_over_time[:,k] * X[:,k-1])
        wRec_grad += np.sum(grad_over_time[:,k] * S[:,k-1])
        # Compute the gradient at the output of the previous layer
        grad_over_time[:,k-1] = grad_over_time[:,k] * wRec

    print(wx_grad, wRec_grad,grad_over_time,X,S,wx,wRec)
    return (wx_grad, wRec_grad), grad_over_time
W_change=[0,0]
stable = False
stable_cnt = 0
last_error=100000
def update_rprop(X, t, W, W_prev_sign, W_delta, eta_p, eta_n):
    """
    Update Rprop values in one iteration.
    X: input data.
    t: targets.
    W: Current weight parameters.
    W_prev_sign: Previous sign of the W gradient.
    W_delta: Rprop update values (Delta).
    eta_p, eta_n: Rprop hyperparameters.
    """
    # Perform forward and backward pass to get the gradients
    S = forward_states(X, W[0], W[1])
    grad_out = output_gradient(S[:,-1], t)
    W_grads, _ = backward_gradient(X, S, grad_out, W[1],W[0])
    print(W_grads)
    W_sign = np.sign(W_grads)  # Sign of new gradient
    # Update the Delta (update value) for each weight parameter seperately
    error=cost(S[:, -1], t)

    for i, _ in enumerate(W):
        if W_sign[i] == W_prev_sign[i]:
            #W_delta[i] *= eta_p
            W_delta[i] = eta_p*W_up_delta[i]
        else:
            #W_delta[i] *= eta_n
            W_change[i]+=1
            if W_change[i] == 10:
                W_up_delta[i]*=0.9
                print("sign changes 10 times ,decay W_up_delta[i]:",i,W_up_delta[i])
                W_change[i]=0
            W_delta[i] = eta_n*W_up_delta[i]
    return W_delta, W_sign,error,W_grads

# Perform Rprop optimisation

# Set hyperparameters
eta_p = 1.2
eta_n = 0.5

# Set initial parameters
W = [-1.5, 2]  # [wx, wRec]
W_delta = [0.1, 0.1]  # Update values (Delta) for W
W_up_delta = [0.01, 0.01]
W_sign = [0, 0]  # Previous sign of W

ls_of_ws = [(W[0], W[1])]  # List of weights to plot
# Iterate over 500 iterations
for r in range(500):
    # Get the update values and sign of the last gradient
    if(r==196):
        W_up_delta[0]*=0.5
        W_up_delta[1]*= 0.5
    W_delta, W_sign,error,W_grads= update_rprop(X, t, W, W_sign, W_delta, eta_p, eta_n)
    # Update each weight parameter seperately
    for i, _ in enumerate(W):
        if(abs(W_grads[i])<abs(W_sign[i] * W_delta[i])):
            print("use W_grads")
            W[i]-=W_grads[i]
        else:
            W[i] -= W_sign[i] * W_delta[i]
    if(abs(error)<abs(last_error)):
        stable_cnt+=1
    else:
        stable_cnt=0

    print(r,W,W_sign,W_delta,error,stable_cnt,last_error)
    last_error = error
    ls_of_ws.append((W[0], W[1]))  # Add weights to list to plot

print('Final weights are: wx = {0},  wRec = {1}'.format(W[0], W[1]))

test_inpt = np.asmatrix([[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1]])
print test_inpt
test_outpt = forward_states(test_inpt, W[0], W[1])[:,-1]
print 'Target output: {:d} vs Model output: {:.2f}'.format(test_inpt.sum(), test_outpt[0])

