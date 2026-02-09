import numpy as np
import json
import collections
import matplotlib.pyplot as plt


np.random.seed(42) ## random seed fixed 

d = 100 # dimensions of data
n = 1000 # number of data points

# generate table 1000*100 data points from N(0,1) distribution
X = np.random.normal(0,1, size=(n,d))
# generate identical table for testing
X_test = np.random.normal(0,1,size=(n,d)) 
# generate a 100*1 weight vector from N(0,1) distribution
# will act as the 'ground truth'
w_true = np.random.normal(0,1, size=(d,1))
y = X.dot(w_true) + np.random.normal(0,0.5,size=(n,1))
y_test = X_test.dot(w_true) + np.random.normal(0,0.5,size=(n,1))

#########   Do not change the code above  ############
######################################################

def square_loss(w,X,y):
	"""
	Implement total squared error given weight w, dataset (X,y)
	Inputs:
	- w: weight for the linear function
	- X: dataset of size (n,d)
	- y: label of size (n,1) 
	Returns:
	- loss: total squared error of w on dataset (X,y)
	"""
	predictions = X @ w
	residuals = predictions - y
	loss = np.sum(np.square(residuals))
	return loss
	

#### Implement closed-form solution given dataset (X,y)
def closed_form(X,y):
	"""
	Implement closed-form solution given dataset (X,y)
	Inputs:
	- X: dataset of size (n,d)
	- y: label of size (n,1) 
	Returns:
	- w_LS: closed form solution of the weight
	- loss: total squared error of w_LS on dataset (X,y)
	"""
	w_LS = np.linalg.inv(X.T @ X) @ X.T @ y
	predictions = X @ w_LS
	residuals = predictions - y
	total_loss = np.sum(np.square(residuals))
	return w_LS, total_loss


def gradient_descent(X, y, lr_set, N_iteration):
    """
    Implement gradient descent on the square-error given dataset (X, y) for each learning rate in lr_set.
    Inputs:
    - X: dataset of size (n, d)
    - y: label of size (n, 1)
    - lr_set: a list of learning rates
    - N_iteration: the number of iterations
    Returns:
    - a plot with k curves where k is the length of lr_set
    - each curve contains 20 data points, in which the i-th data point represents the total squared-error
      with respect to the i-th iteration
    - You can print the final objective value within this function to show the performance of the best step size
    """
    print("\nGRADIENT DESCENT \n")
    # create graph
    plt.figure(figsize=(10,6))
    
    for lr in lr_set:
        w_0 = np.zeros((d,1)) # initialize zero vector for each learning rate
        square_error_history = [] # reset history for each iteration
        for _ in range(N_iteration):
            prediction = X @ w_0 # how aligned is our guess with the weights
            residual = prediction - y
            #gradient calculation and update
            gradient = (2 / len(X)) * (X.T @ residual)
            w_0 = w_0 - lr * gradient
            
            #calculate square error for graphing purposes
            square_error = np.sum(residual**2)     
            square_error_history.append(square_error)
        print(f'lr: {lr} final loss: {square_error}')
        plt.plot(range(1, N_iteration+1), square_error_history, label=f'lr={lr}')
    
    plt.xlabel("Iteration")
    plt.ylabel("Square Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gradient_descent_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def stochastic_gradient_descent(X, y, lr_set, N_iteration):
	"""
	Implement gradient descent on the square-error given dataset (X,y) and for each learning rate in lr_set
	Inputs:
	- X: dataset of size (n,d)
	- y: label of size (n,1)
	- lr_set: a list of learning rate.
	- N_itertion: the number of iterations
	Returns:
	- a plot with k curves where k is the length of lr_set.
	- each curve contains 1000 data points, in which the i-th data point represents the total squared-error with respect to the i-th iteration
	- You can print the final objective value within this function to show the performance of best step size
	"""
	print("\n STOCHASTIC GRADIENT DESCENT \n")
	# create graph
	plt.figure(figsize=(10,6))
	np.random.seed(1) # Use this fixed random_seed in sampling
	for lr in lr_set:
		w_0 = np.zeros((d,1))
		square_error_history = []
		for _ in range(N_iteration):
			idx = np.random.randint(0, len(X))
			x_i, y_i  = X[idx:idx+1], y[idx:idx+1]
			prediction = x_i @ w_0 # how aligned is our guess with the weights
			residual = prediction - y_i
			#gradient calculation and update
			gradient = 2 * (x_i.T @ residual)
			w_0 = w_0 - lr * gradient
			#calculate square error for graphing purposes
			residual = x_i @ w_0 - y_i
			square_error = np.sum(residual**2)     
			square_error_history.append(square_error)
		print(f'lr: {lr} final loss: {square_error}')
		plt.plot(range(1, N_iteration+1), square_error_history, label=f'lr={lr}')
		plt.xlabel("Iteration")
		plt.ylabel("Square Loss")
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(f'sgd_plot_lr{lr}.png', dpi=300, bbox_inches='tight')
		plt.close()


def main():
	### Problem 4.1 ###
	# w_LS, loss_LS_train = closed_form(X,y)
	# w_0 = np.zeros((d,1)) # zero vector as the initial weight
	# loss_0_train = square_loss(w_0,X,y)
	# loss_LS_test = square_loss(w_LS,X_test, y_test)

	# print("F(w_LS)=", loss_LS_train, " on training data")
	# print("F(w_0)=", loss_0_train, " on training data")
	# print("F(w_LS)=", loss_LS_test, " on testing data")


	### Problem 4.2 (Gradient Descent) ###
	### You can plot more options of lr_set if necessary
	lr_set = [0.00005, 0.0005, 0.0007, 0.005, 0.007, 0.05, 0.07, 0.5]
	# w_0 = np.zeros((n,1))
	N_iter = 20
	gradient_descent(X,y,lr_set,N_iter)

	### Problem 4.3 (Stochastic Gradient Descent) ###
	### You can plot more options of lr_set if necessary
	lr_set = [0.0005, 0.005, 0.01]
	w_0 = np.zeros((n,1))
	N_iter = 1000
	stochastic_gradient_descent(X,y,lr_set,N_iter)


if __name__ == "__main__":
	main()