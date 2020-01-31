#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def sigmoid(z):
    return (1/(1+tf.exp(-z)))


# In[ ]:


def wt(H,c1,y):
    w= np.dot(np.dot(np.linalg.pinv((1/c1)+np.dot(H.T,H)),H.T),y);
    return w;


# In[ ]:


def elmae(X,c,n1): # n1 is hidden node no.
    n=len(X.columns)
    w1= np.random.randn(n,n1);
    b1 = np.random.randn(1,n1);
    H= sigmoid(np.dot(X,w1)+b1);
    w= np.dot(np.dot(np.linalg.pinv((1/c)+np.dot(H.T,H)),H.T),X);
    return (w.T);


# In[ ]:


def red(Wn,n2,sigma):
    u = np.zeros([n2,n2-1])
    for i in range(0,n2):
        b=0;
        for j in range(0,n2):
            if (i==j):
                continue;
            else:
                u[i][b]= kernel(Wn[i],Wn[j],sigma)
                b=b+1
    return u


# In[ ]:


def kernel(a,b,sigma):
    return((1/((2*3.141)**.5)/sigma)*np.exp((-np.linalg.norm(a-b)/(2*sigma**2))));


# In[ ]:


def ctpca(W,L1,sigma):
    n1 = len(W) # n1
    W1 = red(W,n1,sigma) #n1 * (n1-1)
    model= PCA(n_components=L1); #
    W2 = model.fit_transform(W1.T); # (n1-1)*L1
    T=np.dot((np.linalg.pinv(W1.T)),W2); # (n1*L1)
    return T;


# In[ ]:


def regression_matrix(input_array,input_hidden_weights,bias):
    input_array = np.array(input_array);
    input_hidden_weights = np.array(input_hidden_weights);
    bias = np.array(bias);
    regression_matrix = np.add(np.dot(input_array,input_hidden_weights),bias);
    return regression_matrix;



# In[ ]:


# Finding hidden layer activations
def hidden_layer_matrix(regression_matrix):
    sigmoidal = [[0.0 for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_inputs)];
    for i in range(0,no_of_inputs):
        for j in range(0,no_of_hidden_neurons):
            sigmoidal[i][j] = (1.0)/(1+math.exp(-(regression_matrix[i][j])))    
    return sigmoidal


# In[ ]:


# Calculating the similarity matrix (S)
def similarity_matrix():
    dist_array = [[0.0 for i in range(0,no_of_inputs)]for j in range(0,no_of_inputs)]
    for i in range(0,no_of_inputs):
        for j in range(0,no_of_inputs):
            for k in range(0,input_dim):
                dist_array[i][j] +=  pow((input_array[i][k] - input_array[j][k]),2);
    
    for i in range(0, no_of_inputs):
        for j in range(0, no_of_inputs):
            dist_array[i][j] = math.exp((-(dist_array[i][j]))/(2*pow(sigma,2.0)));
    return dist_array;


# In[ ]:


# Calculation of Graph Laplacian (L)
def laplacian_matrix(similarity_matrix):
    diagonal_matrix = [[0.0 for i in range(0,no_of_inputs)]for j in range(0,no_of_inputs)];
    diagonal_matrix = np.array(diagonal_matrix);
    similarity_matrix = np.array(similarity_matrix);
    for i in range(0,no_of_inputs):
        for j in range(0,no_of_inputs):
            diagonal_matrix[i][i] += similarity_matrix[i][j];
    
    return np.subtract(diagonal_matrix,similarity_matrix);

