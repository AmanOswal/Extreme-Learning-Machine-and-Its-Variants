#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#ELM with ELMAE and PCA
C= 1
c=1

class Gen_ELM22(object):
    
    def __init__(self,sess,batch_size,input_len,hidden_lens,output_len,elmop,Kdash):
            
        
        self.sess = sess 
        self.batch_size = batch_size
        self.input_len = input_len
        self.hidden_lens = hidden_lens
        self.output_len = output_len
        
        #Training variables
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_len])
        self.t = tf.placeholder(tf.float32, [self.batch_size, self.output_len])
        
        #Testing variables 
        self.x_test = tf.placeholder(tf.float32,[None,self.input_len])
        self.y_test = tf.placeholder(tf.float32,[None,self.output_len])
        
        
        #Weight , bias and beta Variable 
        self.w = []
        self.b = []
        self.T = []
        self.rv = []
        self.h = []
        
        self.w.append(tf.Variable(elmop,trainable=False, dtype=tf.float32))
        self.b.append(tf.Variable(tf.random_normal([self.hidden_lens[0]]),trainable=False, dtype=tf.float32))
        self.h.append(tf.sigmoid(tf.matmul(self.x, self.w[0]) + self.b[0]))
        mod = PCA(n_components = Kdash)
        for i in range(1,len(self.hidden_lens)):
            
            
            if (self.input_len < self.hidden_lens[i-1]): # D < L
                
                identity = tf.constant(np.identity(self.hidden_lens[i-1]), dtype=tf.float32)
                betaHL = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(self.h[i-1]), self.h[i-1]) + identity/C),tf.transpose(self.h[i-1])), self.t)
                # beta_a = (H_T*H + I/C)^(-1)*H_T*T
            else:
                
                identity = tf.constant(np.identity(self.batch_size), dtype=tf.float32)
                
                betaHL = tf.matmul(tf.matmul(tf.transpose(self.h[i-1]),tf.matrix_inverse(tf.matmul(self.h[i-1], tf.transpose(self.h[i-1]))+identity/C)), self.t)
                # beta_a = H_T*(H*H_T + I/C)^(-1)*T 
                                   
            mod.fit_transform(tf.transpose(betaHL))
                                   
            self.V.append(tf.Variable(mod.self_components ))
                                   
            self.rv.append(tf.Variable(self.random_normal(self.batch_size,self.hidden_lens[i]-Kdash),trainable=False, dtype=tf.float32))                    
            self.h.append(tf.concat(tf.matmul(h[i-1],V),self.rv[i-1],0))
                                   
            #self.w.append(tf.Variable,trainable=False, dtype=tf.float32))
            #self.b.append(tf.Variable(tf.random_normal([self.hidden_lens[i+1]]),trainable=False, dtype=tf.float32))
        
        self.beta = tf.Variable(tf.zeros([self.hidden_lens[-1], self.output_len]),trainable=False, dtype=tf.float32)
        
        '''self.var_list = []
        for i in range(0,len(self.w)):
            self.var_list.append(self.w[i])
            self.var_list.append(self.b[i])
        self.var_list.append(self.beta)
        
        #Training hidden outputs
        self.h = []
        self.h_t = []
        
        self.h.append(tf.sigmoid(tf.matmul(self.x, self.w[0]) + self.b[0]))
        self.h_t.append(tf.transpose(self.h[0]))
        
        for i in range(1,len(self.w)):
            self.h.append(tf.sigmoid(tf.matmul(self.h[i-1], self.w[i]) + self.b[i]))
            self.h_t.append(tf.transpose(self.h[i]))'''
            
        
        
        #Testing hidden outputs
        
        self.h_test = []
        #self.h_test_t = []
        
        self.h_test.append(tf.sigmoid(tf.matmul(self.x_test, self.w[0]) + self.b[0]))
        
        for i in range(0,len(self.hidden_lens)-1):
            k = np.dot(self.hidden_lens[i],self.V[i])
            self.h_test.append(tf.concat((k , self.rv[i]),0))
            
        #Finding beta
        
        if (self.input_len < self.hidden_lens[-1]):  # D < L
            identity = tf.constant(np.identity(self.hidden_lens[-1]), dtype=tf.float32)
            self.beta_a = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(self.h[-1]), self.h[-1]) + identity/C),tf.transpose(self.h[-1])), self.t)
            # beta_a = (H_T*H + I/C)^(-1)*H_T*T
        else:
            identity = tf.constant(np.identity(self.batch_size), dtype=tf.float32)   
            #self.beta_a = tf.matmul(tf.matmul(tf.transpose(self.h[-1]),tf.matrix_inverse(tf.matmul(self.h[-1], tf.transpose(self.h[-1]))+identity/C), self.t))
            identity = tf.constant(np.identity(self.hidden_lens[-1]), dtype=tf.float32)
            self.beta_a = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(self.h[-1]), self.h[-1]) + identity/C),tf.transpose(self.h[-1])), self.t)
            # beta_a = H_T*(H*H_T + I/C)^(-1)*T
        
        
        self.assign_beta = self.beta.assign(self.beta_a)
        self.out = tf.sigmoid(tf.matmul(self.h[-1], self.beta))
        self.out_test = tf.sigmoid(tf.matmul(self.h_test[-1], self.beta))
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.out, labels=self.t))

        self._init = False
        self._feed = False

        # for the mnist test
        self.correct_prediction = tf.equal(tf.argmax(self.out_test,1), tf.argmax(self.y_test,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.var_list = []
        self.var_list.append(self.w[0])
        self.var_list.append(self.b[0])
        '''for i in range(0,len(self.h)):
            self.var_list.append(self.h[i])
            if(i!=(len(self.h)-1))
                self.var_list.append(self.V[i])'''
        self.var_list.append(self.beta)
                                    
    def feed(self, x, t):
        
        if not self._init : self.init()
        self.sess.run(self.assign_beta, {self.x:x, self.t:t})
        self._feed = True

    def init(self):
        
        self.sess.run(tf.initialize_variables(self.var_list))
        self._init = True

    def test(self, x, t=None):
        
        if not self._feed : exit("Not feed-forward trained")
        
        
        if t is not None :
            print("Accuracy: {:.9f}".format(self.sess.run(self.accuracy, {self.x_test:x, self.y_test:t})))
        else :
            return self.sess.run(self.out_test, {self.x_test:x})


# In[ ]:


# 1 -beta_a = H_T*(H*H_T + I/C)^(-1)*T should have been used but
# 2- beta_a = (H_T*H + I/C)^(-1)*H_T*T was used because of memory error due to
#very large identity matrix in 1
def elmae(X,C,n1):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        
        w1= tf.random_normal([784,n1])
        b1 = tf.random_normal([1,n1])
        H = tf.sigmoid(tf.matmul(X,w1)+b1)
        I = tf.constant(np.identity(125), dtype=tf.float32)
        w = tf.transpose(tf.matmul(tf.matmul(tf.matrix_inverse(tf.add(tf.matmul( tf.transpose(H), H) , I/C)),tf.transpose(H)), X))
        #sess.run(init)
        return w


# In[ ]:


elmop = elmae(train_x , 1 , 125)


# In[ ]:


batch_size = 5000
hidden_num = [125]
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
elm = Gen_ELM22(sess, batch_size, 784, hidden_num, 10,elmop,20)

# one-step feed-forward training
train_x, train_y = mnist.train.next_batch(batch_size)
elm.feed(train_x, train_y)

# testing
elm.test(mnist.test.images, mnist.test.labels)


# In[ ]:


identity = tf.constant(np.identity(100), dtype=tf.float32)   
a = tf.Variable(tf.random_normal([100,2]))
be = tf.matmul(tf.transpose(a),tf.matrix_inverse(tf.matmul(a, tf.transpose(a))+identity/C))


# In[ ]:


#ELM with ELMAE
#Same as Gen_ELM22(function above). Variable initialization method changed
C= 1
c=1

class Gen_ELM022(object):
    
    def __init__(self,sess,batch_size,input_len,hidden_lens,output_len,elmop,Kdash):
        
        
        self.sess = sess 
        self.batch_size = batch_size
        self.input_len = input_len
        self.hidden_lens = hidden_lens
        self.output_len = output_len
        
        #Training variables
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_len])
        self.t = tf.placeholder(tf.float32, [self.batch_size, self.output_len])
        
        #Testing variables 
        self.x_test = tf.placeholder(tf.float32,[None,self.input_len])
        self.y_test = tf.placeholder(tf.float32,[None,self.output_len])
        
        
        #Weight , bias and beta Variable 
        self.w = []
        self.b = []
        self.T = []
        self.rv = []
        self.h = []
        
        self.w.append(tf.Variable(elmop,trainable=False, dtype=tf.float32))
        self.b.append(tf.Variable(tf.random_normal([self.hidden_lens[0]]),trainable=False, dtype=tf.float32))
        self.h.append(tf.sigmoid(tf.matmul(self.x, self.w[0]) + self.b[0]))
        mod = PCA(n_components = Kdash)
        for i in range(1,len(self.hidden_lens)):
            
            
            if (self.input_len < self.hidden_lens[i-1]): # D < L
                
                identity = tf.constant(np.identity(self.hidden_lens[i-1]), dtype=tf.float32)
                betaHL = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(self.h[i-1]), self.h[i-1]) + identity/C),tf.transpose(self.h[i-1])), self.t)
                # beta_a = (H_T*H + I/C)^(-1)*H_T*T
            else:
                
                identity = tf.constant(np.identity(self.batch_size), dtype=tf.float32)
                
                betaHL = tf.matmul(tf.matmul(tf.transpose(self.h[i-1]),tf.matrix_inverse(tf.matmul(self.h[i-1], tf.transpose(self.h[i-1]))+identity/C)), self.t)
                # beta_a = H_T*(H*H_T + I/C)^(-1)*T 
                                   
            mod.fit_transform(tf.transpose(betaHL))
                                   
            self.V.append(mod.self_components )
                                   
            self.rv.append(self.random_normal(self.batch_size,self.hidden_lens[i]-Kdash),trainable=False, dtype=tf.float32)                    
            self.h.append(tf.concat(tf.matmul(h[i-1],V),self.rv[i-1],0))
                                   
            #self.w.append(tf.Variable,trainable=False, dtype=tf.float32))
            #self.b.append(tf.Variable(tf.random_normal([self.hidden_lens[i+1]]),trainable=False, dtype=tf.float32))
        
        self.beta = tf.Variable(tf.zeros([self.hidden_lens[-1], self.output_len]),trainable=False, dtype=tf.float32)
        
        '''self.var_list = []
        for i in range(0,len(self.w)):
            self.var_list.append(self.w[i])
            self.var_list.append(self.b[i])
        self.var_list.append(self.beta)
        
        #Training hidden outputs
        self.h = []
        self.h_t = []
        
        self.h.append(tf.sigmoid(tf.matmul(self.x, self.w[0]) + self.b[0]))
        self.h_t.append(tf.transpose(self.h[0]))
        
        for i in range(1,len(self.w)):
            self.h.append(tf.sigmoid(tf.matmul(self.h[i-1], self.w[i]) + self.b[i]))
            self.h_t.append(tf.transpose(self.h[i]))'''
            
        
        
        #Testing hidden outputs
        
        self.h_test = []
        #self.h_test_t = []
        
        self.h_test.append(tf.sigmoid(tf.matmul(self.x_test, self.w[0]) + self.b[0]))
        
        for i in range(0,len(self.hidden_lens)-1):
            k = np.dot(self.hidden_lens[i],self.V[i])
            self.h_test.append(tf.concat((k , self.rv[i]),0))
            
        #Finding beta
        
        if (self.input_len < self.hidden_lens[-1]):  # D < L
            identity = tf.constant(np.identity(self.hidden_lens[-1]), dtype=tf.float32)
            self.beta_a = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(self.h[-1]), self.h[-1]) + identity/C),tf.transpose(self.h[-1])), self.t)
            # beta_a = (H_T*H + I/C)^(-1)*H_T*T
        else:
            identity = tf.constant(np.identity(self.batch_size), dtype=tf.float32)   
            #self.beta_a = tf.matmul(tf.matmul(tf.transpose(self.h[-1]),tf.matrix_inverse(tf.matmul(self.h[-1], tf.transpose(self.h[-1]))+identity/C), self.t))
            identity = tf.constant(np.identity(self.hidden_lens[-1]), dtype=tf.float32)
            self.beta_a = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(self.h[-1]), self.h[-1]) + identity/C),tf.transpose(self.h[-1])), self.t)
            # beta_a = H_T*(H*H_T + I/C)^(-1)*T
        
        
        self.assign_beta = self.beta.assign(self.beta_a)
        self.out = tf.sigmoid(tf.matmul(self.h[-1], self.beta))
        self.out_test = tf.sigmoid(tf.matmul(self.h_test[-1], self.beta))
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.out, labels=self.t))

        self._init = False
        self._feed = False

        # for the mnist test
        self.correct_prediction = tf.equal(tf.argmax(self.out_test,1), tf.argmax(self.y_test,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.var_list = []
        self.var_list.append(self.w[0])
        self.var_list.append(self.b[0])
        '''for i in range(0,len(self.h)):
            self.var_list.append(self.h[i])
            if(i!=(len(self.h)-1))
                self.var_list.append(self.V[i])'''
        self.var_list.append(self.beta)
                                    
    def feed(self, x, t):
        
        if not self._init : self.init()
        self.sess.run(self.assign_beta, {self.x:x, self.t:t})
        self._feed = True

    def init(self):
        
        self.sess.run(tf.initialize_variables(self.var_list))
        self._init = True

    def test(self, x, t=None):
        
        if not self._feed : exit("Not feed-forward trained")
        
        
        if t is not None :
            print("Accuracy: {:.9f}".format(self.sess.run(self.accuracy, {self.x_test:x, self.y_test:t})))
        else :
            return self.sess.run(self.out_test, {self.x_test:x})


# In[ ]:


batch_size = 5000
hidden_num = [125]
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
elm = Gen_ELM022(sess, batch_size, 784, hidden_num, 10,elmop,20)

# one-step feed-forward training
train_x, train_y = mnist.train.next_batch(batch_size)
elm.feed(train_x, train_y)

# testing
elm.test(mnist.test.images, mnist.test.labels)

