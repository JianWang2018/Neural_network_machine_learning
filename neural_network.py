import matplotlib.pylab as plt
import numpy as np
import time
import scipy.io


class NeuralNetwork(object):
    def eval_perceptron(self,neg_examples, pos_examples, w):
        """
        Evaluates the perceptron using a given weight vector. Here, evaluation
        refers to finding the data points that the perceptron incorrectly classifies.
        :param neg_examples:The num_neg_examples x 3 matrix for the examples with target 0.
        num_neg_examples is the number of examples for the negative class.
        :param pos_examples:The num_pos_examples x 3 matrix for the examples with target 1.
        num_pos_examples is the number of examples for the positive class.
        :param w:A 3-dimensional weight vector, the last element is the bias.
        :return:
        mistakes0:A vector containing the indices of the negative examples that have been
        incorrectly classified as positive.
        mistakes1:A vector containing the indices of the positive examples that have been
        incorrectly classified as negative.
        """

        num_neg_examples = neg_examples.shape[0]
        num_pos_examples = pos_examples.shape[0]
        mistakes0 = []
        mistakes1 = []
        for i in range(num_neg_examples):
            x = neg_examples[i,:]
            activation = np.dot(x,w)
            if (activation >= 0):
                mistakes0.append(i)


        for i in range(num_pos_examples):
            x = pos_examples[i,:]
            activation = np.dot(x,w)
            if (activation < 0):
                mistakes1.append(i)
        return mistakes0,mistakes1

    def update_weights(self,neg_examples, pos_examples, w_current):
        """
        Updates the weights of the perceptron for incorrectly classified points
        using the perceptron update algorithm. This function makes one sweep
        over the dataset.
        :param neg_examples: The num_neg_examples x 3 matrix for the examples with target 0.
               num_neg_examples is the number of examples for the negative class.
        :param pos_examples: The num_pos_examples x 3 matrix for the examples with target 1.
               num_pos_examples is the number of examples for the positive class.
        :param w_current:  A 3-dimensional weight vector, the last element is the bias.
        :return: w :The weight vector after one pass through the dataset using the perceptron
              learning rule.
        """

        w = np.array(w_current).reshape(np.size(w_current),)
        num_neg_examples = np.array(neg_examples).shape[0]
        num_pos_examples = np.array(pos_examples).shape[0]
        for i in range(num_neg_examples):
            this_case = neg_examples[i,:]
            activation = np.dot(this_case,w)
            if (activation >= 0):

                # update the weights,(0-1) so minus
                w = w-this_case


        for i in range(num_pos_examples):
            this_case = pos_examples[i,:]
            activation = np.dot(this_case,w)
            if (activation < 0):

                #update the weights, (1-0) so +
                w=w+this_case

        return w

    def learn_perceptron(self,neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas,iter_max=1000):
        """
        Learns the weights of a perceptron for a 2-dimensional dataset and plots
        the perceptron at each iteration where an iteration is defined as one
        full pass through the data. If a generously feasible weight vector
        is provided then the visualization will also show the distance
        of the learned weight vectors to the generously feasible weight vector.
        :param neg_examples_nobias:The num_neg_examples x 2 matrix for the examples with target 0
                                   num_neg_examples is the number of examples for the negative class.
        :param pos_examples_nobias:The num_pos_examples x 2 matrix for the examples with target 1.
                                   num_pos_examples is the number of examples for the positive class.
        :param w_init:A 3-dimensional initial weight vector. The last element is the bias.
        :param w_gen_feas:A generously feasible weight vector.
        :return:w - The learned weight vector.
        """

        #bookkeeping
        num_neg_examples=neg_examples_nobias.shape[0]
        num_pos_examples=pos_examples_nobias.shape[0]
        num_err_history=[]
        w_dist_history=[]

        # we add a column of ones to the examples in order to allow us to learn bias parameters
        neg_examples=np.append(neg_examples_nobias,np.ones([num_neg_examples,1]),axis=1)
        pos_examples=np.append(pos_examples_nobias,np.ones([num_pos_examples,1]),axis=1)

        # if weight vectors have not been provided, initialize them appropriately
        if not np.size(w_init):
            w = np.random.randn(3,1)
        else:
            w = w_init

        #find the data points that the perceptron has incorrectly classified and record the number of errors
        #it makes


        iter=0
        mistakes0,mistakes1=self.eval_perceptron(neg_examples,pos_examples,w)
        num_errs=np.array(mistakes0).shape[0]+np.array(mistakes1).shape[0]
        num_err_history.append(num_errs)

        #If a generously feasible weight vector exists, record the distance
        #to it from the initial weight vector.
        if (len(w_gen_feas) != 0):
            w_dist_history.append(np.linalg.norm(w - w_gen_feas))


        #Iterate until the perceptron has correctly classified all points.
        while (num_errs > 0 and iter<iter_max):
            iter = iter + 1

            #Update the weights of the perceptron.
            w = self.update_weights(neg_examples, pos_examples, w)

            #If a generously feasible weight vector exists, record the distance
            #to it from the current weight vector.
            if (len(w_gen_feas) != 0):
                w_dist_history.append(np.linalg.norm(w - w_gen_feas))


            #Find the data points that the perceptron has incorrectly classified.
            #and record the number of errors it makes.
            mistakes0, mistakes1 = self.eval_perceptron(neg_examples,pos_examples,w)
            num_errs=np.array(mistakes0).shape[0]+np.array(mistakes1).shape[0]
            num_err_history.append(num_errs)
            print('Number of errors in iteration %d:\t%d\n'%(iter,num_errs))
            print('weights:\t', w, '\n')

        self.plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history)


        if iter == iter_max:
            print("iteration reaches the maximum numbers and stop loop")
        return w

    def load_data(self,N,data_name):
        """
        This method loads the training, validation and test set.
        It also divides the training set into mini-batches.
        :param N:Mini-batch size
        :param data_name: data file name
        :return:
        train_input: An array of size D X N X M, where
                     D: number of input dimensions (in this case, 3).
                     N: size of each mini-batch (in this case, 100).
                    M: number of minibatches.
        train_target: An array of size 1 X N X M.
        valid_input: An array of size D X number of points in the validation set.
        test: An array of size D X number of points in the test set.
        vocab: Vocabulary containing index to word mapping.
        """
        data=scipy.io.loadmat("./data/"+data_name)
        numdims=data["data"][0][0]["trainData"].shape[0]
        D=numdims-1
        M=int(data["data"][0][0]["trainData"].shape[1]/N)
        train_input=data["data"][0][0]["trainData"][:D, :(N*M)].reshape(D,N,M)
        train_target=data["data"][0][0]["trainData"][D, :(N*M)].reshape(1,N,M)
        valid_input=data["data"][0][0]["validData"][:D,:]
        valid_target=data["data"][0][0]["validData"][D,:]
        test_input=data["data"][0][0]["testData"][:D,:]
        test_target=data["data"][0][0]["testData"][D,:]
        vocab=data["data"][0][0]["vocab"]

        return train_input, train_target,valid_input,valid_target,test_input,test_target,vocab

    def fprop(self,input_batch, word_embedding_weights, embed_to_hid_weights,hid_to_output_weights, hid_bias, output_bias):
        """
        This method forward propagates through a neural network.
        :param input_batch:The input data as a matrix of size numwords X batchsize where,numwords is the number of words,
        batchsize is the number of data points.So,\
        if input_batch(i, j) = k then the ith word in data point j is wordindex k of the vocabulary.
        :param word_embedding_weights: Word embedding as a matrix of size vocab_size X numhid1, where vocab_size is the
        size of the vocabulary, numhid1 is the dimensionality of the embedding space
        :param embed_to_hid_weights:Weights between the word embedding layer and hidden layer as a matrix of size
        numhid1*numwords X numhid2, numhid2 is the number of hidden units.
        :param hid_to_output_weights:Weights between the hidden layer and output softmax unit as a matrix of size
        numhid2 X vocab_size
        :param hid_bias:Bias of the hidden layer as a matrix of size numhid2 X 1.
        :param output_bias:Bias of the output layer as a matrix of size vocab_size X 1.
        :return:
        embedding_layer_state: State of units in the embedding layer as a matrix of size numhid1*numwords X batchsize
        hidden_layer_state: State of units in the hidden layey as a matrix of size numhid2 X batchsize
        output_layer_state: State of units in the output layer as a matrix of size vocab_size X batchsize
        """

        numwords, batchsize=input_batch.shape
        vocab_size, numhid1=word_embedding_weights.shape
        numhid2=embed_to_hid_weights.shape[1]

        # COMPUTE STATE OF WORD EMBEDDING LAYER.
        # Look up the inputs word indices in the word_embedding_weights matrix.
        # -1 means that this dimension will depend on the other dimension when reshape
        # need to minus one because python start with the index 0
        embedding_layer_state=word_embedding_weights[input_batch.reshape(1,-1)-1,:].T.reshape(numhid1*numwords,-1)

        # COMPUTE STATE OF HIDDEN LAYER.
        # Compute inputs to hidden units.
        # tile here is same with repmat in matlab
        input_to_hidden_units=np.dot(embed_to_hid_weights.T,embedding_layer_state)+np.tile(hid_bias,(1,batchsize))

        # Apply logistic activation function.
        # FILL IN CODE. Replace the line below by one of the options.
        hidden_layer_state = np.zeros((numhid2, batchsize))
        # % Options
        # % (a) hidden_layer_state = 1 ./ (1 + exp(inputs_to_hidden_units));
        # % (b) hidden_layer_state = 1 ./ (1 - exp(-inputs_to_hidden_units));
        hidden_layer_state = 1 / (1 + np.exp(-input_to_hidden_units))
        # % (d) hidden_layer_state = -1 ./ (1 + exp(-inputs_to_hidden_units));
        inputs_to_softmax = np.zeros((vocab_size, batchsize))
        #Options
        inputs_to_softmax = np.dot(hid_to_output_weights.T, hidden_layer_state) +  np.tile(output_bias, (1, batchsize))
        # % (b) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, batchsize, 1);
        # % (c) inputs_to_softmax = hidden_layer_state * hid_to_output_weights' +  repmat(output_bias, 1, batchsize);
        # % (d) inputs_to_softmax = hid_to_output_weights * hidden_layer_state +  repmat(output_bias, batchsize, 1);


        # Subtract maximum.
        # Remember that adding or subtracting the same constant from each input to a
        # softmax unit does not affect the outputs. Here we are subtracting maximum to
        # make all inputs <= 0. This prevents overflows when computing their
        # exponents.
        inputs_to_softmax = inputs_to_softmax - np.tile(np.max(inputs_to_softmax), (vocab_size, 1))

        # Compute exp. since all the numbers are negative. so it will not over flow

        output_layer_state = np.exp(inputs_to_softmax)

        #Normalize to get probability distribution.
        # sum the value by the column and repeat it with the vocab_size
        output_layer_state = output_layer_state/np.tile(np.sum(output_layer_state, 0), (vocab_size, 1))
        return embedding_layer_state, hidden_layer_state, output_layer_state

    def train(self,epochs,batchsize = 100,learning_rate = 0.1,momentum = 0.9,numhid1 = 50,numhid2 = 200 ,init_wt = 0.01,
              show_training_CE_after = 100,show_validation_CE_after = 1000):
        """
        This function trains a neural network language model
        :param epochs: Number of epochs to run.
        :param batchsize: Mini-batch size.
        :param learning_rate: Learning rate; default = 0.1.
        :param momentum: Momentum; default = 0.9.
        :param numhid1: Dimensionality of embedding space; default = 50.
        :param numhid2: Number of units in hidden layer; default = 200.
        :param init_wt: Standard deviation of the normal distribution
                         # which is sampled to get the initial weights; default = 0.01
        :param show_training_CE_after:
        :param show_validation_CE_after:
        :return: model: A struct containing the learned weights and biases and vocabulary.
        """
        batchsize = 100
        learning_rate = 0.1
        momentum = 0.9
        numhid1 = 50
        numhid2 = 200
        init_wt = 0.01

        # LOAD DATA.
        train_input, train_target, valid_input, valid_target,test_input, test_target, vocab = self.load_data(batchsize,"data.mat")
        numwords, batchsize, numbatches =train_input.shape
        vocab_size = vocab.shape[1]

        # INITIALIZE WEIGHTS AND BIASES.
        word_embedding_weights = np.dot(init_wt ,np.random.randn(vocab_size, numhid1))
        embed_to_hid_weights = np.dot(init_wt ,np.random.randn(numwords * numhid1, numhid2))
        hid_to_output_weights = np.dot(init_wt ,np.random.randn(numhid2, vocab_size))
        hid_bias = np.zeros((numhid2, 1))
        output_bias = np.zeros((vocab_size, 1))

        word_embedding_weights_delta = np.zeros((vocab_size, numhid1))
        word_embedding_weights_gradient = np.zeros((vocab_size, numhid1))
        embed_to_hid_weights_delta = np.zeros((numwords * numhid1, numhid2))
        hid_to_output_weights_delta = np.zeros((numhid2, vocab_size))
        hid_bias_delta = np.zeros((numhid2, 1))
        output_bias_delta = np.zeros((vocab_size, 1))
        expansion_matrix = np.eye(vocab_size)
        count = 0
        tiny = np.exp(-30)

        # Train
        for epoch in range(epochs):
            print('Epoch %d\n'%epoch)
            this_chunk_CE = 0
            trainset_CE = 0
            # LOOP OVER MINI-BATCHES.
            for m in range(numbatches):
                # each m here is a batch, which contains numwords* batch_size values
                input_batch = train_input[:, :, m]
                target_batch = train_target[:, :, m]

                # FORWARD PROPAGATE.
                # Compute the state of each layer in the network given the input batch
                # and all weights and biases
                embedding_layer_state, hidden_layer_state, output_layer_state =self.fprop(input_batch,word_embedding_weights, embed_to_hid_weights,
                                                                                     hid_to_output_weights, hid_bias, output_bias)

                # COMPUTE DERIVATIVE.
                # Expand the target to a sparse 1-of-K vector.
                expanded_target_batch = expansion_matrix[:, target_batch]
                # Compute derivative of cross-entropy loss function.
                error_deriv = output_layer_state - expanded_target_batch

                # MEASURE LOSS FUNCTION.
                CE = -np.sum(np.sum(np.dot(expanded_target_batch , np.log(output_layer_state + tiny)))) / batchsize
                count +=  1
                this_chunk_CE +=(CE - this_chunk_CE) / count
                trainset_CE +=(CE - trainset_CE) / m
                print('\rBatch %d Train CE %.3f'% (m, this_chunk_CE))
                if m % show_training_CE_after == 0:
                  print( '\n')
                  count = 0
                  this_chunk_CE = 0


                # BACK PROPAGATE.
                # OUTPUT LAYER.
                hid_to_output_weights_gradient =  hidden_layer_state * error_deriv
                output_bias_gradient = np.sum(error_deriv, axis=1)
                back_propagated_deriv_1 = (hid_to_output_weights * error_deriv) * hidden_layer_state * (1 - hidden_layer_state)

                # HIDDEN LAYER.
                #FILL IN CODE. Replace the line below by one of the options.
                embed_to_hid_weights_gradient = np.zeros(numhid1 * numwords, numhid2);
                # Options:
                # % (a) embed_to_hid_weights_gradient = back_propagated_deriv_1' * embedding_layer_state;
                # % (b) embed_to_hid_weights_gradient = embedding_layer_state * back_propagated_deriv_1';
                # % (c) embed_to_hid_weights_gradient = back_propagated_deriv_1;
                # % (d) embed_to_hid_weights_gradient = embedding_layer_state;

                # FILL IN CODE. Replace the line below by one of the options.
                hid_bias_gradient = np.zeros(numhid2, 1);
                # % Options
                # % (a) hid_bias_gradient = sum(back_propagated_deriv_1, 2);
                # % (b) hid_bias_gradient = sum(back_propagated_deriv_1, 1);n
                # % (c) hid_bias_gradient = back_propagated_deriv_1;
                # % (d) hid_bias_gradient = back_propagated_deriv_1';

                # FILL IN CODE. Replace the line below by one of the options.
                back_propagated_deriv_2 = np.zeros(numhid2, batchsize);
                # % Options
                # % (a) back_propagated_deriv_2 = embed_to_hid_weights * back_propagated_deriv_1;
                # % (b) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights;
                # % (c) back_propagated_deriv_2 = back_propagated_deriv_1' * embed_to_hid_weights;
                # % (d) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights';

                word_embedding_weights_gradient[:] = 0
                # EMBEDDING LAYER.
                for w in range(numwords):
                   word_embedding_weights_gradient = word_embedding_weights_gradient +expansion_matrix[:, input_batch[w, :]] *\
                   (back_propagated_deriv_2[1 + (w - 1) * numhid1 : w * numhid1, :])

                # UPDATE WEIGHTS AND BIASES.
                word_embedding_weights_delta = momentum * word_embedding_weights_delta +word_embedding_weights_gradient / batchsize
                word_embedding_weights = word_embedding_weights- learning_rate * word_embedding_weights_delta

                embed_to_hid_weights_delta = momentum * embed_to_hid_weights_delta +embed_to_hid_weights_gradient / batchsize
                embed_to_hid_weights = embed_to_hid_weights- learning_rate * embed_to_hid_weights_delta

                hid_to_output_weights_delta = momentum * hid_to_output_weights_delta + hid_to_output_weights_gradient / batchsize
                hid_to_output_weights = hid_to_output_weights- learning_rate * hid_to_output_weights_delta

                hid_bias_delta = momentum * hid_bias_delta +hid_bias_gradient / batchsize
                hid_bias = hid_bias - learning_rate * hid_bias_delta

                output_bias_delta = momentum * output_bias_delta + output_bias_gradient / batchsize;
                output_bias = output_bias - learning_rate * output_bias_delta;

                # VALIDATE.
                if (m % show_validation_CE_after) == 0:
                  print('\rRunning validation ...')

                  embedding_layer_state, hidden_layer_state, output_layer_state =self.fprop(valid_input, word_embedding_weights, embed_to_hid_weights,
                                                                                            hid_to_output_weights, hid_bias, output_bias)
                  datasetsize = valid_input.shape[1]
                  expanded_valid_target = expansion_matrix[:, valid_target]
                  CE = -sum(sum(expanded_valid_target * np.log(output_layer_state + tiny))) /datasetsize
                  print(' Validation CE %.3f\n' % CE)

            print('\rAverage Training CE %.3f\n' % trainset_CE)




    def plot_perceptron(self,neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history):
        """
        The top-left plot shows the dataset and the classification boundary given by
        the weights of the perceptron. The negative examples are shown as circles
        while the positive examples are shown as squares. If an example is colored
        green then it means that the example has been correctly classified by the
        provided weights. If it is colored red then it has been incorrectly classified.
        The top-right plot shows the number of mistakes the perceptron algorithm has
        made in each iteration so far.
        The bottom-left plot shows the distance to some generously feasible weight
        vector if one has been provided (note, there can be an infinite number of these).
        Points that the classifier has made a mistake on are shown in red,
        while points that are correctly classified are shown in green.
        The goal is for all of the points to be green (if it is possible to do so).
        :param neg_examples:The num_neg_examples x 3 matrix for the examples with target 0.
        num_neg_examples is the number of examples for the negative class.
        :param pos_examples:The num_pos_examples x 3 matrix for the examples with target 1.
        num_pos_examples is the number of examples for the positive class.
        :param mistakes0:A vector containing the indices of the datapoints from class 0 incorrectly
        classified by the perceptron. This is a subset of neg_examples.
        :param mistakes1:A vector containing the indices of the datapoints from class 1 incorrectly
        classified by the perceptron. This is a subset of pos_examples.
        :param num_err_history:A vector containing the number of mistakes for each
        iteration of learning so far.
        :param w:A 3-dimensional vector corresponding to the current weights of the
        perceptron. The last element is the bias.
        :param w_dist_history:A vector containing the L2-distance to a generously
        feasible weight vector for each iteration of learning so far.
        :return: no return
        """

        # find the correct classification index of positive and negative class
        neg_correct_ind=list(set(range(neg_examples.shape[0]))-set(mistakes0))
        pos_correct_ind=list(set(range(pos_examples.shape[0]))-set(mistakes1))

        plt.figure()                # the first figure
        plt.suptitle("Classification with the perceptron learning algorithm")
        plt.subplot(131)             # the first subplot in the first figure
        if (np.size(neg_examples)):
            plt.plot(neg_examples[neg_correct_ind,0],neg_examples[neg_correct_ind,1],'og',markersize=20)
        if (np.size(pos_examples)):
	        plt.plot(pos_examples[pos_correct_ind,0],pos_examples[pos_correct_ind,1],'sg',markersize=20)

        if (np.size(mistakes0) > 0):
	        plt.plot(neg_examples[mistakes0,0],neg_examples[mistakes0,1],'or',markersize=20)

        if (np.size(mistakes1) > 0):
	        plt.plot(pos_examples[mistakes1,0],pos_examples[mistakes1,1],'sr',markersize=20)

        plt.title("Classifier")

        #In order to plot the decision line, we just need to get two points.
        plt.plot([-5,5],[(-w[-1]+5*w[0])/w[1],(-w[-1]-5*w[0])/w[1]],'k')
        plt.xlim([-1,1])
        plt.ylim([-1,1])

        # number of error
        plt.subplot(132)
        plt.bar(list(range(len(num_err_history))),num_err_history)
        plt.xlim([-1,max(15,len(num_err_history))])
        plt.ylim([0,neg_examples.shape[0]+pos_examples.shape[0]+1])
        plt.title('Number of errors')
        plt.xlabel('Iteration')
        plt.ylabel('Number of errors')
        # distance between given weights and computing weights
        plt.subplot(133);
        plt.bar(list(range(len(w_dist_history))),w_dist_history)
        plt.xlim([-1,max(15,len(num_err_history))])
        plt.ylim([0,15])
        plt.title('Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')

        plt.show()


def main():
    # next part for project 1
    # path_load="/media/jianwang/Study/data/load/neural_network"
    # np.random.seed(987612345)
    nn=NeuralNetwork()

    # #data is in dict form
    # data = scipy.io.loadmat(path_load+'/dataset1.mat')
    # w=nn.learn_perceptron(data['neg_examples_nobias'],data['pos_examples_nobias'],data['w_init'],data['w_gen_feas'],iter_max=20)
    # print(w)

    #-------------------------------------------------------------
    # next part for project 2
    #--------------------------------------------------------------

    import os
    cwd = os.getcwd()
    data = scipy.io.loadmat(cwd+"/data/data.mat")
    # print the shape of three data set

    print(data["data"][0][0]["trainData"].shape)
    print(data["data"][0][0]["testData"].shape)
    print(data["data"][0][0]["validData"].shape)

    #load data with mini-batch
    train_input, train_target,valid_input,valid_target,test_input,test_target,vocab=nn.load_data(100,"data.mat")
    batchsize = 100
    learning_rate = 0.1
    momentum = 0.9
    numhid1 = 50
    numhid2 = 200
    init_wt = 0.01

    # LOAD DATA.
    train_input, train_target, valid_input, valid_target,test_input, test_target, vocab = nn.load_data(batchsize,"data.mat")
    numwords, batchsize, numbatches =train_input.shape
    vocab_size = vocab.shape[1]


    # INITIALIZE WEIGHTS AND BIASES.
    word_embedding_weights = init_wt * np.random.randn(vocab_size, numhid1)
    embed_to_hid_weights = init_wt * np.random.randn(numwords * numhid1, numhid2)
    hid_to_output_weights = init_wt * np.random.randn(numhid2, vocab_size)
    hid_bias = np.zeros((numhid2, 1))
    output_bias = np.zeros((vocab_size, 1))

    word_embedding_weights_delta = np.zeros((vocab_size, numhid1))
    word_embedding_weights_gradient = np.zeros((vocab_size, numhid1))
    embed_to_hid_weights_delta = np.zeros((numwords * numhid1, numhid2))
    hid_to_output_weights_delta = np.zeros((numhid2, vocab_size))
    hid_bias_delta = np.zeros((numhid2, 1))
    output_bias_delta = np.zeros((vocab_size, 1))
    expansion_matrix = np.eye(vocab_size)
    count = 0
    tiny = np.exp(-30)

    input_batch = train_input[:, :, 100]
    target_batch = train_target[:, :, 100]

    # FORWARD PROPAGATE.
    # Compute the state of each layer in the network given the input batch
    # and all weights and biases
    embedding_layer_state, hidden_layer_state, output_layer_state =nn.fprop(input_batch,word_embedding_weights, embed_to_hid_weights,
                                                                         hid_to_output_weights, hid_bias, output_bias)


    # test if the shape of load data is right
    nn.train(1)
    print(train_input.shape)









if __name__=="__main__":
    main()
