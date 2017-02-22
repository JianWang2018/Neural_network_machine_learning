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
            w=np.random.randn(3,1)
        else:
            w=w_init

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
    path_load="/media/jianwang/Study/data/load/neural_network"
    np.random.seed(987612345)

    #data is in dict form
    data = scipy.io.loadmat(path_load+'/dataset1.mat')
    w=NeuralNetwork().learn_perceptron(data['neg_examples_nobias'],data['pos_examples_nobias'],data['w_init'],data['w_gen_feas'],iter_max=20)
    print(w)

if __name__=="__main__":
    main()
