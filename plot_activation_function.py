############################################################################
##               Class for plotting activation functions                  ##
############################################################################

# import libraries
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    ''' It returns 1/(1+exp(-x)). where the values lies between zero and one '''
    return 1/(1+np.exp(-x))

def tanh(x):
    ''' It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1.'''
    return np.tanh(x)

def relu(x):
    ''' It returns zero if the input is less than zero otherwise it returns the given input. '''
    x1=[]
    for i in x:
        if i<0:
            x1.append(0)
        else:
            x1.append(i)

    return x1

def leaky_relu(x):
  data = [max(0.05*value,value) for value in x]
  return np.array(data, dtype=float)

def softmax(x):
    ''' Compute softmax values for each sets of scores in x. '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def main():
    ##Sigmoid
    x = np.linspace(-10, 10)
    plt.plot(x, sigmoid(x))
    plt.axis('tight')
    #plt.title('Activation Function: Sigmoid')
    plt.show()

    ##Tanh
    x = np.linspace(-10, 10)
    plt.plot(x, tanh(x))
    plt.axis('tight')
    #plt.title('Activation Function: Tanh')
    plt.show()

    ##ReLU
    x = np.linspace(-10, 10)
    plt.plot(x, relu(x))
    plt.axis('tight')
    #plt.title('Activation Function: ReLU')
    plt.show()

    ##Leaky ReLU
    x = np.linspace(-10,10,100)
    plt.plot(x, leaky_relu(x))
    plt.axis('tight')
    #plt.title('Activation Function: Leaky ReLU')
    plt.show()

    ##Softmax
    x = np.linspace(-10, 10)
    plt.plot(x, softmax(x))
    plt.axis('tight')
    #plt.title('Activation Function: Softmax')
    plt.show()


if __name__ == "__main__":
    main()
