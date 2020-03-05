import numpy as np
class ZeroRule():
    """
    A baseline classifier that classifies all outputs as the most occurring class in the training data
    """

    # to make the training faster and as the likelihood of a similar distribution of classes,
    # the model is only trained once
    trained = False

    def __init__(self):
        self.best = None

    def fit(self, x, y):
        """
        Trains the classifier to recognize the most occurring output class of the training data
        :param x: x is not used, but kept to keep to same format as sklearn
        :param y: the expected ouput for the training data
        """

        #conform to np array
        y = np.asarray(y)

        #get unique values from y
        try:
            unique = np.unique(y, axis=0)
        except:
            unique = np.unique(y)
        occurances = []

        # check occurances of unique values in y
        for uni in unique:
            occur = 0
            for _y in y:
                if np.array_equal([uni], [_y]):
                    occur += 1
            occurances.append(occur)

        # find the index of best result
        best = 0
        for i in range(len(occurances)):
            if occurances[best] < occurances[i]:
                best = i

        # set the most occuring class as best
        self.best = unique[best]
        self.trained = True

    def predict(self, x):
        """
        uses a trained model to predict outputs
        :param x: the input variable. only used for looping
        :return: returns the predicted values
        """

        assert self.trained == True

        y_pred = []
        for i in x:
            y_pred.append(self.best)
        return np.asarray(y_pred)