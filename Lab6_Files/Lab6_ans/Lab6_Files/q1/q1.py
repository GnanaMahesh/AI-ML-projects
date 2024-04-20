import numpy as np
import random
# Don't import any other libraries here

# Set seed
def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)

def gini_index(classes):
    """
    Calculate the gini index of a set of labels. 

    This is a measure of how often a randomly chosen element
    drawn from the class vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector. THINK!

    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        classes (numpy.ndarray): numpy array of labels
    
    Returns:
        float: A float value representing the gini index of the set of labels.   
    """
    #### Student Code Start : TASK 1
    gini_impurity_value = None

    # Get unique labels and their counts in the target dataset
    total = len(classes)
    label,count = np.unique(classes,return_counts=True)
    # Calculate the probabilities of each unique label occurring
    count = count/total
    # Calculate the gini index value using the formula: 1 - Σ(p_i^2)
    gini_impurity_value = 1 - np.sum(count **2)
    
    return gini_impurity_value
    #### Student Code End

def entropy(classes):
    """
    Calculate the entropy of a set of labels.

    Args:
        classes (numpy.ndarray): numpy array of labels 
    
    Returns:
        float: A float value representing the entropy of the set of labels.  
    """
    #### Student Code Start : TASK 1
    entropy_value = None

    # Get unique labels and their counts in the target dataset
    total = len(classes)
    label,count = np.unique(classes,return_counts=True)
    # Calculate the probabilities of each unique label occurring
    count = count/total
    count = count + 1e-12
    # Calculate the entropy value using the formula: -Σ(p_i * log2(p_i))
    entropy_value = -np.sum(count * np.log2(count))
    # Add a small constant (1e-12) to prevent logarithm of zero
    
    return entropy_value
    #### Student Code End



def information_gain(examples, classes, attr):
    """
    Calculate the information gain from splitting the dataset on a given attribute.

    Args:
        examples (numpy.ndarray): A numpy array of shape containing input examples.
        classes (numpy.ndarray): A numpy array of shape (number_of_examples) containing target labels.
        attr (int): An integer specifying the attribute to split on, ranging from 0 to (number_of_features - 1) inclusive.
    
    Returns:
        float: The information gain obtained from splitting the dataset on the specified attribute.
    """

    #### Student Code Start : TASK 2
    information_gain_value = None
    
    # Get unique values and their counts for the specified attribute
    col = examples[:, attr] 
    label,count = np.unique(col,return_counts=True)
    count = count/np.sum(count)
    # Calculate the entropy of the entire dataset before splitting
    entropy1= entropy(classes)
    ## Calculate the entropy after splitting for all unique values of the attribute
    entropy2 =0
    for i in range(len(label)) :
        indices = np.where(col == label[i])[0]
        filtered_entries = classes[indices]
        entropy2 += entropy(filtered_entries) * count[i]
    
    # Duplicate the classes vertically, to match each example
    # Calculate the weighted sum of entropies for each attribute value

    information_gain_value = entropy1 - entropy2
    # Calculate and return the information gain

    return information_gain_value

    #### Student Code End


def gini_gain(example, classes, attr):
    """
    Calculate the gini gain from splitting the dataset on a given attribute.

    Args:
        example (numpy.ndarray): A numpy array of shape containing input examples.
        classes (numpy.ndarray): A numpy array of shape (number_of_examples) containing target labels.
        attr (int): An integer specifying the attribute to split on, ranging from 0 to (number_of_features - 1) inclusive.
    
    Returns:
        float: The gini gain obtained from splitting the dataset on the specified attribute.
    """

    #### Student Code Start : TASK 3
    gini_gain_value = None

    # Get unique values and their counts for the specified attribute
    col = example[:, attr] 
    label,count = np.unique(col,return_counts=True)
    count = count/np.sum(count)
    # Calculate the gini index of the entire dataset before splitting
    gini1= gini_index(classes)
    ## Calculate the gini index after splitting for all unique values of the attribute
    # For each unique value of the attribute, calculate the gini index of the split, and add it to the weighted sum
    gini2 =0
    for i in range(len(label)) :
        indices = np.where(col == label[i])
        filtered_entries = classes[indices]
        gini2 += gini_index(filtered_entries) * count[i]
    # Calculate and return the gini gain
    gini_gain_value = gini1 - gini2
    return gini_gain_value
    #### Student Code End



def get_precision(model, input, target):
    """Given the model, data and classes, calculates precision."""

    correct_preds = 0
    for index in range(len(input)):
        correct_preds += (model.predict(input[index]) == target[index])
    precision = correct_preds / len(input)
    return precision


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, default, attr_to_split = -1, children = {}):
        """
        Initialize a DecisionNode object.

        Args:
            default(int): The default prediction value for this node.
            attr_to_split(int): The index of the attribute to split on (-1 for leaf nodes). The one the node is split on.
            children(dict): A dictionary containing child nodes.
        """

        self.attr_to_split = attr_to_split
        self.children = children
        self.default = default
    
    def predict(self, attributes):
        """
        Predict the outcome based on the input attributes.

        Args:
            attributes: A list of attribute values to make a prediction.

        Returns:
            The predicted outcome.
        """

        # If this node is a leaf node (no attribute to split on), return the default prediction.
        if (self.attr_to_split == -1):
            return self.default
        
        # Retrieve the value of the splitting attribute for this example.
        key = attributes[self.attr_to_split]

        # Check if there is a child node associated with the attribute value.
        if (key in self.children):
            # Recursively call the prediction for the child node.
            return self.children[key].predict(attributes)
        else:
            # If no child node found, return the default prediction.
            return self.default



class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, criterion = "gini", depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit
        self.criterion = criterion

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes, list(range(features.shape[1])))

    def __build_tree__(self, features, classes, attributes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """

        # Edge/Base Case
        if (classes.sum() == classes.size):
            # Return a leaf node with prediction of class 1
            return DecisionNode(1)
        
        # Check if all examples belong to the positive class
        if (classes.sum() == 0):
            # Return a leaf node with prediction of class 0
            return DecisionNode(0)
        
        # Check if there are no remaining attributes to split on.
        if (len(attributes) == 0):
            # Make a prediction based on majority class.
            if (classes.sum()*2 >= classes.size):
                return DecisionNode(1)
            else:
                return DecisionNode(0)

        else:
            # Build tree recursively

            #### Student Code Start : TASK 2

            # Find the best attribute to split on based on information gain.
            best_attr = attributes[np.argmax(information_gain(features, classes, attributes))]

            # Get unique values of the best attribute for splitting.
            col = features[:, best_attr] 
            label,count = np.unique(col,return_counts=True)
            # Create a dictionary to hold child nodes.
            children = {}

            # Remove the best attribute from the list of attributes to pass to child nodes.
            remaining_attributes = [attr for attr in attributes if attr != best_attr]
            # Split the dataset based on the unique values of the best attribute.
            for i in range(len(label)):
                indices = np.where(col == label[i])[0]
                splitfet=features[indices] 
                splilab=classes[indices]
                children[label[i]]= self.__build_tree__(splitfet,splilab,remaining_attributes,depth+1)
            # Determine the default prediction
            default_for_empty = None
            if (classes.sum()*2 >= classes.size):
                default_for_empty = 1
            else:
                default_for_empty = 0

            return DecisionNode(default_for_empty, best_attr, children)
            #### Student Code End

    def predict(self, attributes):
        return self.root.predict(attributes)
    


if __name__ == "__main__":
    ## Load all the relevant data

    ##### UNCOMMENT TO GET THE OUTPUT FOR THE PUBLIC TESTCASES FOR ENTROPY FUNCTION #####
    np.random.seed(0)
    entropy_array = np.random.randint(0, 3, 15)

    entropy_value = entropy(entropy_array)
    print("Entropy Value for seed 0:", entropy_value)

    np.random.seed(1)
    entropy_array = np.random.randint(0, 3, 15)

    entropy_value = entropy(entropy_array)
    print("Entropy Value for seed 1:", entropy_value)

    
    ##### UNCOMMENT TO GET THE OUTPUT FOR THE PUBLIC TESTCASES FOR GINI INDEX FUNCTION #####
    np.random.seed(0)
    gini_array = np.random.randint(0, 3, 15)

    gini_index_value = gini_index(gini_array)
    print("Gini Index Value for seed 0:", gini_index_value)

    np.random.seed(1)
    gini_array = np.random.randint(0, 3, 15)

    gini_index_value = gini_index(gini_array)
    print("Gini Index Value for seed 1:", gini_index_value)




    # Training inputs
    train_inputs = np.loadtxt("data/X_train.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:]

    # Training classes: we convert the string labels to integer
    train_classes = np.loadtxt("data/Y_train.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:].squeeze(1).astype('int')

    # Test inputs
    test_inputs = np.loadtxt("data/X_test.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:]

    # Tets classes: we convert the string labels to integer
    test_classes = np.loadtxt("data/Y_test.csv", delimiter=',', dtype=np.dtype(str), ndmin=2)[1:].squeeze(1).astype('int')


    #### UNCOMMENT TO GET THE OUTPUT FOR THE PUBLIC TESTCASES FOR INFORMATION GAIN FUNCTION #####
    ### Check if the information gain function is working correctly, test it for the first and second attribute
    # ### extract first 13 samples from the training data
    sample_train_inputs = train_inputs[:150]
    sample_train_classes = train_classes[:150]

    information_gain_value_0 = information_gain(sample_train_inputs, sample_train_classes, 0)
    print("Information Gain Value for first attribute:", information_gain_value_0)

    information_gain_value_1 = information_gain(sample_train_inputs, sample_train_classes, 1)
    print("Information Gain Value for second attribute:", information_gain_value_1)

    #### UNCOMMENT TO GET THE OUTPUT FOR THE PUBLIC TESTCASES FOR GINI GAIN FUNCTION #####
    ### Check if the gini gain function is working correctly, test it for the first and second attribute
    ### extract first 13 samples from the training data
    sample_train_inputs = train_inputs[:150]
    sample_train_classes = train_classes[:150]

    gini_gain_value_0 = gini_gain(sample_train_inputs, sample_train_classes, 0)
    print("Gini Gain Value for first attribute:", gini_gain_value_0)

    gini_gain_value_1 = gini_gain(sample_train_inputs, sample_train_classes, 1)
    print("Gini Gain Value for second attribute:", gini_gain_value_1)

    
    # Logging relevant statistics
    num_train_data = train_inputs.shape[0]
    print("Number of Training Data:", num_train_data)

    num_test_data = test_inputs.shape[0]
    print("Number of Test Data:", num_test_data)
    
    num_features = train_inputs.shape[1]
    print("Number of Features:", num_features)
    
    # Initialize the Decision Tree Classifier using Training Instances
    DTree = DecisionTree()
    DTree.fit(train_inputs, train_classes)

    # Print the precision for training and test instances
    train_precision = get_precision(DTree, train_inputs, train_classes)
    print("Training precision:", train_precision)

    test_precision = get_precision(DTree, test_inputs, test_classes)
    print("Test precision:", test_precision)
