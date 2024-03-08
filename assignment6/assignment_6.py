import numpy as np
from pathlib import Path
from typing import Tuple
import random



class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)



def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value


#Fucntion that calculates entropy
def entropy(probability):
    p = probability
    return -(p*np.log2(p)+(1-p)*np.log2*(1-p))

def information_gain(examples: np.ndarray) -> float:
    p = 0
    n = 0

    p_set = 0
    n_set = 0

    for item in examples[1:][:-1]: #Determining the amount of p (positive) and n (negative) examples
            if item == 1:
                p += 1
            else:
                n += 1

    B = entropy(p/(p+n))
    remainder = 0

    for item in examples[1:]:
        for i in range(len(item)-1):
            if item[i] == 1:
                p_set += 1
            else:
                n_set += 1
        remainder += sum(p_set, n_set)/sum(p, n) * entropy(p_set/sum(p_set, n_set))
    gain = B - remainder

    return gain


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    # TODO implement the importance function for both measure = "random" and measure = "information_gain"
    if measure == "random":
        random_number = random.randint(1,6)
        return attributes[random_number]

    elif measure == "information_gain":

        #Struggeling to understand how the information gain can determine the best attribute from the testdata. 
        
        

        


        



    pass


def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # TODO implement the steps of the pseudocode in Figure 19.5 on page 678

    return node



def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test




if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "random"

    tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)

    print(f"Training Accuracy {accuracy(tree, train)}")
    print(f"Test Accuracy {accuracy(tree, test)}")
