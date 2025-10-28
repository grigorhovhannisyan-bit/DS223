"""
  Responsible for defining the bandit class and the functions that will be used to train the bandit.
"""
from abc import ABC, abstractmethod



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p): 
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#

# These class should also be put in a sperate file, Since this is an abstarct 
# Bandit class file, it should only be concerned with defining bandit functions and not the visualization.
# Instead of the visualization class, we will have a separate Performance class that will be responsible for the visualization of the bandit's performance and the comparison of the two algorithms.

"""
class Visualization():

    def plot1(self):
        # Visualize the performance of each bandit: linear and log
        pass

    def plot2(self):
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        pass
"""

# It's better for those classes to be seperate files too since they are not part of the bandit class but inherit it.

"""
#--------------------------------------#  

class EpsilonGreedy(Bandit):
    pass

#--------------------------------------#

class ThompsonSampling(Bandit):
    pass
"""

# Those should be their own file as well for the same reason as the visualization class.

"""
def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    pass

if __name__=='__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
"""
