# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    process = []
    method = util.Stack()
    method.push((start, []))
    while not method.isEmpty():
        start, path = method.pop()
        process.append(start)
        if problem.isGoalState(start):
            return path
        for pos in problem.getSuccessors(start):
            if pos[0] not in process:
                method.push((pos[0], path + [pos[1]]))
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    fringe = Queue()  # Fringe to manage which states to expand
    fringe.push(problem.getStartState())
    visited = []  # List to check whether state has already been visited
    tempPath = []  # Temp variable to get intermediate paths
    path = []  # List to store final sequence of directions
    pathToCurrent = Queue()  # Queue to store direction to children (currState and pathToCurrent go hand in hand)
    currState = fringe.pop()
    while not problem.isGoalState(currState):
        if currState not in visited:
            visited.append(currState)
            successors = problem.getSuccessors(currState)
            for child, direction, cost in successors:
                fringe.push(child)
                tempPath = path + [direction]
                pathToCurrent.push(tempPath)
        currState = fringe.pop()
        path = pathToCurrent.pop()

    return path
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    def priority(item):
        return problem.getCostOfActions(item[1])

    # add an item to the queue with priority from 'priority'
    queue_ucs = util.PriorityQueueWithFunction(priority)  # initialize an empty queue
    extended = []  # record extended node
    queue_ucs.push([problem.getStartState(), []])  # insert starting node into queue

    while not queue_ucs.isEmpty():
        # next_node = [state, path]
        next_node = queue_ucs.pop()

        # if the node the the goal state, return the search path
        if problem.isGoalState(next_node[0]):
            return next_node[1]

        # for unexpanded nodes, expand the search tree
        if next_node[0] not in extended:
            extended.append(next_node[0])
            # successors = (successor, action, stepCost)
            for successors in problem.getSuccessors(next_node[0]):
                if successors[0] not in extended:
                    path = next_node[1][0:]
                    # 'path = next_node[1]' is wrong!!!   (好大一个坑)
                    # print(id(next_node[1]))
                    # print(id(path))
                    path.append(str(successors[1]))
                    queue_ucs.push([successors[0], path])
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    extent_state = []
    # 使用优先队列，每次扩展都是选择当前代价最小的方向
    states = util.PriorityQueue()
    states.push((start, []), nullHeuristic(start, problem))
    nCost = 0
    while not states.isEmpty():
        state, actions = states.pop()
        # 目标测试
        if problem.isGoalState(state):
            return actions
        # 检查重复
        if state not in extent_state:
            # 扩展
            successors = problem.getSuccessors(state)
            for node in successors:
                coordinate = node[0]
                direction = node[1]
                if coordinate not in extent_state:
                    newActions = actions + [direction]
                    # 计算动作代价和启发式函数值得和
                    newCost = problem.getCostOfActions(newActions) + heuristic(coordinate, problem)
                    states.push((coordinate, actions + [direction]), newCost)
        extent_state.append(state)
    # 返回动作序列
    return actions
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
