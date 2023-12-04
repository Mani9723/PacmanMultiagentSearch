# multiAgents.py
# --------------
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
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

# Mani Shah
# CS580
# mshah22
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        score = successorGameState.getScore()
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]

        if len(foodDistances):
            score += 10/min(foodDistances)
            score -= len(newFood.asList())*10 + 10 if newPos in newFood.asList() else 0

        if manhattanDistance(newPos,newGhostStates[0].configuration.getPosition()) <= 4:
            score -= float("inf")

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):


        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def isTerminal(gameState, depth):
            return depth == self.depth or gameState.isWin() or gameState.isLose()

        def minimax(gameState, depth, agent):
            if agent == gameState.getNumAgents():
                return self.evaluationFunction(gameState) if isTerminal(gameState, depth) else minimax(gameState,depth+1,0)
            else:
                successor = []
                for legalAction in gameState.getLegalActions(agent):
                    successor.append(minimax(gameState.generateSuccessor(agent, legalAction), depth, agent + 1))
                if len(successor) == 0: return self.evaluationFunction(gameState)

            return min(successor) if agent else max(successor)

        legalActions = gameState.getLegalActions(agentIndex=0)
        return max(legalActions, key=lambda X: minimax(gameState.generateSuccessor(0, X), 1, 1))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def isTerminal(gameState, depth):
            return depth == self.depth or gameState.isWin() or gameState.isLose()
        def maxValue(gameState, alpha, beta, agentIndex, depth):
            if isTerminal(gameState,depth):
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            v = float('-inf')
            for action in actions:
                childState = gameState.generateSuccessor(agentIndex, action)
                childMinValue = minValue(childState, alpha, beta, 1, depth)
                v = childMinValue if childMinValue > v else v
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(gameState, alpha, beta, agentIndex, depth):
            if isTerminal(gameState, depth):
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            v = float('inf')

            for action in actions:
                childState = gameState.generateSuccessor(agentIndex, action)
                childMaxValue = maxValue(childState, alpha, beta, 0, depth + 1) \
                    if agentIndex % (gameState.getNumAgents() - 1) == 0 \
                    else minValue(childState, alpha, beta, agentIndex + 1, depth)
                v = childMaxValue if childMaxValue < v else v
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def value():
            beta = float('inf')
            alpha = float('-inf')
            maxVal = -99999999
            moves = {}
            for action in gameState.getLegalActions(0):
                childState = gameState.generateSuccessor(0, action)
                minScore = minValue(childState, alpha, beta, 1, 0)
                maxVal = max(maxVal, minScore)
                moves[minScore] = action
                if minScore > beta:
                    return moves[maxVal]
                alpha = max(alpha, maxVal)
            return moves[alpha]

        return value()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def isTerminal(gameState, depth):
            return depth == self.depth or gameState.isWin() or gameState.isLose()

        def average(list):
            return float(sum(list))/float(len(list))
        def maxValue(gameState, agentIndex, depth):
            v = float("-inf")
            if isTerminal(gameState,depth):
                return self.evaluationFunction(gameState)

            for action in gameState.getLegalActions(agentIndex):
                child = gameState.generateSuccessor(agentIndex, action)
                v = max(v, minValue(child, 1, depth))
            return v

        def minValue(gameState, agentIndex, depth):
            scores = []
            if isTerminal(gameState,depth):
                return self.evaluationFunction(gameState)

            for action in gameState.getLegalActions(agentIndex):
                child = gameState.generateSuccessor(agentIndex, action)
                scores.append(maxValue(child, 0, depth + 1)
                              if agentIndex % (gameState.getNumAgents() - 1) == 0
                              else minValue(child, agentIndex + 1, depth))

            return average(scores)
        def value(gameState):
            v = 0
            move = None
            for action in gameState.getLegalActions(0):
                score = minValue(gameState.generateSuccessor(0, action), 1, 0)
                v = max(v, score)
                move = action if v == score else move
            return move
        return value(gameState)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def getAverage(list):
        return sum(val for val in list)/len(list)

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    ghostAverage = getAverage([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates])
    foodScore = 0 if not newFood.asList() else min(manhattanDistance(newPos, food) for food in newFood.asList())

    return currentGameState.getScore() + min(newScaredTimes) + 1/(foodScore+0.1) - (1/(ghostAverage + 0.11))

# Abbreviation
better = betterEvaluationFunction
