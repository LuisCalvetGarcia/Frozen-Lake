"""
Abstract Base Agent Module.
This module defines the foundational structure for all reinforcement learning 
and evolutionary agents in the project. It enforces a standard interface 
for action selection and memory management.
"""

import abc

class Trampes(Exception):
    """
    Custom exception raised when an agent attempts to access memory 
    beyond its officially permitted limit (acting as an anti-cheat mechanism).
    """

    def __init__(self) -> None:
        # The error message is kept in the original language to maintain 
        # compatibility with potential automated grading scripts.
        self.message = "Has fet trampes, no pots fer-ho"
        super().__init__(self.message)


class Agent:
    """
    Abstract Base Class for all AI agents.
    Defines the core attributes (memory, ID) and requires subclasses to 
    implement specific behaviors like action selection ('actua').
    """
    
    # Class-level variable to keep track of the number of instantiated agents
    ID_AGENT = 1

    def __init__(self, long_memoria: int) -> None:
        """
        Initializes the base agent.

        Args:
            long_memoria (int): The maximum number of past states/actions 
                                the agent is allowed to remember and access.
        """
        self.__memoria_permesa = long_memoria
        self.__memoria = []
        
        # Auto-assign a unique name based on the class instance count
        self.nom = f"Agent {Agent.ID_AGENT}"
        Agent.ID_AGENT += 1

        # Internal variable for UI rendering purposes
        self._posicio_pintar = None

    def get_memoria(self, temps: int) -> dict:
        """
        Retrieves a saved memory from a specific number of iterations ago.

        Args:
            temps (int): The number of steps back in time to retrieve memory from. 
                         Must be less than or equal to `long_memoria`.

        Returns:
            dict | any: The stored information from the specified iteration, 
                        or None if the requested time exceeds the current memory length.

        Raises:
            Trampes: If the agent attempts to look back further than `__memoria_permesa`.
        """
        if temps > self.__memoria_permesa:
            raise Trampes

        mem = None

        # Check if we have enough recorded history to look back 'temps' steps
        if len(self.__memoria) > (temps - 1):
            mem = self.__memoria[len(self.__memoria) - temps]

        return mem

    def set_posicio(self, posicio: tuple) -> None:
        """
        Internal setter for the agent's rendering coordinates.
        NOTE: This should not be called manually by the agent's logic.

        Args:
            posicio (tuple): The (x, y) coordinates for the UI.
        """
        self._posicio_pintar = posicio

    def set_memoria(self, info) -> None:
        """
        Stores information (like state, action, or reward) into the agent's history.

        Args:
            info (any): The data to be recorded at the current time step.
        """
        self.__memoria.append(info)

    @abc.abstractmethod
    def actua(self, percepcio):
        """
        Core decision-making method. 
        Must be implemented by all subclasses (e.g., SARSA, Q-Learning, Genetic).

        Args:
            percepcio (any): The current state or observation from the environment.

        Returns:
            The action chosen by the agent.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pinta(self, display) -> None:
        """
        Renders the agent onto the provided display.
        Must be implemented by subclasses if custom visualization is required.

        Args:
            display (any): The UI canvas or rendering surface.
        """
        raise NotImplementedError