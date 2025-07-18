# file nim.py
class Nim():

    def __init__(self, initial = [1, 3, 5, 7]):

        '''
            Initialize game board.
            Each game board has
                - 'piles'  : a list of how many elements remain in each pile
                - 'player' : 0 or 1 to indicate which player's turn
                - 'winner' : None, 0, or 1 to indicate who the winner is
        '''
        
        self.piles = initial.copy()
        self.player = 0
        self.winner = None
        
    @staticmethod
    def available_actions(piles):
        
        '''
            self.avaliable_actions(piles) takes a 'piles' list as input
            and returns all of the available actions '(i, j)' int that state.

            Action '(i, j)' represents the action of removing 'j' items
            from pile 'i'.
        '''

        actions = set()

        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))

        return actions

    def other_player(self, player):

        '''
            self.other_player(player) returns the player that is not
            'player'. Assumes 'player' is either 0 or 1.
        '''

        return 0 if player == 1 else 1

    def switch_player(self):

        '''
            Switch the current player to the other player.
        '''

        self.player = self.other_player(self.player)

    def move(self, action):
        
        '''
            Make the move 'action' for the current player.
            'action' must be a tuple '(i, j)'.
        '''

        pile, count = action

        # check for errors
        if self.winner is not None:
            raise Exception('Game already won.')
        else:
            if pile < 0 or pile >= len(self.piles):
                raise Exception('Invalid pile.')
            else:
                if count < 1 or count > self.piles[pile]:
                    raise Exception('Invalid number of objects.')
                
        # update pile
        self.piles[pile] -= count

        # check the winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player
        else:
            self.switch_player()
