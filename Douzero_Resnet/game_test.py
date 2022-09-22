from douzero.env import game
import numpy as np
import game_eval
import random
from collections import Counter

Card2Column = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5,
               11: 6, 12: 7, 13: 8, 14: 9, 17: 10}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}


class BailanAgent():

    def __init__(self):
        self.name = 'bailan'

    def act(self, infoset):
        action = []
        action_len = 20
        if len(infoset.legal_actions) == 2 and infoset.legal_actions[0][0] == 0:
            return random.choice(infoset.legal_actions)
        for a in infoset.legal_actions:
            l = len(a)
            if l <= action_len:
                action_len = l
                action = a
        # print("摆烂", action)
        # return random.choice(infoset.legal_actions)
        # print(action, infoset.legal_actions)
        return action




class RandomAgent():

    def __init__(self):
        self.name = 'Random'

    def act(self, infoset):
        return random.choice(infoset.legal_actions)

class RandomAgent2():

    def __init__(self):
        self.name = 'Random'

    def act(self, infoset):
        return random.choice(infoset.legal_actions)


def load_card_play_models():
    players = {}
    for position in ['first', 'landlord']:
        players[position] = RandomAgent()
    for position in ['second', 'landlord_down']:
        players[position] = RandomAgent()
    for position in ['third', 'landlord_up']:
        players[position] = RandomAgent2()
    return players

def deck_set():
    deck = []
    for i in range(3, 15):
        deck.extend([i for _ in range(4)])
    deck.extend([17 for _ in range(4)])
    deck.extend([20, 30])

    _deck = deck.copy()
    np.random.shuffle(_deck)
    card_play_data = {'first': _deck[:17],
                      'second': _deck[20:37],
                      'third': _deck[37:],
                      'three_landlord_cards': _deck[17:20],
                      }
    return card_play_data

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(46, dtype=np.int8)

    matrix = np.zeros([4, 11], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))

def _action_seq_list2array_lstm(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 46))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(4, 92)
    return action_seq_array

def _process_action_seq(sequence, length=15, new_model=True):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    sequence = sequence[::-1]
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    """

    position_map = {"landlord": 0, "farmer": 1}
    action_seq_array = np.ones((len(action_seq_list), 46)) * -1  # Default Value -1 for not using area
    for row, list_cards in enumerate(action_seq_list):
        if list_cards != []:
            action_seq_array[row, :46] = _cards2array(list_cards[1])
    return action_seq_array

if __name__ == "__main__":

    players = load_card_play_models()

    gameenv = game.GameEnv(players)
    for i in range(100):
        card_play_data = deck_set()
        gameenv.bid_init(card_play_data)
        while not gameenv.bid_over:
            gameenv.step()
            print(gameenv.bid_info)
        if not gameenv.draw:
            while not gameenv.game_over:
                gameenv.step()
        print(gameenv.step_count)
        gameenv.reset()
        print("++++++++++++++++++++++++++++"+str(i)+"+++++++++++++++++++++++++++++++++++")

