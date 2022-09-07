from collections import Counter
import numpy as np
import random
import torch
import BidModel

from douzero.env.game import GameEnv


Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])


class Env:
    """
    Doudizhu multi-agent wrapper
    """

    def __init__(self, objective):
        """
        Objective is wp/adp/logadp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.objective = objective

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['first', 'second', 'third', 'landlord', 'landlord_down', 'landlord_up']:
            self.players[position] = DummyAgent(position)

        # Initialize the internal environment
        self._env = GameEnv(self.players)
        self.total_round = 0
        self.infoset = None

    def reset(self, model, device, flags=None):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'first': _deck[:17],
                          'second': _deck[20:37],
                          'third': _deck[37:],
                          'three_landlord_cards': _deck[17:20]
                          }
        for key in card_play_data:
            card_play_data[key].sort()
        # 重置叫牌
        self._env.bid_init(card_play_data)
        # self.bid_infoset = self._bid_infoset
        bid_over = self._bid_over
        self.infoset = self._bid_infoset
        return get_obs(self.infoset, bid_over)


    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """
        if not self._draw:
            if self._bid_over:
                pos = self._acting_player_position
            else:
                pos = self._bidding_player_position

            self.players[pos].set_action(action)
            self._env.step()
        if not self._draw:
            if self._bid_over:
                self.infoset = self._game_infoset
            else:
                self.infoset = self._bid_infoset

        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = {
                "play": {
                    "landlord": self._get_reward("landlord"),
                    "landlord_down": self._get_reward("landlord_down"),
                    "landlord_up": self._get_reward("landlord_up"),
                    "first": self._get_reward_bidding("first"),
                    "second": self._get_reward_bidding("second"),
                    "third": self._get_reward_bidding("third"),
                }
            }
            obs = None
        elif self._draw:
            obs = None
        else:
            obs = get_obs(self.infoset, self._bid_over)
        return obs, reward, done, self._draw, {}

    def _get_reward(self, pos):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner

        bomb_num = self._game_bomb_num
        self_bomb_num = self._env.pos_bomb_num[pos]
        bid_count = self._env.bid_count
        spring = 1.4 if self._env.spring else 1
        multiply = 2
        if pos != 'landlord':
            pos = 'farmer'
            multiply = 1
        if pos == winner:
            if self.objective == 'adp':
                return (1.1 - self._env.step_count * 0.0033) * 1.3 ** (bomb_num + bid_count) * spring * multiply / 8
            elif self.objective == 'logadp':
                return (1.0 - self._env.step_count * 0.0033) * 1.3**self_bomb_num / 4
            else:
                return 1.0 - self._env.step_count * 0.0033
        else:
            if self.objective == 'adp':
                return (-1.1 + self._env.step_count * 0.0033) * 1.3 ** (bomb_num + bid_count) * spring * multiply / 8
            elif self.objective == 'logadp':
                return (-1.0 + self._env.step_count * 0.0033) * 1.3**self_bomb_num / 4
            else:
                return -1.0 + self._env.step_count * 0.0033

    def _get_reward_bidding(self, pos):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._bid_winner
        bomb_num = self._game_bomb_num
        bid_count = self._env.bid_count
        spring = 1.4 if self._env.spring else 1
        multiply = 1 if '&' in winner else 2
        _multiply = 2 if multiply == 1 else 1
        if pos in winner:
            return (1.1 - self._env.step_count * 0.0033) * 1.3 ** (bomb_num + bid_count) * spring * multiply / 8
        else:
            return (-1.1 + self._env.step_count * 0.0033) * 1.3 ** (bomb_num + bid_count) * spring * _multiply / 8

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _bid_infoset(self):
        return self._env.bid_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()


    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _bidding_player_position(self):
        return self._env.bidding_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

    @property
    def _bid_over(self):
        return self._env.bid_over

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _bid_winner(self):
        """ A string of first/second
        """
        return self._env.get_winner_bid()

    @property
    def _draw(self):
        return self._env.draw

class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """

    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action


def get_obs(infoset, bid_over, new_model=True):
    if new_model:
        if infoset.player_position not in ["first", 'second', 'third', 'landlord', 'landlord_down', 'landlord_up']:
            raise ValueError('')
        if bid_over:
            return _get_obs_resnet(infoset)
        if infoset.player_position in ["first", 'second', 'third']:
            return _get_bid_obs_resnet(infoset)
        return _get_obs_general(infoset, infoset.player_position)
    else:
        if bid_over:
            return _get_obs_resnet(infoset)
        if infoset.player_position in ["first", 'second', 'third']:
            return _get_bid_obs_resnet(infoset)



def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    if num_left_cards > 0:
        one_hot[num_left_cards - 1] = 1

    return one_hot


def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
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


def _action_seq_list2array(action_seq_list):
    action_seq_array = np.ones((len(action_seq_list), 54)) * -1  # Default Value -1 for not using area
    for row, list_cards in enumerate(action_seq_list):
        if list_cards != []:
            action_seq_array[row, :54] = _cards2array(list_cards[1])
    return action_seq_array


def _action_seq_list2array_lstm(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
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
        if new_model:
            sequence.extend(empty_sequence)
        else:
            empty_sequence.extend(sequence)
            sequence = empty_sequence
    return sequence


def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot


def _get_obs_resnet(infoset):
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    spring = np.array([1]) if infoset.spring else np.array([0])
    spring_batch = np.repeat(spring[np.newaxis, :],
                             num_legal_actions, axis=0)

    position_map = {
        "landlord": [1, 0, 0],
        "landlord_up": [0, 1, 0],
        "landlord_down": [0, 0, 1]
    }
    position_info = np.array(position_map[infoset.player_position])
    position_info_batch = np.repeat(position_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    bid_info = np.array(infoset.bid_info).flatten()
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)


    three_landlord_cards = _cards2array(infoset.three_landlord_cards)
    three_landlord_cards_batch = np.repeat(three_landlord_cards[np.newaxis, :],
                                           num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    other_handcards_left_list = []
    for pos in ["landlord", "landlord_up", "landlord_up"]:
        if pos != infoset.player_position:
            other_handcards_left_list.extend(infoset.all_handcards[pos])

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    num_cards_left = np.hstack((
                         landlord_num_cards_left,  # 20
                         landlord_up_num_cards_left,  # 17
                         landlord_down_num_cards_left))

    x_batch = np.hstack((
                         bid_info_batch,  # 4
                         bomb_num_batch,  # 15
                         ))
    x_no_action = np.hstack((
                             bid_info,
                             bomb_num,
                             ))
    z = np.vstack((
                  num_cards_left,  # 54
                  my_handcards,  # 54
                  other_handcards,  # 54
                  three_landlord_cards,  # 54
                  landlord_played_cards,  # 54
                  landlord_up_played_cards,  # 54
                  landlord_down_played_cards,  # 54
                  _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq, 82))
                  ))

    _z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:, np.newaxis, :]
    z_batch = np.zeros([len(_z_batch), 90, 54], int)
    for i in range(0, len(_z_batch)):
        z_batch[i] = np.vstack((my_action_batch[i], _z_batch[i]))
    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def _get_bid_obs_resnet(infoset):
    num_legal_actions = 2
    my_handcards = _cards2array(infoset.player_hand_cards)

    bid_info = np.array(infoset.bid_info)
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)
    bid_info_z = np.multiply(bid_info, np.ones((54, 4))).transpose((1, 0))

    my_action_batch = np.zeros((2, 54))
    my_action_batch[1] = np.ones(54)

    x_batch = np.hstack((
        bid_info_batch,  # 5
    ))
    x_no_action = np.hstack((
        bid_info,
    ))
    z = np.vstack((
        my_handcards,  # 54
        bid_info_z
    ))

    _z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:, np.newaxis, :]
    z_batch = np.concatenate((my_action_batch, _z_batch), axis=1)

    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs




def _get_obs_general(infoset, position):
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    position_map = {
        "landlord": [1, 0, 0],
        "landlord_up": [0, 1, 0],
        "landlord_down": [0, 0, 1]
    }
    position_info = np.array(position_map[position])
    position_info_batch = np.repeat(position_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    bid_info = np.array(infoset.bid_info).flatten()
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)

    multiply_info = np.array(infoset.multiply_info)
    multiply_info_batch = np.repeat(multiply_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    three_landlord_cards = _cards2array(infoset.three_landlord_cards)
    three_landlord_cards_batch = np.repeat(three_landlord_cards[np.newaxis, :],
                                           num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    other_handcards_left_list = []
    for pos in ["landlord", "landlord_up", "landlord_up"]:
        if pos != position:
            other_handcards_left_list.extend(infoset.all_handcards[pos])

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    num_cards_left = np.hstack((
                         landlord_num_cards_left,  # 20
                         landlord_up_num_cards_left,  # 17
                         landlord_down_num_cards_left))

    x_batch = np.hstack((
                         bid_info_batch,  # 12
                         multiply_info_batch))  # 3
    x_no_action = np.hstack((
                             bid_info,
                             multiply_info))
    z =np.vstack((
                  num_cards_left,
                  my_handcards,  # 54
                  other_handcards,  # 54
                  three_landlord_cards,  # 54
                  landlord_played_cards,  # 54
                  landlord_up_played_cards,  # 54
                  landlord_down_played_cards,  # 54
                  _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq, 32, False))
                  ))

    _z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:,np.newaxis,:]
    z_batch = np.zeros([len(_z_batch),40,54],int)
    for i in range(0, len(_z_batch)):
        z_batch[i] = np.vstack((my_action_batch[i],_z_batch[i]))
    obs = {
        'position': position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs