import os
import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dou Dizhu Evaluation')
    parser.add_argument('--player_1_bid', type=str, default='random')

    parser.add_argument('--player_2_bid', type=str, default='random')

    parser.add_argument('--player_3_bid', type=str, default='random')

    parser.add_argument('--player_1_playcard', type=str,
                        default='douzero_checkpoints/douzero/general_landlord_3840800.ckpt')

    parser.add_argument('--player_2_playcard', type=str, default='random')

    parser.add_argument('--player_3_playcard', type=str, default='random')

    parser.add_argument('--eval_data', type=str, default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    use_random_bid = False
    use_random_playcard = False

    if use_random_bid:
        args.player_1_bid = 'random'
        args.player_2_bid = 'random'
        args.player_3_bid = 'random'

    if use_random_playcard:
        args.player_1_playcard = 'random'
        args.player_2_playcard = 'random'
        args.player_3_playcard = 'random'

    evaluate(args.player_1_bid,
             args.player_2_bid,
             args.player_3_bid,
             args.player_1_playcard,
             args.player_2_playcard,
             args.player_3_playcard,
             args.eval_data,
             args.num_workers)
