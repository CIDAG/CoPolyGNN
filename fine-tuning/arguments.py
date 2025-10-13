import argparse


def arg_parse():

    parser = argparse.ArgumentParser(
        description='Polymer Multitask Framework with Task Affinity Grouping'
    )

    parser.add_argument('--device', dest='device', type=str, default='cuda')
    parser.add_argument('--seed', dest='seed', type=int, default=1234)

    parser.add_argument('--batch', dest='batch_size', type=int, default=16)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.1)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-6)

    parser.add_argument('--hidden-channels', dest='hidden_channels', type=int, default=64)
    parser.add_argument('--output', dest='output', type=str, default='results')

    return parser.parse_args()
