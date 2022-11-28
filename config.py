import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='SF_extract', help='data property:'
                        )
    parser.add_argument('--batch_size', type=int, default=2, help='train batch size')
    parser.add_argument('--validation_size', type=int, default=16, help='validation batch in training')
    parser.add_argument('--resolution', type=int, default=512, help='resolution')
    parser.add_argument('--num_epochs', type=int, default=50, help='train epoch')
    parser.add_argument('--num_workers', type=int, default=0, help='0 for windows')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
    parser.add_argument('--lamb', type=float, default=100, help='')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    parser.add_argument('--augmentation_prob', type=float, default=1.0, help='augmentation')
    parser.add_argument('--LV2_augmentation', type=bool, default=False, help='LV2 augmentation')
    parser.add_argument('--RGB', type=bool, default=False, help='if RGB')

    opt = parser.parse_args()

    return opt
