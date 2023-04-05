from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="The batch size. Default: 2")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="类别数. Default: 3")

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs. Default: 20")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="The learning rate. Default: 0.0001")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum. Default: 0.9")
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.99,
        help="The learning rate decay factor. Default: 0.1")
    parser.add_argument(
        "--lr_decay_epochs",
        type=int,
        default=1,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 2")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")

    parser.add_argument(
        '--crop_size',
        help='输入网络的图像大小',
        nargs='+',
        type=int,
        default=[640, 480]
    )

    parser.add_argument(
        '--preprocess',
        help='是否对输入图像进行预处理， store_false 为True// store_true 为False',
        action='store_true'
    )
    parser.add_argument(
        '--model',
        help='网络模型：UNet、ENet、FastSCNN、LiteSegMobileNet',
        type=str,
        default='ENet'
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=r'F:\WorkPlace\dataset\narraw_seam_2\dataset',
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="celoss",
    )
    parser.add_argument(
        "--gama",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
    )

    parser.add_argument(
        "--with-unlabeled",
        dest='ignore_unlabeled',
        action='store_false',
        help="The unlabeled class is not ignored.")
    parser.add_argument(
        "--resume",
        action='store_true',
        help="是否从断点处恢复训练")
    parser.add_argument(
        "--pre_train",
        action='store_true',
        help="是否加载预训练模型")
    parser.add_argument(
        "--pre_train_path",
        default='',
        type=str,
        help="预训练模型地址")

    args = parser.parse_args()
    args.crop_size *= 2 if len(args.crop_size) == 1 else 1
    return args

if __name__ == '__main__':
    get_arguments()