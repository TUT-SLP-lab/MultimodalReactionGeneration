from argparse import ArgumentParser, Namespace

from mr_gen.utils.arg_manager import IntegratedBasicArgmentParser


def add_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--acostic-feat-size", type=int, default=81)
    parser.add_argument("--motion-feat-size", type=int, default=18)
    parser.add_argument("--motion-num-lstm", type=int, default=1)
    parser.add_argument("--acostic-num-lstm", type=int, default=1)
    parser.add_argument("--acostic-num-layers", type=int, default=5)
    parser.add_argument("--motion-num-layers", type=int, default=5)
    parser.add_argument("--acostic-lstm-size", type=int, default=128)
    parser.add_argument("--motion-lstm-size", type=int, default=128)
    parser.add_argument("--acostic-lstm-out-size", type=int, default=256)
    parser.add_argument("--motion-lstm-out-size", type=int, default=256)
    parser.add_argument("--acostic-affine-size", type=int, default=256)
    parser.add_argument("--motion-affine-size", type=int, default=256)
    parser.add_argument("--acostic-bottleneck-size", type=int, default=64)
    parser.add_argument("--motion-bottleneck-size", type=int, default=64)
    parser.add_argument("--acostic-output-size", type=int, default=256)
    parser.add_argument("--motion-output-size", type=int, default=256)
    parser.add_argument("--att-heads", type=int, default=1)
    parser.add_argument("--att-num-layers", type=int, default=1)
    parser.add_argument("--att-use-residual", action="store_true")
    parser.add_argument("--att-use-layer-norm", action="store_true")
    parser.add_argument("--decoder-num-layers", type=int, default=5)
    parser.add_argument("--decoder-num-lstm", type=int, default=1)
    parser.add_argument("--decoder-lstm-size", type=int, default=128)
    parser.add_argument("--decoder-affine-size", type=int, default=256)
    parser.add_argument("--decoder-bottleneck-size", type=int, default=64)
    parser.add_argument("--decoder-output-size", type=int, default=256)
    parser.add_argument("--decoder-mapping-size", type=int, default=64)
    parser.add_argument("--dropout-rate", type=int, default=0.5)
    parser.add_argument("--output-size", type=int, default=18)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--use-layer-norm", action="store_true")
    parser.add_argument("--use-relu", action="store_true")
    parser.add_argument("--use-mixing", action="store_true")
    parser.add_argument("--use-residual", action="store_true")
    parser.add_argument("--decoder-bidirectional", action="store_true")
    parser.add_argument("--decoder-use-layer-norm", action="store_true")
    parser.add_argument("--decoder-use-relu", action="store_true")
    parser.add_argument("--decoder-use-mixing", action="store_true")
    parser.add_argument("--decoder-use-residual", action="store_true")

    return parser


def get_args() -> Namespace:
    parser = IntegratedBasicArgmentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    return args
