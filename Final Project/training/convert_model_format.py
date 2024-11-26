import argparse

from utils import get_device, load_model


def convert_model(model_path, target_format, int8, half, optimize, simplify):
    """
    load model and convert to target format

    Args:
        model_path: model path, only support .pt format
        target_format: target format
    """
    device = get_device()

    # load model
    model = load_model(model_path)

    # export to target format
    model.export(
        format=target_format,
        device=device,
        int8=int8,
        half=half,
        optimize=optimize,
        simplify=simplify,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--format", type=str, required=False, default="", help="target format")
    parser.add_argument(
        "--int8", type=bool, default=False, help="use int8 quantization"
    )
    parser.add_argument("--half", type=bool, default=False, help="use half precision")
    parser.add_argument(
        "--optimize",
        type=bool,
        default=False,
        help="optimize model when exporting to torchscript",
    )
    parser.add_argument(
        "--simplify",
        type=bool,
        default=False,
        help="simplify model when exporting to onnx",
    )

    args = parser.parse_args()

    if not args.model.endswith(".pt"):
        raise ValueError("model path must end with .pt")

    if args.optimize and args.format != "torchscript":
        raise ValueError("optimize flag only supported when exporting to torchscript")

    if args.simplify and args.format != "onnx":
        raise ValueError("simplify flag only supported when exporting to onnx")

    convert_model(
        args.model, args.format, args.int8, args.half, args.optimize, args.simplify
    )


if __name__ == "__main__":
    main()
