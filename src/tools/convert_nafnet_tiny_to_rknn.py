import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert NAFNet-tiny ONNX to RKNN")
    parser.add_argument("--onnx", required=True, help="Path to nafnet_tiny.onnx")
    parser.add_argument("--output", default="nafnet_tiny.rknn", help="Output RKNN path")
    parser.add_argument("--dataset", default=None, help="Calibration dataset txt path, required when --quant")
    parser.add_argument("--target", default="rv1126b", help="RKNN target platform, e.g. rv1126b")
    parser.add_argument("--input-size", default="256,256", help="Input size as H,W")
    parser.add_argument("--quant", action="store_true", help="Enable INT8 quantization")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise SystemExit(
            "RKNN Toolkit2 is not installed in this Python environment. "
            "Please install it on your conversion machine first."
        ) from exc

    onnx_path = Path(args.onnx)
    output_path = Path(args.output)

    if not onnx_path.exists():
        raise SystemExit(f"ONNX file not found: {onnx_path}")

    dataset_path = None
    if args.quant:
        if args.dataset is None:
            raise SystemExit("--dataset is required when --quant is enabled")
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise SystemExit(f"Dataset txt not found: {dataset_path}")

    try:
        input_h, input_w = [int(v.strip()) for v in args.input_size.split(",", 1)]
    except Exception as exc:
        raise SystemExit("--input-size must look like 256,256") from exc

    rknn = RKNN(verbose=True)

    print("[1/5] Configuring RKNN")
    ret = rknn.config(
        target_platform=args.target,
        optimization_level=3,
    )
    if ret != 0:
        raise SystemExit(f"rknn.config failed: {ret}")

    print("[2/5] Loading ONNX")
    ret = rknn.load_onnx(
        model=str(onnx_path),
        input_size_list=[[3, input_h, input_w]],
    )
    if ret != 0:
        raise SystemExit(f"rknn.load_onnx failed: {ret}")

    print("[3/5] Building RKNN")
    ret = rknn.build(
        do_quantization=args.quant,
        dataset=str(dataset_path) if dataset_path else None,
    )
    if ret != 0:
        raise SystemExit(f"rknn.build failed: {ret}")

    print("[4/5] Exporting RKNN")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise SystemExit(f"rknn.export_rknn failed: {ret}")

    print("[5/5] Done")
    print(f"Generated: {output_path}")
    rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
