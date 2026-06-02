import sys
from datetime import datetime

from train_cnn_cifar10 import main


def has_option(option_name):
    return any(arg == option_name or arg.startswith(f"{option_name}=") for arg in sys.argv[1:])


def main_scratch():
    if has_option("--weights"):
        raise SystemExit(
            "train_resnet18_scratch_cifar10.py always uses --weights none. "
            "Use train_cnn_cifar10.py if you want to choose weights manually."
        )

    sys.argv.extend(["--weights", "none"])
    if not has_option("--run-name"):
        run_name = datetime.now().strftime("cifar10_resnet18_scratch_%Y%m%d_%H%M%S")
        sys.argv.extend(["--run-name", run_name])

    main()


if __name__ == "__main__":
    main_scratch()
