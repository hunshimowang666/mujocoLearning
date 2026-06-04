"""Call Unitree Go2 official high-level flip actions.

This script uses Unitree's official SDK2 Python SportClient. It does not
implement a low-level MuJoCo torque policy; the flip motion is executed inside
the robot firmware when the high-level sport API accepts the command.

By default this script is a dry run. A real flip requires the explicit
--execute-real-robot flag.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SDK_ROOT = REPO_ROOT / "examples" / "third_party" / "unitree_sdk2_python"


def add_sdk_to_path() -> None:
    if not SDK_ROOT.exists():
        raise FileNotFoundError(
            f"Unitree SDK2 Python was not found at {SDK_ROOT}. "
            "Clone https://github.com/unitreerobotics/unitree_sdk2_python there first."
        )
    sys.path.insert(0, str(SDK_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely call official Go2 sport flip actions from Unitree SDK2 Python."
    )
    parser.add_argument(
        "--network-interface",
        default=None,
        help="Robot network interface, for example eth0/enp3s0/wlan0. Omit to use SDK default.",
    )
    parser.add_argument(
        "--action",
        choices=("backflip", "frontflip", "leftflip"),
        default="backflip",
        help="Official high-level sport action to request.",
    )
    parser.add_argument(
        "--execute-real-robot",
        action="store_true",
        help="Actually send the command to a real Go2. Without this, the script only prints.",
    )
    parser.add_argument(
        "--skip-stand-up",
        action="store_true",
        help="Do not send StandUp before the flip.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="SportClient RPC timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Go2 official sport flip client")
    print(f"SDK path: {SDK_ROOT}")
    print(f"Action: {args.action}")
    print(f"Network interface: {args.network_interface or '<SDK default>'}")

    if not args.execute_real_robot:
        print()
        print("DRY RUN: no command was sent.")
        print("To execute on a real Go2, rerun with --execute-real-robot.")
        print("Make sure the robot is in a clear open area, battery is sufficient,")
        print("people are away from the robot, and emergency stop/recovery is ready.")
        return

    confirm = input(
        "This will command a REAL Go2 to perform an acrobatic motion. "
        "Type EXECUTE to continue: "
    )
    if confirm != "EXECUTE":
        print("Cancelled.")
        return

    add_sdk_to_path()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    sport_client = SportClient()
    sport_client.SetTimeout(args.timeout)
    sport_client.Init()

    if not args.skip_stand_up:
        print("Sending StandUp()...")
        ret = sport_client.StandUp()
        print(f"StandUp return code: {ret}")
        time.sleep(2.0)

    print(f"Sending {args.action}()...")
    if args.action == "backflip":
        ret = sport_client.BackFlip()
    elif args.action == "frontflip":
        ret = sport_client.FrontFlip()
    else:
        ret = sport_client.LeftFlip()
    print(f"{args.action} return code: {ret}")

    time.sleep(1.0)
    print("Sending StopMove()...")
    ret = sport_client.StopMove()
    print(f"StopMove return code: {ret}")


if __name__ == "__main__":
    main()
