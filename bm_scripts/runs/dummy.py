"""Ëxample run input script"""
import time

from commonnnbm import base


def dummy_setup(*args, **kwargs):
    return time.sleep


run_list = (
    base.Run(
        "test_run",
        [
            base.BMUnit(
                x,
                setup_func=dummy_setup,
                timed_args=(x,)
                )
            for x in range(1, 4)
        ]
    ),
)

runs_report_dir = "dummy"
