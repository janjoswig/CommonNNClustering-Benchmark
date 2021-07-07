import json
from operator import itemgetter
import pathlib

from IPython.core.magics.execution import TimeitResult


def raw_to_timeitresult(all_runs):
    loops = 1
    repeat = len(all_runs)
    best = min(all_runs)
    worst = max(all_runs)
    precision = 3
    compile_time = 0

    return TimeitResult(
        loops, repeat, best, worst, all_runs, compile_time, precision
        )


class TimeitResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TimeitResult):
            return {
                "_type": "TimeitResult",
                "loops": obj.loops,
                "repeat": obj.repeat,
                "best": obj.best,
                "worst": obj.worst,
                "all_runs": obj.all_runs,
                "compile_time": obj.compile_time,
                "precision": obj._precision
                }
        return super().default(self, obj)


class TimitResultDecoder(json.JSONDecoder):
    """Use :func:`as_TimeitResult` object hook on json load instead"""
    def default(self, obj):
        decoded_obj = super().default(self, obj)
        if isinstance(decoded_obj, list) and obj[0] == "TimeitResult":
            return TimeitResult(*obj[1].values())
        return decoded_obj


def as_TimeitResult(obj):
    try:
        _type = obj.pop('_type')
    except KeyError:
        _type = None

    if _type is None:
        return obj

    if _type == 'TimeitResult':
        return TimeitResult(*obj.values())

    return obj


def combine_timeit_results(*results):
    """Merge TimeitResults

    Args:
        *results: TimeitResults instances

    Returns:
        TimeitResult

    Note:
        Be careful about merging results from different sources.
        Merging e.g. takes the minimum number of loops done in
        each timing.
    """

    loops = min(r.loops for r in results)
    repeat = sum(r.repeat for r in results)
    all_runs = []
    for r in results:
        all_runs.extend(r.all_runs)
    best = min(all_runs)
    worst = max(all_runs)
    precision = max(r._precision for r in results)
    compile_time = max(r.compile_time for r in results)

    return TimeitResult(
        loops, repeat, best, worst, all_runs, compile_time, precision
        )


def save_report(timings, report_file, overwrite=False):
    report_file = pathlib.Path(report_file)
    report_file.parent.mkdir(exist_ok=True, parents=True)

    if report_file.is_file() and not overwrite:
        print("Report already exists. Set `overwrite = True` to force overwrite.")
        return

    with open(report_file, "w") as f:
        json.dump(timings, f, indent=4, cls=TimeitResultEncoder)


def load_report(report_file):

    with open(report_file) as f:
        timings = json.load(f, object_hook=as_TimeitResult)

    return timings


def add_previous_reports(timings, *previous_timings):

    for pt in previous_timings:
        for run_id, timing in pt.items():
            if run_id in timings:
                combined = combine_timeit_results(
                    timings[run_id], timing
                )
            else:
                combined = timing

            timings[run_id] = combined


def get_ratios(timings, base=None, which="best"):
    """Get relative performance of runs based on timings

    Args:
        timinings: Timings mapping (maps run ID strings to TimeitResults)

    Keyword args:
        base: Use the timing for this run ID as the baseline (factor 1)
        which: Which timing should be shown ("average", "best", or "worst")
    """

    if base is not None:
        base = getattr(timings[base], which)
    else:
        base = min(
            getattr(x, which)
            for x in timings.values()
            if isinstance(x, TimeitResult)
        )

    sorted_ratios = sorted(
        [
            (k, getattr(v, which) / base)
            for k, v in timings.items()
            if isinstance(v, TimeitResult)
        ],
        key=itemgetter(1)
    )

    return sorted_ratios


def print_ratios(ratios):
    """Pretty print timing ratios

    Args:
        ratios: Ratios obtained with :func:`get_ratios` as
            list of run ID/factor tuples.
    """

    print(f"{'Run ID':>10}: Factor")
    print("=======================")
    for run_id, factor in ratios:
        print(f"{run_id:>10}: {factor:.2f}")
