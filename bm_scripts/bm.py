import argparse
import importlib
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute a benchmark')
    # parser.add_argument(
    #     "-i", "--imports",
    #     type=str,
    #     nargs="+",
    #     default=["sklearn.metrics"],
    #     help="List of modules to import"
    #     )
    parser.add_argument(
        "-runs", "--runs",
        type=str,
        required=True,
        help="The name of a module that defines a run list"
        )
    parser.add_argument(
        "-m", "--machine",
        type=str,
        default="curta",
        help="Machine identifier"
        )
    parser.add_argument(
        "-r", "--repeats",
        type=int,
        default=10,
        help="Number of timings per benchmark unit"
        )
    parser.add_argument(
        "-p", "--part",
        type=str,
        default="1",
        help="Report part"
        )
    parser.add_argument(
        "-b", "--black_list",
        type=int,
        nargs="+",
        default=None,
        help="Run indices not to consider for execution"
        )

    args = parser.parse_args()

    runs = importlib.import_module(str(args.runs))

    if args.black_list is None:
        black_list = set()
    else:
        black_list = set(int(x) for x in args.black_list)

    repo_dir = pathlib.Path(__file__).absolute().parent.parent
    report_dir = repo_dir / f"reports/{args.machine}/{runs.runs_report_dir}"
    if not report_dir.is_dir():
        report_dir.mkdir(parents=True, exist_ok=True)

    for i, run in enumerate(runs.run_list):
        if i in black_list:
            continue

        report_file = report_dir / f"{run.run_name}_raw_{args.part}.json"
        print(report_file.absolute())

        run.collect(
            v=True,
            report_file=report_file,
            repeats=args.repeats
            )
