
import argparse
import os
import pathlib

from scalene import scalene_profiler

from commonnn import _types
from commonnnbm import base
import commonnnbm.cases.commonnn_fit as cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make and run a BMUnit')
    parser.add_argument(
        "-n", "--n_points",
        type=int,
        required=True,
        help="Number of points in data set"
        )
    parser.add_argument(
        "-d", "--n_dim",
        type=int,
        default=2,
        help="Number of dimensions of the data set"
        )
    parser.add_argument(
        "-id", "--identifier",
        type=str,
        default=None,
        help="BMUnit ID"
        )
    parser.add_argument(
        "-r", "--radius_cutoff",
        type=float,
        default=1.0,
        help="Radius cutoff"
        )
    parser.add_argument(
        "-sim", "--similarity_cutoff",
        type=int,
        default=0,
        help="Similarity cutoff"
        )
    parser.add_argument(
        "-m", "--machine",
        type=str,
        default=None,
        help="Machine identifier"
        )
    parser.add_argument(
        "-repeats", "--repeats",
        type=int,
        default=10,
        help="Number of timings per benchmark unit"
        )
    parser.add_argument(
        "-p", "--part",
        type=str,
        default="",
        help="Report part"
        )
    parser.add_argument(
        "-profile", "--profile",
        action="store_true",
        help="Profile with scalene instead of timings"
        )

    args = parser.parse_args()

    if args.identifier is None:
        args.identifier = f"{args.n_points}_{args.n_dim}_{args.radius_cutoff}_{args.similarity_cutoff}"

    if args.machine is None:
        args.machine = os.environ.get("HOST", "unk")

    unit = base.BMUnit(
        id=args.identifier,
        gen_func=base.gen_no_structure_points,
        gen_args=((args.n_points, args.n_dim),),
        setup_func=cases.setup_commonnn_clustering__fit,
        setup_kwargs={"recipe": "coordinates", "prep": "pass"},
        timed_args=(
            _types.CommonNNParameters.from_mapping({
                "radius_cutoff": args.radius_cutoff,
                "similarity_cutoff": args.similarity_cutoff
                }),
            ),
        timed_kwargs={}
        )

    repo_dir = pathlib.Path(__file__).absolute().parent.parent.parent
    report_dir = repo_dir / f"reports_unit/{args.machine}/no_structure/commonnn__fit/points"
    if not report_dir.is_dir():
        report_dir.mkdir(parents=True, exist_ok=True)

    if args.profile:
        @profile
        def profiled():
            base.profile_unit(unit)
        profiled()
    else:
        report_file = report_dir / f"{args.identifier}{args.part}.dat"
        print("Saving timings report to: ", report_file.absolute())
        timings = base.time_unit(unit, repeats=args.repeats)

        with open(report_file, 'w') as outf:
            for t in timings:
                outf.write(f'{t}\n')
