
import pathlib

from commonnn import _types
from commonnnbm import base
import commonnnbm.cases.commonnn_fit as cases


if __name__ == "__main__":
    args = base.parse_standard_arguments()

    unit = base.BMUnit(
        id=args.identifier,
        gen_func=base.gen_no_structure_points,
        gen_args=((args.n_points, args.n_dim),),
        transform_func=base.compute_neighbours,
        transform_args=(args.radius_cutoff,),
        transform_kwargs={"sort": False},
        setup_func=cases.setup_commonnn_clustering__fit,
        setup_kwargs={"recipe": "neighbourhoods"},
        timed_args=(
            _types.CommonNNParameters.from_mapping({
                "radius_cutoff": args.radius_cutoff,
                "similarity_cutoff": args.similarity_cutoff
                }),
            ),
        timed_kwargs={}
        )

    this_dir = pathlib.Path(__file__).absolute().parent
    report_dir = this_dir / f"{args.machine}"
    if not report_dir.is_dir():
        report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / f"{args.identifier}{args.part}.dat"
    print("Saving timings report to: ", report_file.absolute())
    timings = base.time_unit(unit, repeats=args.repeats)

    with open(report_file, 'w') as outf:
        for t in timings:
            outf.write(f'{t}\n')
