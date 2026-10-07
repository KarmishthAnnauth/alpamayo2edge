"""Stock Bench2Drive evaluator, for the official towns.

Runs the vendored, unmodified `leaderboard_evaluator.main()` of
`~/projects/TriTrack/TriTrack/Bench2Drive` with ONE of TriTrack's patches: the
server command line (`patches.patch_server_args`), because this box needs
`-RenderOffScreen` (tritrack.env) and because the evaluator's crash cleanup
otherwise `kill -9`s every CARLA on the same graphics adapter, other users'
included. None of TriTrack's world patches (traffic density, junction gate,
emergency-vehicle speed, pedestrian timing, route planner, parked vehicles) is
applied: those exist for the `complete_scene_3` twin and would make a score here
incomparable with published Bench2Drive numbers (RUNNING_EVALS.md).

Every argument is the evaluator's own. Started by `scripts/06_b2d_closed_loop.sh`,
which sets CARLA_ROOT / PYTHONPATH / TRITRACK_CARLA_ARGS.
"""
import sys


def main() -> int:
    from tritrack import patches
    patches.patch_server_args()
    from leaderboard import leaderboard_evaluator
    try:
        leaderboard_evaluator.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
