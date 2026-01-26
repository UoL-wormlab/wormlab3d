from argparse import ArgumentParser, Namespace

from simple_worm.material_parameters import MP_DEFAULT_K
from wormlab3d.data.model.midline3d import M3D_SOURCES, M3D_SOURCE_MF
from wormlab3d.particles.args.parameter_args import ParameterArgs
from wormlab3d.toolkit.util import str2bool
from wormlab3d.trajectories.displacement import DISPLACEMENT_AGGREGATION_OPTIONS, DISPLACEMENT_AGGREGATION_SQUARED_SUM
from wormlab3d.trajectories.util import APPROXIMATION_METHODS, APPROXIMATION_METHOD_FIND_PEAKS

_args = None


def get_args(
        include_trajectory_options: bool = True,
        include_msd_options: bool = True,
        include_K_options: bool = True,
        include_planarity_options: bool = True,
        include_helicity_options: bool = True,
        include_manoeuvre_options: bool = True,
        include_approximation_options: bool = True,
        include_pe_options: bool = True,
        include_fractal_dim_options: bool = True,
        include_video_options: bool = True,
        include_evolution_options: bool = True,
        validate_source: bool = True
) -> Namespace:
    """
    Parse command line arguments for the trajectory scripts.
    Not all arguments are used for all scripts, but this saves duplication.
    """
    global _args
    if _args is not None:
        return _args

    parser = ArgumentParser(description='Wormlab3D trajectory script.')

    # Source arguments
    parser.add_argument('--dataset', type=str, help='Dataset id.', required=False)
    parser.add_argument('--reconstruction', type=str, help='Reconstruction id.', required=False)
    parser.add_argument('--trial', type=int, help='Trial id.', required=False)
    parser.add_argument('--trials', type=lambda s: [int(item) for item in s.split(',')], help='Trial ids.',
                        required=False)
    parser.add_argument('--midline3d-source', type=str, default=M3D_SOURCE_MF, choices=M3D_SOURCES,
                        help='Midline3D source.')
    parser.add_argument('--midline3d-source-file', type=str, help='Midline3D source file.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--resample-points', type=int, default=-1, help='Resample the curve points.')
    parser.add_argument('--rebuild-cache', type=str2bool, help='Rebuild the trajectory cache.', default=False)
    parser.add_argument('--tracking-only', type=str2bool, help='Use the tracking trajectory only.', default=False)

    # Trajectory options
    if include_trajectory_options:
        parser.add_argument('--smoothing-window', type=int,
                            help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')
        parser.add_argument('--directionality', type=str, choices=['forwards', 'backwards'],
                            help='Use only forwards/backwards frames. Default=both.')
        parser.add_argument('--prune-slowest-ratio', type=float,
                            help='Prune the slowest x% frames from the trajectory, stitching together afterwards.')
        parser.add_argument('--trajectory-point', type=float, default=None,
                            help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass. '
                                 'Leave empty (default) to return full trajectory.')
        parser.add_argument('--projection', type=str, choices=['3D', 'x', 'y', 'z', 'xy', 'yz', 'xz'], default='3D',
                            help='Use a projection of the midline, or not (default=3D).')
        parser.add_argument('--smoothing-window-curvature', type=int,
                            help='Smooth the curvature using average in a sliding window. Size defined in number of frames.')
        parser.add_argument('--eigenworms', type=str,
                            help='Eigenworms by id.')

    # MSD arguments
    if include_msd_options:
        parser.add_argument('--deltas', type=lambda s: [int(item) for item in s.split(',')], default=[1, 10, 100],
                            help='Time lag sizes.')
        parser.add_argument('--min-delta', type=int, default=1, help='Minimum time lag.')
        parser.add_argument('--max-delta', type=int, default=10000, help='Maximum time lag.')
        parser.add_argument('--delta-step', type=float, default=1, help='Step between deltas. -ve=exponential steps.')
        parser.add_argument('--min-duration', type=int, default=-1,
                            help='Min trajectory duration to include. -ve=include all.')
        parser.add_argument('--aggregation', type=str, choices=DISPLACEMENT_AGGREGATION_OPTIONS,
                            default=DISPLACEMENT_AGGREGATION_SQUARED_SUM,
                            help='Displacements can be taken as L2 norms or as the squared sum of components.')

    # K estimation arguments
    if include_K_options:
        parser.add_argument('--K0', type=float, default=MP_DEFAULT_K, help='Initial value of K for the optimiser.')
        parser.add_argument('--K-sample-frames', type=int, default=5,
                            help='Number of frames from which to calculate the K estimate.')

    # Planarity arguments
    if include_planarity_options:
        parser.add_argument('--planarity-window', type=int, default=5,
                            help='Number of frames to use when calculating the planarity measure.')

    # Helicity arguments
    if include_helicity_options:
        parser.add_argument('--helicity-window', type=int, default=50,
                            help='Number of frames to use when calculating the helicity measure.')

    # Manoeuvre arguments
    if include_manoeuvre_options:
        parser.add_argument('--min-forward-frames', type=int, default=25,
                            help='Minimum number of forward frames before counting a forward locomotion section.')
        parser.add_argument('--min-forward-speed', type=float, default=0.,
                            help='Minimum speed of forward frames before counting a forward locomotion section.')
        parser.add_argument('--min-forward-distance', type=float, default=0.,
                            help='Minimum distance of forward frames before counting a forward locomotion section.')
        parser.add_argument('--min-reversal-frames', type=int, default=25,
                            help='Minimum number of reversal frames to use to identify a manoeuvre.')
        parser.add_argument('--min-reversal-distance', type=float, default=0.,
                            help='Minimum reversal distance to use to identify a manoeuvre.')
        parser.add_argument('--manoeuvre-window', type=int, default=500,
                            help='Number of frames to include either side of a detected manoeuvre.')
        parser.add_argument('--use-postures-in-pca', type=str2bool, default=False,
                            help='Fit PCA planes using full midlines (postures) rather than the trajectory point.')

    # Approximation arguments
    if include_approximation_options:
        parser.add_argument('--approx-method', type=str, choices=APPROXIMATION_METHODS,
                            default=APPROXIMATION_METHOD_FIND_PEAKS,
                            help='Approximation algorithm.')
        parser.add_argument('--approx-error-limit', type=float,
                            help='Target approximation error.')
        parser.add_argument('--smoothing-window-K', type=int, default=101,
                            help='Curvature smoothing window.')
        parser.add_argument('--planarity-window-vertices', type=int, default=5,
                            help='Number of vertices to use when calculating the planarity measure.')
        parser.add_argument('--approx-distance', type=int, default=500,
                            help='Min distance between vertices.')
        parser.add_argument('--approx-curvature-height', type=int, default=50,
                            help='Min height of curvature peaks to detect vertices.')
        parser.add_argument('--approx-run-max-K', type=int, default=50,
                            help='Max curvature for detecting runs.')
        parser.add_argument('--approx-min-run-duration', type=int, default=50,
                            help='Min duration for detecting runs (in frames).')
        parser.add_argument('--approx-min-run-distance', type=float,
                            help='Min distance for detecting runs.')
        parser.add_argument('--approx-min-run-speed', type=float,
                            help='Min speed for detecting runs.')
        parser.add_argument('--approx-max-attempts', type=int, default=50,
                            help='Max attempts to find an approximation.')
        parser.add_argument('--approx-use-euler-angles', type=str2bool, default=True,
                            help='Use the euler angles to extract the planar/nonplanar tumble angles.')

    # Particle explorer arguments
    if include_pe_options:
        ParameterArgs.add_args(parser)
        parser.add_argument('--approx-noise', type=float,
                            help='Noise level to add to trajectories when used.')
        parser.add_argument('--npas-min', type=float, default=1e-6,
                            help='Minimum non-planar angle sigma to use for sweeping.')
        parser.add_argument('--npas-max', type=float, default=1e1,
                            help='Maximum non-planar angle sigma to use for sweeping.')
        parser.add_argument('--npas-num', type=int, default=3,
                            help='Number of non-planar angle sigmas to use for sweeping.')
        parser.add_argument('--npas-fix-one', type=str2bool, default=False,
                            help='Ensure that the value "1" is included in the npas range by shifting it so it does.')
        parser.add_argument('--vxs', type=float, default=1e-1,
                            help='Single voxel size to use.')
        parser.add_argument('--vxs-min', type=float, default=1e-1,
                            help='Minimum voxel size to use for sweeping.')
        parser.add_argument('--vxs-max', type=float, default=1e1,
                            help='Maximum voxel size to use for sweeping.')
        parser.add_argument('--vxs-num', type=int, default=3,
                            help='Number of voxel sizes to use for sweeping.')
        parser.add_argument('--durations-min', type=int, default=1,
                            help='Minimum duration to use for sweeping (in minutes).')
        parser.add_argument('--durations-max', type=int, default=60,
                            help='Maximum duration to use for sweeping (in minutes). Ignored if durations-intervals=quadratic.')
        parser.add_argument('--durations-num', type=int, default=3,
                            help='Number of durations to use for sweeping.')
        parser.add_argument('--durations-intervals', type=str, choices=['quadratic', 'exponential'],
                            default='quadratic', help='Interval between durations.')
        parser.add_argument('--pauses-min', type=int, default=0,
                            help='Minimum pause to use for sweeping (in seconds).')
        parser.add_argument('--pauses-max', type=int, default=60,
                            help='Maximum pause to use for sweeping (in seconds). Ignored if pauses-intervals=quadratic.')
        parser.add_argument('--pauses-num', type=int, default=3,
                            help='Number of pauses to use for sweeping.')
        parser.add_argument('--pauses-intervals', type=str, choices=['quadratic', 'exponential'], default='quadratic',
                            help='Interval between pauses.')
        parser.add_argument('--volume-metric', type=str, choices=['disks', 'cuboids'], default='disks',
                            help='How to calculate the volume estimates.')

    if include_fractal_dim_options:
        parser.add_argument('--fd-plateau-threshold', type=float, default=0.95,
                            help='Percentage of the best fit value to include when finding the plateau range.')
        parser.add_argument('--fd-sample-size', type=int, default=100,
                            help='Number of randomisations to average over when calculating the box dimension.')
        parser.add_argument('--fd-sf-min', type=float, default=0.9,
                            help='Minimum trajectory scaling factor.')
        parser.add_argument('--fd-sf-max', type=float, default=1.1,
                            help='Maximum trajectory scaling factor.')

    if include_video_options:
        parser.add_argument('--video-width', type=int, default=1920, help='Width of video in pixels.')
        parser.add_argument('--video-height', type=int, default=1080, help='Height of video in pixels.')

    if include_evolution_options:
        parser.add_argument('--n-generations', type=int, default=10, help='Number of generations to run.')
        parser.add_argument('--pop-size', type=int, default=5, help='Population size.')
        parser.add_argument('--de-variant', type=str, default='DE/rand/1/bin', help='DE variant.')
        parser.add_argument('--de-cr', type=float, default=0.3, help='DE crossover rate.')

    args = parser.parse_args()

    # Check trajectory point
    if include_trajectory_options and args.trajectory_point is not None:
        assert args.trajectory_point == -1 or 0 <= args.trajectory_point <= 1, 'trajectory-point must be -1 for centre of mass or between 0 and 1.'

    # Dataset, reconstruction id or trial id is usually required
    if validate_source:
        assert sum([getattr(args, k) is not None for k in ['dataset', 'reconstruction', 'trial', 'trials']]) == 1, \
            'Specify just one of dataset, reconstruction, trial OR trials.'

    _args = args

    return args
