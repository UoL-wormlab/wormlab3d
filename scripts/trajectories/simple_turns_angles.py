import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import GridSpec
from sklearn.decomposition import PCA

from simple_worm.plot3d import interactive
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.particles.tumble_run import find_approximation
from wormlab3d.particles.util import calculate_trajectory_frame
from wormlab3d.toolkit.util import normalise, print_args, str2bool
from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.manoeuvres import align_with_traj, get_manoeuvres
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import calculate_speeds

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)
DATA_KEYS = [
    'distances',
    'displacements',
    'durations',
    # 'speed_avg_abs',
    # 'speed_avg_signed',
    'intervals',
    'angles_t',
    'angles_n',
    'angles_p',
]

# show_plots = True
# save_plots = False
show_plots = False
save_plots = True
interactive_plots = False
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot simple turn angle statistics.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset by id.')

    # Trajectory args
    parser.add_argument('--trajectory-point', type=float, default=-1,
                        help='Number between 0 (head) and 1 (tail). Set to -1 to use centre of mass.')
    parser.add_argument('--smoothing-window', type=int, default=25,
                        help='Smooth the trajectory using average in a sliding window. Size defined in number of frames.')

    # Approximation arguments
    parser.add_argument('--approx-error', type=float,
                        help='Target approximation error.')
    parser.add_argument('--smoothing-window-K', type=int, default=101,
                        help='Curvature smoothing window.')
    parser.add_argument('--planarity-window', type=int, default=25,
                        help='Number of frames to use when calculating the planarity measure.')
    parser.add_argument('--planarity-window-vertices', type=int, default=5,
                        help='Number of vertices to use when calculating the planarity measure.')
    parser.add_argument('--approx-distance', type=int, default=500,
                        help='Initial min distance between vertices.')
    parser.add_argument('--approx-curvature-height', type=int, default=100,
                        help='Initial min height of curvature peaks to detect vertices.')
    parser.add_argument('--approx-smooth-e0', type=int, default=201,
                        help='Smoothing window for e0.')
    parser.add_argument('--approx-smooth-K', type=int, default=201,
                        help='Smoothing window for K.')
    parser.add_argument('--approx-max-attempts', type=int, default=50,
                        help='Max attempts to find an approximation.')
    parser.add_argument('--use-postures-in-pca', type=str2bool, default=False,
                        help='Fit PCA planes using full midlines (postures) rather than the trajectory point.')

    # Identifying complex turns (with reversals)
    parser.add_argument('--min-reversal-frames', type=int, default=25,
                        help='Minimum number of reversal frames to use to identify a manoeuvre.')
    parser.add_argument('--min-reversal-distance', type=float, default=0.,
                        help='Minimum reversal distance to use to identify a manoeuvre.')
    parser.add_argument('--min-forward-frames', type=int, default=25,
                        help='Minimum number of forward frames between reversals.')
    parser.add_argument('--min-forward-distance', type=float, default=0.,
                        help='Minimum forward distance between reversals.')
    parser.add_argument('--manoeuvre-window', type=int, default=100,
                        help='Number of frames to include either side of a detected manoeuvre.')

    # Simple turn arguments
    parser.add_argument('--turn-window', type=int, default=100,
                        help='Number of frames to include either side of a detected simple turn.')
    parser.add_argument('--turn-window-min', type=int, default=25,
                        help='Minimum number of frames required either side of a detected simple turn.')
    parser.add_argument('--max-cluster-turn-interval', type=int, default=25,
                        help='Maximum number of frames between turns in a cluster.')
    parser.add_argument('--max-cluster-size', type=int, default=5,
                        help='Maximum cluster size to consider.')

    args = parser.parse_args()

    print_args(args)

    return args


def _identifiers(args: Namespace, include_cluster_params: bool = False) -> str:
    id_str = f'ds={args.dataset}' \
             f'_u={args.trajectory_point}' \
             f'_sw={args.smoothing_window}' \
             f'_err={args.approx_error}' \
             f'_swk={args.smoothing_window_K}' \
             f'_pw={args.planarity_window}' \
             f'_pwv={args.planarity_window_vertices}' \
             f'_ad={args.approx_distance}' \
             f'_ah={args.approx_curvature_height}' \
             f'_ase={args.approx_smooth_e0}' \
             f'_ask={args.approx_smooth_K}' \
             f'_aa={args.approx_max_attempts}' \
             f'_rf={args.min_reversal_frames}' \
             f'_rd={args.min_reversal_distance}' \
             f'_ff={args.min_forward_frames}' \
             f'_fd={args.min_forward_distance}' \
             f'_tw={args.turn_window}' \
             f'_twm={args.turn_window_min}'

    if include_cluster_params:
        id_str += f'_cti={args.max_cluster_turn_interval}' \
                  f'_mcs={args.max_cluster_size}'

    if args.use_postures_in_pca:
        id_str += '_ppca=1'

    return id_str


def _nonp(pca: PCA) -> np.ndarray:
    r = pca.explained_variance_ratio_.T
    return r[2] / np.where(r[2] == 0, 1, np.sqrt(r[1] * r[0]))


def _calculate_trial_turn_data(
        reconstruction: Reconstruction,
        args: Namespace,
):
    """
    Find all simple turns in the trial and calculate statistics.
    """
    fps = reconstruction.trial.fps

    # Fetch reconstruction and centre
    X, _ = get_trajectory(
        reconstruction_id=reconstruction.id,
        smoothing_window=args.smoothing_window
    )
    X = X - X.mean(axis=(0, 1), keepdims=True)
    T = X.shape[0]
    N = X.shape[1]

    # Pick trajectory point
    if args.trajectory_point == -1:
        Xt = X.mean(axis=1)
    else:
        u = round(args.trajectory_point * N)
        if u == N:
            u -= 1
        assert 0 <= u < N, f'Incompatible trajectory point: {u}.'
        Xt = X[:, u]

    # Get PCAs and frame
    pcas, _ = generate_or_load_pca_cache(
        trial_id=reconstruction.trial.id,
        smoothing_window=args.smoothing_window,
        window_size=args.planarity_window,
        tracking_only=True,
    )
    e0, e1, e2 = calculate_trajectory_frame(X, pcas, args.planarity_window)

    # Find the approximation
    approx, distance, height, smooth_e0, smooth_K = find_approximation(
        X=X,
        e0=e0,
        error_limit=args.approx_error,
        planarity_window_vertices=args.planarity_window_vertices,
        distance_first=args.approx_distance,
        height_first=args.approx_curvature_height,
        smooth_e0_first=args.approx_smooth_e0,
        smooth_K_first=args.approx_smooth_K,
        max_iterations=args.approx_max_attempts,
        quiet=True
    )
    X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles_j, nonplanar_angles_j, twist_angles_j, _, _, _ = approx

    # Fetch complex turns so we can exclude them.
    # If --manoeuvre-window=-1 is used for manoeuvre-angle windowing elsewhere, still exclude using a finite buffer.
    manoeuvre_exclusion_window = args.manoeuvre_window
    if manoeuvre_exclusion_window < 0:
        manoeuvre_exclusion_window = args.turn_window if args.turn_window >= 0 else args.turn_window_min

    manoeuvres = get_manoeuvres(
        X,
        Xt,
        min_reversal_frames=args.min_reversal_frames,
        min_reversal_distance=args.min_reversal_distance,
        min_forward_frames=args.min_forward_frames,
        min_forward_distance=args.min_forward_distance,
        window_size=manoeuvre_exclusion_window,
        cut_windows_at_manoeuvres=False,
    )
    exclude_frames = np.zeros(T, dtype=bool)
    for m in manoeuvres:
        exclude_frames[m['start_idx']:m['end_idx'] + 1] = 1

    # Get the simple turn statistics
    turns = []
    signed_speeds = calculate_speeds(X, signed=True)

    def _fit_pca(start_idx: int, end_idx: int) -> PCA:
        X_seg = X[start_idx:end_idx] if args.use_postures_in_pca else Xt[start_idx:end_idx]
        if X_seg.ndim == 3:
            X_seg = X_seg.reshape((-1, 3))
        pca = PCA(svd_solver='full', copy=True, n_components=3)
        pca.fit(X_seg)
        return pca

    for i, tumble_idx in enumerate(tumble_idxs):
        if args.turn_window < 0:
            start_idx = 0
            end_idx = len(X)
        else:
            start_idx = max(0, tumble_idx - args.turn_window)
            end_idx = min(len(X), tumble_idx + args.turn_window)

        # Cut windows at turns
        if i > 0:
            start_idx = max(start_idx, tumble_idxs[i - 1])
        if i < len(tumble_idxs) - 1:
            end_idx = min(end_idx, tumble_idxs[i + 1])

        # If --turn-window=-1, clamp the run windows at manoeuvres too.
        if args.turn_window < 0:
            if exclude_frames[tumble_idx]:
                continue

            man_frames_prev = exclude_frames[start_idx:tumble_idx]
            if np.any(man_frames_prev):
                last_exclude_offset = man_frames_prev.nonzero()[0].max()
                start_idx = max(start_idx, start_idx + int(last_exclude_offset) + 1)

            man_frames_next = exclude_frames[tumble_idx:end_idx]
            if np.any(man_frames_next):
                first_exclude_offset = man_frames_next.nonzero()[0].min()
                end_idx = min(end_idx, tumble_idx + int(first_exclude_offset))

            if start_idx >= tumble_idx or end_idx <= tumble_idx:
                continue

        # Skip if the window is not big enough on either side
        if end_idx - tumble_idx < args.turn_window_min \
                or tumble_idx - start_idx < args.turn_window_min:
            continue

        # Skip if any part of the window is also part of a complex turn
        if np.any(exclude_frames[start_idx:end_idx]):
            continue

        # Calculate incoming/outgoing planes.
        pca_inc = _fit_pca(start_idx, tumble_idx)
        pca_out = _fit_pca(tumble_idx, end_idx)
        pca_all = _fit_pca(start_idx, end_idx)

        # Calculate the distance, speed and non-planarity for the whole manoeuvre window (trajectory point only).
        X_inc = Xt[start_idx:tumble_idx]
        X_out = Xt[tumble_idx:end_idx]
        X_window = Xt[start_idx:end_idx]
        duration = end_idx - start_idx
        distance = np.linalg.norm(X_window[1:] - X_window[:-1], axis=-1).sum()
        displacement = np.linalg.norm(X_window[0] - X_window[-1])
        speed_avg_abs = np.abs(signed_speeds[start_idx:end_idx]).mean()
        speed_avg_signed = signed_speeds[start_idx:end_idx].mean()

        # Get component vectors for each section
        inc_t = align_with_traj(pca_inc.components_[0], X_inc)
        out_t = align_with_traj(pca_out.components_[0], X_out)
        inc_n = align_with_traj(pca_inc.components_[2], pca_all.components_[2])
        out_n = align_with_traj(pca_out.components_[2], pca_all.components_[2])

        # Create a "normal" component to the plane formed by the two sections
        # normal = normalise(np.cross(inc_t, out_t))
        if i > 0:
            prev_run_start_idx = max(0, tumble_idxs[i - 1])
        else:
            prev_run_start_idx = 0
        if i < len(tumble_idxs) - 1:
            next_run_end_idx = min(len(Xt) - 1, tumble_idxs[i + 1])
        else:
            next_run_end_idx = len(Xt) - 1
        run_prev = Xt[tumble_idx] - Xt[prev_run_start_idx]
        run_next = Xt[next_run_end_idx] - Xt[tumble_idx]
        normal = normalise(np.cross(run_prev, run_next))

        # Calculate angles
        angle_t = calculate_angle(inc_t, out_t)
        angle_n = calculate_angle(inc_n, out_n)

        turns.append({
            'idx': tumble_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'duration': duration / fps,
            'distance': distance,
            'displacement': displacement,
            'speed_avg_abs': speed_avg_abs * fps,
            'speed_avg_signed': speed_avg_signed * fps,
            'inc_t': inc_t,
            'inc_n': inc_n,
            'out_t': out_t,
            'out_n': out_n,
            'normal': normal,
            'angle_t': angle_t,
            'angle_n': angle_n,
            'nonp_inc': _nonp(pca_inc),
            'nonp_out': _nonp(pca_out),
            'nonp_all': _nonp(pca_all),
        })

    # Calculate stats between pairs of turns
    distances = np.zeros((len(turns), len(turns)))
    displacements = np.zeros((len(turns), len(turns)))
    durations = np.zeros((len(turns), len(turns)))
    intervals = np.zeros((len(turns), len(turns)))
    angles_t = np.zeros((len(turns), len(turns)))
    angles_n = np.zeros((len(turns), len(turns)))
    angles_p = np.zeros((len(turns), len(turns)))
    for i, turn1 in enumerate(turns):
        for j, turn2 in enumerate(turns):
            if j < i:
                continue
            traj = Xt[turn1['start_idx']:turn2['end_idx']]
            distances[i, j] = np.linalg.norm(traj[1:] - traj[:-1], axis=-1).sum()
            displacements[i, j] = np.linalg.norm(traj[-1] - traj[0])
            durations[i, j] = turn2['end_idx'] - turn1['start_idx']
            intervals[i, j] = turn2['idx'] - turn1['idx']
            angles_t[i, j] = calculate_angle(turn1['inc_t'], turn2['out_t'])
            angles_n[i, j] = calculate_angle(turn1['inc_n'], turn2['out_n'])
            if i != j:
                angles_p[i, j] = calculate_angle(turn1['normal'], turn2['normal'])

    res = {
        'distances': distances,
        'displacements': displacements,
        'durations': durations,
        'intervals': intervals,
        'angles_t': angles_t,
        'angles_n': angles_n,
        'angles_p': angles_p,
    }

    return res


def _calculate_data(
        args: Namespace,
        ds: Dataset,
) -> Tuple[
    Dict[float, np.ndarray],
    Dict[float, np.ndarray]
]:
    """
    Calculate the data.
    """
    logger.info('Calculating data.')
    metas = ds.metas

    # Values
    ids = {}
    results = {}

    # Calculate the model for all trials
    for rid in metas['reconstruction']:
        reconstruction = Reconstruction.objects.get(id=rid)
        trial = reconstruction.trial
        logger.info(f'Computing tumble-run model for trial={trial.id}.')

        # Add concentration to results
        c = trial.experiment.concentration
        if c not in ids:
            ids[c] = []
            results[c] = {k: [] for k in DATA_KEYS}

        # Calculate trial and save results
        res = _calculate_trial_turn_data(reconstruction, args)
        for k in DATA_KEYS:
            results[c][k].append(res[k])

        # Add ids
        ids[c].append(trial.id)

    # Sort by concentration
    ids = {c: np.array(v) for c, v in sorted(list(ids.items()))}
    results = {c: v for c, v in sorted(list(results.items()))}

    return ids, results


def _generate_or_load_data(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Tuple[
    Dataset,
    Dict[float, np.ndarray],
    Dict[float, np.ndarray]
]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    ids = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            ids = {}
            results = {}
            for name in data.files:
                c, fn = name.split('_')
                if fn != 'ids':
                    continue
                c = float(c)
                ids[c] = data[name]
                results[c] = {
                    k: [
                        data[f'{c}_{k.replace("_", "-")}-{id_val}'] for id_val in ids[c]
                    ]
                    for k in DATA_KEYS
                }

            ids = {k: v for k, v in sorted(list(ids.items()))}
            results = {k: v for k, v in sorted(list(results.items()))}
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            ids = None
            logger.warning(f'Could not load cache: {e}')

    if ids is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        ids, results = _calculate_data(args, ds)
        data = {}
        for c, id_vals in ids.items():
            data[f'{c}_ids'] = id_vals
            for k in DATA_KEYS:
                for i, id_val in enumerate(id_vals):
                    data[f'{c}_{k.replace("_", "-")}-{id_val}'] = results[c][k][i]
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **data)

    return ds, ids, results


def plot_simple_turn_angles():
    """
    Plot the angles between simple turns.
    """
    args = parse_args()
    ds, ids, results = _generate_or_load_data(args, rebuild_cache=False, cache_only=False)
    concs = [float(c) for c in results.keys()]

    # Collate data
    logger.info('Collating data.')
    trial_distances = [d for v in [results[c]['distances'] for c in concs] for d in v]
    trial_displacements = [d for v in [results[c]['displacements'] for c in concs] for d in v]
    trial_durations = [d for v in [results[c]['durations'] for c in concs] for d in v]
    trial_intervals = [t for v in [results[c]['intervals'] for c in concs] for t in v]
    trial_angles_t = [a for v in [results[c]['angles_t'] for c in concs] for a in v]
    trial_angles_n = [a for v in [results[c]['angles_n'] for c in concs] for a in v]
    trial_angles_p = [a for v in [results[c]['angles_p'] for c in concs] for a in v]

    # Loop over the trial results and put results into clusters
    logger.info('Building clusters.')

    def _filter_extend(out, values, offset, mask):
        matched = values.diagonal(offset=offset)
        out.extend(matched[mask])

    cluster_sizes = np.arange(1, args.max_cluster_size + 1)
    distances = {c: [] for c in cluster_sizes}
    displacements = {c: [] for c in cluster_sizes}
    durations = {c: [] for c in cluster_sizes}
    angles_t = {c: [] for c in cluster_sizes}
    angles_n = {c: [] for c in cluster_sizes}
    angles_p = {c: [] for c in cluster_sizes}
    for i in range(len(trial_angles_t)):
        for c in cluster_sizes:
            # Find which turns are close enough to include in a cluster
            intv = trial_intervals[i].diagonal(offset=c - 1)
            close_enough = intv <= (c - 1) * args.max_cluster_turn_interval

            # Add results
            _filter_extend(distances[c], trial_distances[i], c - 1, close_enough)
            _filter_extend(displacements[c], trial_displacements[i], c - 1, close_enough)
            _filter_extend(durations[c], trial_durations[i], c - 1, close_enough)
            _filter_extend(angles_t[c], trial_angles_t[i], c - 1, close_enough)
            _filter_extend(angles_n[c], trial_angles_n[i], c - 1, close_enough)
            _filter_extend(angles_p[c], trial_angles_p[i], c - 1, close_enough)

    # Set up plots
    logger.info('Plotting.')
    plt.rc('axes', titlesize=7, titlepad=4)  # fontsize of the title
    plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=5)  # fontsize of the legend

    gs = GridSpec(
        nrows=2,
        ncols=args.max_cluster_size + 1,
        width_ratios=(4,) * args.max_cluster_size + (2,),
        height_ratios=(2, 3),
        wspace=0,
        hspace=0,
        top=0.88,
        bottom=0.12,
        left=0.06,
        right=0.99,
    )
    fig = plt.figure(figsize=(8, 2.4))

    cmap_traj = plt.get_cmap('autumn_r')
    cmap_planar = plt.get_cmap('winter_r')
    colour_traj = cmap_traj(.6)
    colour_planar = cmap_planar(.6)
    dists_all = np.concatenate([d for d in distances.values()])
    durations_all = np.concatenate([d for d in durations.values()])
    scatter_args = dict(s=10, alpha=0.6, vmin=durations_all.min(), vmax=durations_all.max())
    hist_args = dict(bins=10, density=True, rwidth=0.9, range=(0, 1))

    def _make_scat(ax_, angles_t_, angles_n_, distances_, durations_):
        s = ax_.scatter(np.sin(angles_t_), distances_, marker='x', cmap=cmap_traj, label='Trajectory angles',
                        c=durations_, **scatter_args)
        s2 = ax_.scatter(np.sin(angles_n_), distances_, marker='$\u25EF$', cmap=cmap_planar, label='Planar angles',
                         c=durations_, **scatter_args)
        ax_.set_xlabel('sin(Angle)')
        ax_.set_xlim(left=-0.1, right=1 + 0.1)
        ax_.set_xticks([0, 1])
        ax_.xaxis.set_label_coords(.5, -.1)
        ax_.spines['top'].set_visible(False)

        return s, s2

    def _make_angles_hist(ax_, angles_t_, angles_n_):
        ax_.hist([np.sin(angles_t_), np.sin(angles_n_)],
                 color=[colour_traj, colour_planar], **hist_args)
        ax_.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_.spines['bottom'].set(linestyle='--', color='grey')

    # Plot the clusters
    for i, c in enumerate(cluster_sizes):

        # Histograms on top
        ax_hist = fig.add_subplot(gs[0, i])
        _make_angles_hist(ax_hist, angles_t[c], angles_p[c])
        ax_hist.set_title(f'Number of turns = {c}')

        # Scatter plots below
        ax_scat = fig.add_subplot(gs[1, i])
        scat_t, scat_n = _make_scat(ax_scat, angles_t[c], angles_p[c], distances[c], durations[c])

        if i == 0:
            ax_hist.set_ylabel('Density')
            ax_scat.set_ylabel('Distance (mm)')
            ax_scat.set_ylim(bottom=0, top=dists_all.max() * 1.05)
            ax_hist_share = ax_hist
            ax_scat_share = ax_scat
        else:
            ax_hist.sharey(ax_hist_share)

            ax_hist.spines['left'].set(linestyle='--', color='grey')
            if i < len(cluster_sizes) - 1:
                ax_hist.spines['right'].set_visible(False)

            ax_hist.tick_params(axis='y', left=False, labelleft=False)
            ax_scat.sharey(ax_scat_share)
            ax_scat.spines['left'].set(linestyle='--', color='grey')
            ax_scat.tick_params(axis='y', left=False, labelleft=False)

    # Colourbars
    ax_cb = fig.add_subplot(gs[:, args.max_cluster_size])
    cax = ax_cb.inset_axes([0.04, 0.02, 0.14, 0.7], transform=ax_cb.transAxes)
    cax.spines['right'].set_visible(False)
    cb = fig.colorbar(scat_t, ax=ax_cb, cax=cax, ticks=None)
    cb.set_ticks([])
    cb.set_label('Duration (s)', rotation=270, labelpad=35, fontsize=5)
    cb.outline.set_visible(False)
    cb.solids.set(alpha=1)
    cax2 = ax_cb.inset_axes([0.18, 0.02, 0.14, 0.7], transform=ax_cb.transAxes)
    cax2.spines['left'].set_visible(False)
    cb2 = fig.colorbar(scat_n, ax=ax_cb, cax=cax2)
    cb2.outline.set_visible(False)
    cb2.solids.set(alpha=1)
    # cb2.set_ticks([5, 15, 25])
    ax_cb.spines['left'].set_visible(False)
    ax_cb.spines['bottom'].set_visible(False)
    ax_cb.axis('off')

    # Legend
    legend = ax_scat.legend()
    legend.legendHandles[0].set_color(colour_traj)
    legend.legendHandles[0].set_sizes([6.0])
    legend.legendHandles[0].set_alpha(1.)
    legend.legendHandles[1].set_color(colour_planar)
    legend.legendHandles[1].set_sizes([6.0])
    legend.legendHandles[1].set_alpha(1.)
    handles = legend.legendHandles
    labels = [t.get_text() for t in legend.get_texts()]
    ax_cb.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.93, 0.94),
                 bbox_transform=fig.transFigure)
    legend.remove()

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}_scatter_hists'
                            f'_{_identifiers(args, include_cluster_params=True)}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_dataset_ip_angles_vs_n_turns(
        which_angle: str = 'planar',
        layout: str = 'paper'
):
    """
    Plot the distributions of turn angles against number of simple turns in a dataset.
    """
    args = parse_args()
    ds, ids, results = _generate_or_load_data(args, rebuild_cache=False, cache_only=False)
    concs = [float(c) for c in results.keys()]

    # Collate data
    logger.info('Collating data.')
    trial_intervals = [t for v in [results[c]['intervals'] for c in concs] for t in v]
    use_angles = {
        'trajectory': 'angles_t',
        'planar': 'angles_n',
        'runs': 'angles_p',
    }
    trial_angles = [a for v in [results[c][use_angles[which_angle]] for c in concs] for a in v]

    # Loop over the trial results and put results into clusters
    logger.info('Building clusters.')

    def _filter_extend(out, values, offset, mask):
        matched = values.diagonal(offset=offset)
        out.extend(matched[mask])

    cluster_sizes = np.arange(1, args.max_cluster_size + 1)
    angles = {c: [] for c in cluster_sizes}
    for i in range(len(trial_angles)):
        for c in cluster_sizes:
            # Find which turns are close enough to include in a cluster
            intv = trial_intervals[i].diagonal(offset=c - 1)
            close_enough = intv <= (c - 1) * args.max_cluster_turn_interval

            # Add results
            _filter_extend(angles[c], trial_angles[i], c - 1, close_enough)

    # Divide up into bins
    data = []
    labels = []
    start_c = 2 if which_angle == 'runs' else 1
    for c in range(start_c, args.max_cluster_size + 1):
        data.append(np.arcsin(np.sin(angles[c])))
        labels.append(f'{c} turns\n({len(angles[c])})')

    # Set up plot
    if layout == 'paper':
        plt.rc('axes', titlesize=7)  # fontsize of the title
        plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=5)  # fontsize of the legend
        fig, ax = plt.subplots(1, figsize=(3, 2.13), gridspec_kw={
            'top': 0.93,
            'bottom': 0.2,
            'left': 0.13,
            'right': 0.99,
        })
        # ax.set_title(f'Number of simple turns vs {which_angle} angle', pad=3)
        xlabel_pad = 4
        ylabel_pad = 0
    else:
        plt.rc('axes', labelsize=8)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6.5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=6.5)  # fontsize of the y tick labels
        fig, ax = plt.subplots(1, figsize=(2.75, 2.3), gridspec_kw={
            'top': 0.99,
            'bottom': 0.22,
            'left': 0.17,
            'right': 0.99,
        })
        xlabel_pad = 6
        ylabel_pad = 7

    ax.set_xlabel('Number of simple turns', labelpad=xlabel_pad)
    ax.set_ylabel('Interplane angle', labelpad=ylabel_pad)
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, np.pi / 2])
    ax.set_yticklabels([0, '$\pi/2$'])
    ax.boxplot(data, labels=labels, widths=0.6)

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}_{which_angle}_candlesticks'
                            f'_{_identifiers(args, include_cluster_params=True)}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_dataset_ip_angles_vs_n_turns_with_complex_turn_comparison(
        which_angle: str = 'planar',
        layout: str = 'paper'
):
    """
    Plot the distributions of turn angles against number of simple turns in a dataset, with an additional candlestick
    showing the distribution of complex turns.
    """
    args = parse_args()
    ds, ids, results = _generate_or_load_data(args, rebuild_cache=False, cache_only=False)
    concs = [float(c) for c in results.keys()]

    # Collate data
    logger.info('Collating data.')
    trial_intervals = [t for v in [results[c]['intervals'] for c in concs] for t in v]
    use_angles = {
        'trajectory': 'angles_t',
        'planar': 'angles_n',
        'runs': 'angles_p',
    }
    trial_angles = [a for v in [results[c][use_angles[which_angle]] for c in concs] for a in v]

    # Loop over the trial results and put results into clusters
    logger.info('Building clusters.')

    def _filter_extend(out, values, offset, mask):
        matched = values.diagonal(offset=offset)
        out.extend(matched[mask])

    cluster_sizes = np.arange(1, args.max_cluster_size + 1)
    angles = {c: [] for c in cluster_sizes}
    for i in range(len(trial_angles)):
        for c in cluster_sizes:
            # Find which turns are close enough to include in a cluster
            intv = trial_intervals[i].diagonal(offset=c - 1)
            close_enough = intv <= (c - 1) * args.max_cluster_turn_interval

            # Add results
            _filter_extend(angles[c], trial_angles[i], c - 1, close_enough)

    # Divide up into bins
    data = []
    labels = []
    start_c = 2 if which_angle == 'runs' else 1
    for c in range(start_c, args.max_cluster_size + 1):
        data.append(np.arcsin(np.sin(angles[c])))
        labels.append(f'{c} turns\n({len(angles[c])})')

    # Add complex turn data
    from scripts.trajectories.manoeuvres_3d import _generate_or_load_manoeuvre_stats
    args2 = Namespace()
    args2.dataset = args.dataset
    args2.trial = None
    args2.midline3d_source = None
    args2.midline3d_source_file = None
    args2.start_frame = None
    args2.end_frame = None
    args2.directionality = None
    args2.prune_slowest_ratio = None
    args2.projection = None
    args2.tracking_only = False
    args2.rebuild_cache = False
    args2.trajectory_point = args.trajectory_point
    args2.smoothing_window = args.smoothing_window
    args2.use_postures_in_pca = args.use_postures_in_pca
    args2.planarity_window = args.planarity_window
    args2.planarity_window_vertices = args.planarity_window_vertices
    args2.smoothing_window_K = args.smoothing_window_K
    args2.approx_error_limit = args.approx_error
    args2.approx_distance = args.approx_distance
    args2.approx_curvature_height = args.approx_curvature_height
    args2.approx_max_attempts = args.approx_max_attempts
    args2.min_forward_frames = args.min_forward_frames
    args2.min_forward_distance = args.min_forward_distance
    args2.min_reversal_frames = args.min_reversal_frames
    args2.min_reversal_distance = args.min_reversal_distance
    args2.manoeuvre_window = args.turn_window
    res = _generate_or_load_manoeuvre_stats(args2, rebuild_cache=False, cache_only=False)
    complex_angles = np.arcsin(np.sin(res['angles']))
    data.append(complex_angles)
    labels.append(f'Complex turns\n({len(complex_angles)})')

    # Set up plot
    if layout == 'paper':
        plt.rc('axes', titlesize=7)  # fontsize of the title
        plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=5)  # fontsize of the legend
        fig, ax = plt.subplots(1, figsize=(3.3, 2.13), gridspec_kw={
            'top': 0.93,
            'bottom': 0.2,
            'left': 0.13,
            'right': 0.99,
        })
        # ax.set_title(f'Number of simple turns vs {which_angle} angle', pad=3)
        xlabel_pad = 4
        ylabel_pad = 0
    else:
        plt.rc('axes', labelsize=8)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6.5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=6.5)  # fontsize of the y tick labels
        fig, ax = plt.subplots(1, figsize=(2.75, 2.3), gridspec_kw={
            'top': 0.99,
            'bottom': 0.22,
            'left': 0.17,
            'right': 0.99,
        })
        xlabel_pad = 6
        ylabel_pad = 7

    ax.set_xlabel('Number of simple turns', labelpad=xlabel_pad)
    ax.set_ylabel('IP angle', labelpad=ylabel_pad)
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, np.pi / 2])
    ax.set_yticklabels([0, '$\pi/2$'])
    ax.boxplot(data, labels=labels, widths=0.6)

    if save_plots:
        path = LOGS_PATH / (f'{START_TIMESTAMP}_{which_angle}_candlesticks_with_complex_turns'
                            f'_{_identifiers(args, include_cluster_params=True)}'
                            f'_u={args2.trajectory_point}'
                            f'_rf={args2.min_reversal_frames}'
                            f'_rd={args2.min_reversal_distance}'
                            f'_ff={args2.min_forward_frames}'
                            f'_fd={args2.min_forward_distance}'
                            f'_mw={args2.manoeuvre_window}'
                            f'.{img_extension}')
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    if interactive_plots:
        interactive()

    # plot_simple_turn_angles()
    # plot_dataset_ip_angles_vs_n_turns(which_angle='planar')
    plot_dataset_ip_angles_vs_n_turns(which_angle='planar', layout='paper')
    # plot_dataset_ip_angles_vs_n_turns_with_complex_turn_comparison(which_angle='planar', layout='paper')
