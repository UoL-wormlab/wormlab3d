import os
from argparse import Namespace
from typing import List, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import animation
from matplotlib.axes import Axes, GridSpec
from matplotlib.ticker import LogLocator, FormatStrFormatter, NullFormatter
from mayavi import mlab
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.stats import levy_stable, ks_1samp

from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist, Arrow3D, MidpointNormalize
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset, Reconstruction, Trial
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d_mlab
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio, make_box_from_pca
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.manoeuvres import get_manoeuvres, get_forward_durations, get_forward_stats
from wormlab3d.trajectories.util import calculate_speeds, DEFAULT_FPS

DATA_CACHE_PATH = LOGS_PATH / 'cache'
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

animate = False
show_plots = True
save_plots = True
img_extension = 'svg'
fps_anim = 25
playback_speed = 4
n_revolutions = 0.5

arrow_colours = {
    'e0': 'red',
    'e1': 'blue',
    'e2': 'green',
}


def get_trajectory(args: Namespace):
    X_slice = get_trajectory_from_args(args)
    trajectory_point = args.trajectory_point
    args.trajectory_point = None
    X_full = get_trajectory_from_args(args)
    args.trajectory_point = trajectory_point
    return X_full, X_slice


def add_pca_arrows(X, pca):
    arrows = []
    # Add PCA component vectors
    centre = X.mean(axis=0)
    for i in range(2, -1, -1):
        vec = pca.components_[i] * pca.singular_values_[i] / 5
        if vec.sum() == 0:
            continue
        origin = centre - vec / 2
        arrow = Arrow3D(
            origin=origin,
            vec=vec,
            color=arrow_colours[f'e{i}'],
            mutation_scale=25,
            arrowstyle='->',
            linewidth=3,
            alpha=0.9
        )
        arrows.append(arrow)
    return arrows


def plot_manoeuvre_3d(
        X_slice: np.ndarray,
        X_full: np.ndarray = None,
        title: str = None,
        folder: str = None,
        filename: str = None,
        colours: np.ndarray = None,
        cmap: str = 'jet',
        show_colourbar: bool = False,
        ax: Axes = None,
        worm_idxs: Union[int, List[int]] = 0,
        arrows: List[Arrow3D] = None,
        planes: List[Poly3DCollection] = None,
        azim: int = -60,
        elev: int = 30,
):
    x, y, z = X_slice.T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d', azim=azim, elev=elev)

    # Scatter the vertices
    s = ax.scatter(x, y, z, c=colours, cmap=cmap, s=10, alpha=0.4, zorder=-1, norm=MidpointNormalize(midpoint=0))
    if show_colourbar:
        cb = fig.colorbar(s, shrink=0.5)
        cb.ax.tick_params(labelsize=12)
        cb.ax.set_ylabel('Speed (mm/s)', rotation=270, fontsize=14)
        cb.solids.set(alpha=1)

    # Draw lines connecting points
    points = X_slice[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
    ax.add_collection(lc)

    # Add worm
    if worm_idxs != -1:
        if type(worm_idxs) != list:
            worm_idxs = [worm_idxs, ]
        for worm_idx in worm_idxs:
            if worm_idx >= len(X_full):
                worm_idx = len(X_full) - 1
            FS = FrameSequenceNumpy(x=X_full.transpose(0, 2, 1))
            fa = FrameArtist(F=FS[int(worm_idx)], midline_opts={'zorder': 100, 's': 100})
            fa.add_midline(ax)

    # Add arrows
    if arrows is not None:
        for arrow in arrows:
            ax.add_artist(arrow)

    # Add planes
    if planes is not None:
        for plane in planes:
            ax.add_collection3d(plane)

    equal_aspect_ratio(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    if animate:
        # Aspects
        azims = np.linspace(
            start=azim - 360 * n_revolutions / 2,
            stop=azim + 360 * n_revolutions / 2,
            num=len(X_slice)
        )
        ax.view_init(azim=azims[0], elev=elev)  # elev

        def update(frame_num: int):
            # Rotate the view.
            ax.view_init(azim=azims[frame_num])

            # Update the worm
            if worm_idxs != -1:
                fa.update(FS[frame_num])
            return ()

        idxs_mask = np.array([i % playback_speed == 0 for i in np.arange(np.round(len(X_slice)))])
        frame_nums = np.arange(len(X_slice))[idxs_mask].tolist()
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frame_nums,
            blit=True,
            interval=1 / fps_anim
        )

    if save_plots:
        assert filename is not None
        if folder is not None:
            path = LOGS_PATH / f'{START_TIMESTAMP}_{folder}'
            os.makedirs(path, exist_ok=True)
            path = path / filename
        else:
            path = LOGS_PATH / f'{START_TIMESTAMP}_{filename}'
        if animate:
            metadata = dict(
                title=title,
                artist='WormLab Leeds'
            )
            save_path = path.parent / (path.name + f'_speed={playback_speed}x.mp4')
            logger.info(f'Saving animation to {save_path}.')
            ani.save(save_path, writer='ffmpeg', fps=fps_anim, metadata=metadata)
        else:
            save_path = path.with_suffix(f'.{img_extension}')
            logger.info(f'Saving plot to {save_path}.')
            plt.savefig(save_path)

    if show_plots:
        plt.show()

    plt.close(fig)


def plot_all_manoeuvres():
    """
    Plot all manoeuvres detected in the trajectory.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    signed_speeds = calculate_speeds(X_full, signed=True)
    # plt.plot(signed_speeds)
    # plt.show()
    # exit()

    manoeuvres = get_manoeuvres(
        X_full,
        X_slice,
        min_reversal_frames=args.min_reversal_frames,
        window_size=args.manoeuvre_window
    )

    folder = f'manoeuvres_trial={args.trial}_{args.midline3d_source}' \
             f'_rev={args.min_reversal_frames}' \
             f'_ws={args.manoeuvre_window}' \
             f'_sw={args.smoothing_window}'

    # Loop over manoeuvres
    for i, m in enumerate(manoeuvres):
        plane_prev = make_box_from_pca(m['X_prev'], m['pca_prev'], 'orange')
        plane_next = make_box_from_pca(m['X_next'], m['pca_next'], 'green')

        # Xm = X_slice[m['start_idx']:m['end_idx']]
        # pca_all = PCA(svd_solver='full', copy=True, n_components=3)
        # pca_all.fit(Xm)
        # plane_all = get_plane(Xm, pca_all, 'blue')

        plot_manoeuvre_3d(
            title=f'Trial {args.trial}. '
                  f'Frames {m["start_idx"]}-{m["end_idx"]}. '
                  f'Reversal duration={m["reversal_duration"] / 25:.1f}s. '
                  f'Window size={args.manoeuvre_window} '
                  f'Smoothing window={args.smoothing_window}',
            X_slice=X_slice[m['start_idx']:m['end_idx']],
            X_full=X_full[m['start_idx']:m['end_idx']],
            folder=folder,
            filename=f'frames={m["start_idx"]}-{m["end_idx"]}',
            colours=signed_speeds[m['start_idx']:m['end_idx']],
            cmap='PRGn',
            show_colourbar=False,
            worm_idxs=int((m['end_idx'] - m['start_idx']) / 2) + 50,
            # planes=[plane_prev, plane_next, plane_all],
            planes=[plane_prev, plane_next],
        )
        # exit()


def plot_angles_and_durations():
    """
    Plot the angles between the planes coming into and going out of a manoeuvre.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    manoeuvres = get_manoeuvres(
        X_full,
        X_slice,
        min_reversal_frames=args.min_reversal_frames,
        window_size=args.manoeuvre_window
    )

    # Loop over manoeuvres
    angles = []
    durations = []
    for i, m in enumerate(manoeuvres):
        angles.append(m['angle'])
        durations.append(m['reversal_duration'] / 25)

    fig, axes = plt.subplots(3, figsize=(6, 6))
    fig.suptitle(f'Trial {args.trial}. '
                 f'Num reversal frames={args.min_reversal_frames}. '
                 f'Window size={args.manoeuvre_window}')

    # Plot histogram of angles
    ax = axes[0]
    ax.hist(angles, bins=50, density=True, facecolor='green', alpha=0.75)
    ax.set_ylabel('P(angle)')
    ax.set_xlabel('Angle')

    # Plot histogram of the reversal durations
    ax = axes[1]
    ax.hist(durations, bins=50, density=True, facecolor='green', alpha=0.75)
    ax.set_ylabel('P(duration)')
    ax.set_xlabel('Duration (s)')

    # Scatter
    ax = axes[2]
    ax.scatter(x=durations, y=angles)
    ax.set_ylabel('Angles')
    ax.set_xlabel('Durations (s)')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH + '/' + START_TIMESTAMP \
               + f'_angles_v_durations' \
                 f'_trial={args.trial}_{args.midline3d_source}' \
                 f'_rev={args.min_reversal_frames}' \
                 f'_ws={args.manoeuvre_window}'
        save_path = path + f'.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_angles_and_durations_varying_parameters():
    """
    Plot the angles between the planes coming into and going out of a manoeuvre.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    reversal_frames = [25, 50, 100]
    window_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000]

    fig, axes = plt.subplots(len(reversal_frames), len(window_sizes), figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle(f'Trial {args.trial}.')

    for row_idx, rf in enumerate(reversal_frames):
        for col_idx, ws in enumerate(window_sizes):
            manoeuvres = get_manoeuvres(
                X_full,
                X_slice,
                min_reversal_frames=rf,
                window_size=ws
            )

            # Loop over manoeuvres
            angles = []
            durations = []
            for i, m in enumerate(manoeuvres):
                angles.append(m['angle'])
                durations.append(m['reversal_duration'] / 25)

            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(f'Window size={ws}')
            if col_idx == 0:
                ax.set_ylabel(f'Reversal frames={rf}')

            ax.scatter(x=durations, y=angles, s=15, alpha=0.7)

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH + '/' + START_TIMESTAMP \
               + f'_angles_v_durations' \
                 f'_trial={args.trial}_{args.midline3d_source}' \
                 f'_rev={",".join([str(rf) for rf in reversal_frames])}' \
                 f'_ws={",".join([str(ws) for ws in window_sizes])}'
        save_path = path + f'.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_manoeuvre_rate():
    """
    Plot the rate of manoeuvres.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    manoeuvres = get_manoeuvres(
        X_full,
        X_slice,
        min_reversal_frames=args.min_reversal_frames,
        window_size=args.manoeuvre_window
    )

    # Loop over manoeuvres, recording the idxs they occurred
    idxs = [m['centre_idx'] for m in manoeuvres]
    forward_durations = np.diff(idxs) / 25
    reversal_durations = np.array([m['reversal_duration'] for m in manoeuvres]) / 25

    fig, axes = plt.subplots(2, figsize=(6, 6), sharex=True, sharey=True)
    fig.suptitle(f'Trial {args.trial}. '
                 f'Min reversal frames={args.min_reversal_frames}. ')

    # Plot histogram of the forward durations
    ax = axes[0]
    ax.hist(forward_durations, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Forward duration (s)')

    # Plot histogram of the reversal durations
    ax = axes[1]
    ax.hist(reversal_durations, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Reversal duration (s)')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH + '/' + START_TIMESTAMP \
               + f'_locomotive_durations' \
                 f'_trial={args.trial}_{args.midline3d_source}' \
                 f'_rev={args.min_reversal_frames}'
        save_path = path + f'.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_dataset_distributions():
    """
    Plot the distributions for a dataset.
    """
    args = get_args(validate_source=False)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    angles = {}
    durations_fwd = {}
    durations_bck = {}
    angles_all = []
    durations_fwd_all = []
    durations_bck_all = []

    # Loop over reconstructions
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps

        X_full, X_slice = get_trajectory(args)
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            window_size=args.manoeuvre_window
        )

        # Loop over manoeuvres
        durations_fwd_r = get_forward_durations(X_full, min_forward_frames=args.min_forward_frames) / fps
        angles_r = []
        durations_bck_r = []
        for i, m in enumerate(manoeuvres):
            angles_r.append(m['angle'])
            durations_bck_r.append(m['reversal_duration'] / fps)

        # Collate
        c = reconstruction.trial.experiment.concentration
        if c not in angles:
            angles[c] = []
        if c not in durations_fwd:
            durations_fwd[c] = []
        if c not in durations_bck:
            durations_bck[c] = []
        angles[c].extend(angles_r)
        durations_fwd[c].extend(durations_fwd_r)
        durations_bck[c].extend(durations_bck_r)
        angles_all.extend(angles_r)
        durations_fwd_all.extend(durations_fwd_r)
        durations_bck_all.extend(durations_bck_r)

    # Sort by concentration
    angles = {k: v for k, v in sorted(list(angles.items()))}
    durations_fwd = {k: v for k, v in sorted(list(durations_fwd.items()))}
    durations_bck = {k: v for k, v in sorted(list(durations_bck.items()))}
    concs = [c for c, _ in angles.items()]

    # Set up plots
    n_rows = 1 + len(angles)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, n_rows * 3))
    fig.suptitle(f'Min reversal frames={args.min_reversal_frames}. ')

    # First row shows the collated results
    ax = axes[0, 0]
    ax.hist(angles_all, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Angle')

    ax = axes[0, 1]
    ax.hist(durations_fwd_all, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Forward duration (s)')

    ax = axes[0, 2]
    ax.hist(durations_bck_all, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Backward duration (s)')

    for i, c in enumerate(concs):
        ax = axes[i + 1, 0]
        ax.set_title(f'Concentration = {c}')
        ax.hist(angles[c], bins=50, density=False, facecolor='green', alpha=0.75)
        ax.set_ylabel('Count')
        ax.set_xlabel('Angle')

        ax = axes[i + 1, 1]
        ax.set_title(f'Concentration = {c}')
        ax.hist(durations_fwd[c], bins=50, density=False, facecolor='green', alpha=0.75)
        ax.set_ylabel('Count')
        ax.set_xlabel('Forward duration (s)')

        ax = axes[i + 1, 2]
        ax.set_title(f'Concentration = {c}')
        ax.hist(durations_bck[c], bins=50, density=False, facecolor='green', alpha=0.75)
        ax.set_ylabel('Count')
        ax.set_xlabel('Backward duration (s)')

    fig.tight_layout()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_durations_and_angles' \
               f'_ds={args.dataset}' \
               f'_rev={args.min_reversal_frames}' \
               f'_fwd={args.min_forward_frames}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_dataset_reversal_durations_vs_angles():
    """
    Plot the distributions of reversal durations against angles for a dataset.
    """
    args = get_args(validate_source=False)
    hist_args = dict(bins=20, density=True, rwidth=0.9)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    # Loop over manoeuvres
    angles = {
        'prev_rev_t': [],
        'prev_rev_n': [],
        'rev_next_t': [],
        'rev_next_n': [],
        'prev_next_t': [],
        'prev_next_n': [],
    }
    durations = []
    distances = []

    # Loop over reconstructions
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps

        X_full, X_slice = get_trajectory(args)
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            min_reversal_distance=args.min_reversal_distance,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=False
        )

        # Loop over manoeuvres
        for i, m in enumerate(manoeuvres):
            for k in angles.keys():
                angles[k].append(m[f'angle_{k}'])
            durations.append(m['reversal_duration'] / fps)
            distances.append(m['reversal_distance'])

    def _plot_hist(ax_scat_, ax_histx_, ax_histy_, ax_cb_, x, y, c):
        # Scatter plot
        s = ax_scat_.scatter(x, y, c=c)
        ax_scat_.set_xlabel('Angle')
        ax_scat_.set_xlim(left=-0.1, right=np.pi + 0.1)
        ax_scat_.set_xticks([0, np.pi])
        ax_scat_.set_xticklabels(['0', '$\pi$'])
        ax_scat_.spines['top'].set_visible(False)
        ax_scat_.spines['right'].set_visible(False)

        ax_histx_.hist(x, **hist_args)
        ax_histx_.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_histx_.spines['bottom'].set(linestyle='--', color='grey')

        ax_histy_.hist(y, orientation='horizontal', **hist_args)
        ax_histy_.tick_params(axis='y', left=False, labelleft=False)
        ax_histy_.spines['left'].set(linestyle='--', color='grey')

        cax = ax_cb_.inset_axes([0.06, 0.06, 0.2, 0.88], transform=ax_cb_.transAxes)
        cb = fig.colorbar(s, ax=ax_cb_, cax=cax)
        cb.set_label('Reversal\nDuration (s)', rotation=270, labelpad=25)
        ax_cb_.spines['left'].set_visible(False)
        ax_cb_.spines['bottom'].set_visible(False)
        ax_cb_.axis('off')

    for k, angles in angles.items():
        # Set up plots
        gs = GridSpec(
            nrows=2,
            ncols=2,
            width_ratios=(7, 2),
            height_ratios=(2, 7),
            wspace=0,
            hspace=0,
            top=0.93,
            bottom=0.08,
            left=0.08,
            right=0.98,
        )
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle({
                         'prev_rev_t': 'Trajectory angles between incoming and reversal.',
                         'prev_rev_n': 'Planar angles between incoming and reversal.',
                         'rev_next_t': 'Trajectory angles between reversal and outgoing.',
                         'rev_next_n': 'Planar angles between reversal and outgoing.',
                         'prev_next_t': 'Trajectory angles between incoming and outgoing.',
                         'prev_next_n': 'Planar angles between incoming and outgoing.',
                     }[k])

        # Plot angles vs reversal durations
        ax_scat = fig.add_subplot(gs[1, 0])
        ax_scat.set_ylabel('Reversal distance (mm)')
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scat)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scat)
        ax_cb = fig.add_subplot(gs[0, 1])
        _plot_hist(ax_scat, ax_histx, ax_histy, ax_cb, angles, distances, durations)

        if save_plots:
            fn = START_TIMESTAMP \
                 + f'_rev_dist_vs_angles_{k}' \
                   f'_ds={args.dataset}' \
                   f'_sw={args.smoothing_window}' \
                   f'_rf={args.min_reversal_frames}' \
                   f'_rd={args.min_reversal_distance}' \
                   f'_ws={args.manoeuvre_window}'
            save_path = LOGS_PATH / (fn + f'.{img_extension}')
            logger.info(f'Saving plot to {save_path}.')
            plt.savefig(save_path)

        if show_plots:
            plt.show()
        plt.close(fig)


def plot_dataset_reversal_distance_vs_prev_next_angles():
    """
    Plot the distributions of reversal distance against the incoming/outgoing
    trajectory and planar angles for a dataset.
    """
    args = get_args(validate_source=False)
    hist_args = dict(bins=10, density=True, rwidth=0.9)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    # Loop over manoeuvres
    traj_angles = []
    planar_angles = []
    durations = []
    distances = []

    # Loop over reconstructions
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps

        X_full, X_slice = get_trajectory(args)
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            min_reversal_distance=args.min_reversal_distance,
            min_forward_frames=args.min_forward_frames,
            min_forward_distance=args.min_forward_distance,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=True,
            align_components_with_traj=True
        )

        # Loop over manoeuvres
        for i, m in enumerate(manoeuvres):
            traj_angles.append(m['angle_prev_next_t'])
            planar_angles.append(m['angle_prev_next_n'])
            durations.append(m['reversal_duration'] / fps)
            distances.append(m['reversal_distance'])

    # Set up plots
    plt.rc('axes', titlesize=7)  # fontsize of the title
    plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=5)  # fontsize of the legend

    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=(4, 2),
        height_ratios=(2, 3),
        wspace=0,
        hspace=0,
        top=0.93,
        bottom=0.12,
        left=0.12,
        right=0.95,
    )
    fig = plt.figure(figsize=(2.94, 2.176))
    fig.suptitle('Angles between pre- and post-manoeuvre trajectories', fontsize=6)

    cmap_traj = plt.get_cmap('autumn_r')
    cmap_planar = plt.get_cmap('winter_r')
    colour_traj = cmap_traj(.6)
    colour_planar = cmap_planar(.6)

    # Scatter plot
    ax_scat = fig.add_subplot(gs[1, 0])
    scatter_args = dict(s=6, c=durations, alpha=0.6)
    s = ax_scat.scatter(traj_angles, distances, marker='x', cmap=cmap_traj, label='Trajectory angles', **scatter_args)
    s2 = ax_scat.scatter(planar_angles, distances, marker='$\u25EF$', cmap=cmap_planar, label='IP angles',
                         **scatter_args)
    ax_scat.set_xlabel('Angle')
    ax_scat.set_xlim(left=-0.1, right=np.pi + 0.1)
    ax_scat.set_xticks([0, np.pi])
    ax_scat.set_xticklabels(['0', '$\pi$'])
    ax_scat.xaxis.set_label_coords(.5, -.1)
    ax_scat.set_ylabel('Reversal distance (mm)')
    ax_scat.yaxis.set_label_coords(-.15, .5)
    ax_scat.set_yticks([0.5, 1, 1.5])
    legend = ax_scat.legend()
    legend.legendHandles[0].set_color(colour_traj)
    legend.legendHandles[0].set_sizes([6.0])
    legend.legendHandles[0].set_alpha(1.)
    legend.legendHandles[1].set_color(colour_planar)
    legend.legendHandles[1].set_sizes([6.0])
    legend.legendHandles[1].set_alpha(1.)
    ax_scat.spines['top'].set_visible(False)
    ax_scat.spines['right'].set_visible(False)

    ax_hist_angles = fig.add_subplot(gs[0, 0], sharex=ax_scat)
    ax_hist_angles.hist([traj_angles, planar_angles],
                        color=[colour_traj, colour_planar], **hist_args)
    ax_hist_angles.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_hist_angles.spines['bottom'].set(linestyle='--', color='grey')
    ax_hist_angles.set_ylabel('Density')
    ax_hist_angles.set_yticks([0, 0.4, 0.8])
    ax_hist_angles.yaxis.set_label_coords(-.15, .5)

    ax_hist_dists = fig.add_subplot(gs[1, 1], sharey=ax_scat)
    ax_hist_dists.hist(distances, orientation='horizontal', color='green', **hist_args)
    ax_hist_dists.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_dists.spines['left'].set(linestyle='--', color='grey')
    ax_hist_dists.set_xlabel('Density')
    ax_hist_dists.set_xticks([0, 2])
    ax_hist_dists.xaxis.set_label_coords(.5, -.15)

    ax_cb = fig.add_subplot(gs[0, 1])
    cax = ax_cb.inset_axes([0.04, 0.02, 0.14, 0.6], transform=ax_cb.transAxes)
    cax.spines['right'].set_visible(False)
    cb = fig.colorbar(s, ax=ax_cb, cax=cax, ticks=None)
    cb.set_ticks([])
    cb.set_label('Reversal\nDuration (s)', rotation=270, labelpad=35, fontsize=5)
    cb.outline.set_visible(False)
    cb.solids.set(alpha=1)
    cax2 = ax_cb.inset_axes([0.18, 0.02, 0.14, 0.6], transform=ax_cb.transAxes)
    cax2.spines['left'].set_visible(False)
    cb2 = fig.colorbar(s2, ax=ax_cb, cax=cax2)
    cb2.outline.set_visible(False)
    cb2.solids.set(alpha=1)
    cb2.set_ticks([5, 15, 25])
    ax_cb.spines['left'].set_visible(False)
    ax_cb.spines['bottom'].set_visible(False)
    ax_cb.axis('off')

    handles = legend.legendHandles
    labels = [t.get_text() for t in legend.get_texts()]
    ax_cb.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.83, 0.96),
                 handletextpad=0.4, labelspacing=0.3, bbox_transform=fig.transFigure)
    legend.remove()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_rev_dist_vs_inout_angles' \
               f'_ds={args.dataset}' \
               f'_u={args.trajectory_point}' \
               f'_sw={args.smoothing_window}' \
               f'_rf={args.min_reversal_frames}' \
               f'_rd={args.min_reversal_distance}' \
               f'_ff={args.min_forward_frames}' \
               f'_fd={args.min_forward_distance}' \
               f'_mw={args.manoeuvre_window}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_dataset_reversal_durations_vs_rev_angles():
    """
    Plot the distributions of reversal durations against the
    incoming/rev and rev/outgoing angles for a dataset.
    """
    args = get_args(validate_source=False)
    hist_args = dict(bins=10, density=True, rwidth=0.9)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    # Loop over manoeuvres
    traj_angles_in_rev = []
    traj_angles_rev_out = []
    planar_angles_in_rev = []
    planar_angles_rev_out = []
    durations = []
    distances = []

    # Loop over reconstructions
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps

        X_full, X_slice = get_trajectory(args)
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            min_reversal_distance=args.min_reversal_distance,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=True
        )

        # Loop over manoeuvres
        for i, m in enumerate(manoeuvres):
            traj_angles_in_rev.append(m['angle_prev_rev_t'])
            planar_angles_in_rev.append(m['angle_prev_rev_n'])
            traj_angles_rev_out.append(m['angle_rev_next_t'])
            planar_angles_rev_out.append(m['angle_rev_next_n'])
            durations.append(m['reversal_duration'] / fps)
            distances.append(m['reversal_distance'])

    # Set up plots
    plt.rc('axes', titlesize=7)  # fontsize of the title
    plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=5)  # fontsize of the legend

    gs = GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=(4, 4, 2),
        height_ratios=(2, 3),
        wspace=0,
        hspace=0,
        top=0.89,
        bottom=0.12,
        left=0.08,
        right=0.99,
    )
    fig = plt.figure(figsize=(6, 2.4))

    cmap_traj = plt.get_cmap('autumn_r')
    cmap_planar = plt.get_cmap('winter_r')
    colour_traj = cmap_traj(.6)
    colour_planar = cmap_planar(.6)
    scatter_args = dict(s=10, c=durations, alpha=0.6)

    def _make_scat(ax_, traj_angles, planar_angles):
        s = ax_.scatter(traj_angles, distances, marker='x', cmap=cmap_traj, label='Trajectory angles', **scatter_args)
        s2 = ax_.scatter(planar_angles, distances, marker='o', cmap=cmap_planar, label='Planar angles', **scatter_args)
        ax_.set_xlabel('Angle')
        ax_.set_xlim(left=-0.1, right=np.pi + 0.1)
        ax_.set_xticks([0, np.pi])
        ax_.set_xticklabels(['0', '$\pi$'])
        ax_.xaxis.set_label_coords(.5, -.1)
        ax_.spines['top'].set_visible(False)
        ax_.spines['right'].set_visible(False)

        return s, s2

    # Scatter plot - incoming/reversal
    ax_scat_in_rev = fig.add_subplot(gs[1, 0])
    s_ir, s2_ir = _make_scat(ax_scat_in_rev, traj_angles_in_rev, planar_angles_in_rev)
    ax_scat_in_rev.set_ylabel('Reversal distance (mm)')
    ax_scat_in_rev.yaxis.set_label_coords(-.15, .5)
    # ax_scat_in_rev.set_yticks([0.5, 1, 1.5])

    # Scatter plot - reversal/outgoing
    ax_scat_rev_out = fig.add_subplot(gs[1, 1])
    _make_scat(ax_scat_rev_out, traj_angles_rev_out, planar_angles_rev_out)
    ax_scat_rev_out.tick_params(axis='y', left=False, labelleft=False)
    ax_scat_rev_out.spines['left'].set(linestyle='--', color='grey')

    # Get legend
    legend = ax_scat_in_rev.legend()
    legend.legendHandles[0].set_color(colour_traj)
    legend.legendHandles[0].set_sizes([6.0])
    legend.legendHandles[0].set_alpha(1.)
    legend.legendHandles[1].set_color(colour_planar)
    legend.legendHandles[1].set_sizes([6.0])
    legend.legendHandles[1].set_alpha(1.)

    def _make_angles_hist(ax_, traj_angles, planar_angles):
        ax_.hist([traj_angles, planar_angles],
                 color=[colour_traj, colour_planar], **hist_args)
        ax_.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_.spines['bottom'].set(linestyle='--', color='grey')

    # Angles in/rev histogram
    ax_hist_angles_ir = fig.add_subplot(gs[0, 0], sharex=ax_scat_in_rev)
    _make_angles_hist(ax_hist_angles_ir, traj_angles_in_rev, planar_angles_in_rev)
    ax_hist_angles_ir.set_ylabel('Density')
    # ax_hist_angles_ir.set_yticks([0, 0.2, 0.4])
    ax_hist_angles_ir.yaxis.set_label_coords(-.15, .5)
    ax_hist_angles_ir.set_title('Angles between incoming\nand reversal trajectories.', fontsize=6)

    # Angles rev/out histogram
    ax_hist_angles_ro = fig.add_subplot(gs[0, 1], sharex=ax_scat_in_rev, sharey=ax_hist_angles_ir)
    _make_angles_hist(ax_hist_angles_ro, traj_angles_rev_out, planar_angles_rev_out)
    ax_hist_angles_ro.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_angles_ro.spines['left'].set(linestyle='--', color='grey')
    ax_hist_angles_ro.set_title('Angles between reversal\nand outgoing trajectories.', fontsize=6)

    # Distances histogram
    ax_hist_dists = fig.add_subplot(gs[1, 2], sharey=ax_scat_in_rev)
    ax_hist_dists.hist(distances, orientation='horizontal', color='green', **hist_args)
    ax_hist_dists.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_dists.spines['left'].set(linestyle='--', color='grey')
    ax_hist_dists.set_xlabel('Density')
    ax_hist_dists.set_xticks([0, 2])
    ax_hist_dists.xaxis.set_label_coords(.5, -.15)

    ax_cb = fig.add_subplot(gs[0, 2])
    cax = ax_cb.inset_axes([0.04, 0.02, 0.14, 0.7], transform=ax_cb.transAxes)
    cax.spines['right'].set_visible(False)
    cb = fig.colorbar(s_ir, ax=ax_cb, cax=cax, ticks=None)
    cb.set_ticks([])
    cb.set_label('Reversal\nDuration (s)', rotation=270, labelpad=35, fontsize=5)
    cb.outline.set_visible(False)
    cb.solids.set(alpha=1)
    cax2 = ax_cb.inset_axes([0.18, 0.02, 0.14, 0.7], transform=ax_cb.transAxes)
    cax2.spines['left'].set_visible(False)
    cb2 = fig.colorbar(s2_ir, ax=ax_cb, cax=cax2)
    cb2.outline.set_visible(False)
    cb2.solids.set(alpha=1)
    cb2.set_ticks([5, 15, 25])
    ax_cb.spines['left'].set_visible(False)
    ax_cb.spines['bottom'].set_visible(False)
    ax_cb.axis('off')

    handles = legend.legendHandles
    labels = [t.get_text() for t in legend.get_texts()]
    ax_cb.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.9, 0.94),
                 bbox_transform=fig.transFigure)
    legend.remove()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_rev_dist_vs_rev_angles' \
               f'_ds={args.dataset}' \
               f'_rev={args.min_reversal_frames}' \
               f'_revd={args.min_reversal_distance}' \
               f'_u={args.trajectory_point}' \
               f'_sw={args.smoothing_window}' \
               f'_mw={args.manoeuvre_window}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_dataset_reversal_durations_vs_angles_combined():
    """
    Plot the distributions of reversal durations against the
    incoming/rev, rev/outgoing and incoming/outgoing angles for a dataset.
    """
    args = get_args(validate_source=False)
    hist_args = dict(bins=10, density=True, rwidth=0.9)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    # Loop over manoeuvres
    traj_angles_in_rev = []
    traj_angles_rev_out = []
    traj_angles_in_out = []
    planar_angles_in_rev = []
    planar_angles_rev_out = []
    planar_angles_in_out = []
    durations = []
    distances = []

    # Loop over reconstructions
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps

        X_full, X_slice = get_trajectory(args)
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            min_reversal_distance=args.min_reversal_distance,
            min_forward_frames=args.min_forward_frames,
            min_forward_distance=args.min_forward_distance,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=True,
            align_components_with_traj=True
        )

        # Loop over manoeuvres
        for i, m in enumerate(manoeuvres):
            traj_angles_in_rev.append(m['angle_prev_rev_t'])
            planar_angles_in_rev.append(m['angle_prev_rev_n'])
            traj_angles_rev_out.append(m['angle_rev_next_t'])
            planar_angles_rev_out.append(m['angle_rev_next_n'])
            traj_angles_in_out.append(m['angle_prev_next_t'])
            planar_angles_in_out.append(m['angle_prev_next_n'])
            durations.append(m['reversal_duration'] / fps)
            distances.append(m['reversal_distance'])

    # Set up plots
    plt.rc('axes', titlesize=7, titlepad=4)  # fontsize of the title
    plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=5)  # fontsize of the legend

    gs = GridSpec(
        nrows=2,
        ncols=4,
        width_ratios=(4, 4, 4, 2),
        height_ratios=(2, 3),
        wspace=0,
        hspace=0,
        top=0.88,
        bottom=0.12,
        left=0.06,
        right=0.99,
    )
    fig = plt.figure(figsize=(6, 2.4))

    cmap_traj = plt.get_cmap('autumn_r')
    cmap_planar = plt.get_cmap('winter_r')
    colour_traj = cmap_traj(.6)
    colour_planar = cmap_planar(.6)
    scatter_args = dict(s=10, c=durations, alpha=0.6)

    def _make_scat(ax_, traj_angles, planar_angles):
        s = ax_.scatter(traj_angles, distances, marker='x', cmap=cmap_traj, label='Trajectory angles',
                        **scatter_args)
        s2 = ax_.scatter(planar_angles, distances, marker='$\u25EF$', cmap=cmap_planar, label='Planar angles',
                         **scatter_args)
        ax_.set_xlabel('Angle')
        ax_.set_xlim(left=-0.1, right=np.pi + 0.1)
        ax_.set_xticks([0, np.pi])
        ax_.set_xticklabels(['0', '$\pi$'])
        ax_.xaxis.set_label_coords(.5, -.1)
        ax_.spines['top'].set_visible(False)
        ax_.spines['right'].set_visible(False)

        return s, s2

    # Scatter plot - incoming/reversal
    ax_scat_in_rev = fig.add_subplot(gs[1, 0])
    s_ir, s2_ir = _make_scat(ax_scat_in_rev, traj_angles_in_rev, planar_angles_in_rev)
    ax_scat_in_rev.set_ylabel('Reversal distance (mm)')
    ax_scat_in_rev.yaxis.set_label_coords(-.15, .5)
    ax_scat_in_rev.set_yticks([0.5, 1, 1.5])

    # Scatter plot - reversal/outgoing
    ax_scat_rev_out = fig.add_subplot(gs[1, 1])
    _make_scat(ax_scat_rev_out, traj_angles_rev_out, planar_angles_rev_out)
    ax_scat_rev_out.tick_params(axis='y', left=False, labelleft=False)
    ax_scat_rev_out.spines['left'].set(linestyle='--', color='grey')

    # Scatter plot - incoming/outgoing
    ax_scat_in_out = fig.add_subplot(gs[1, 2])
    _make_scat(ax_scat_in_out, traj_angles_in_out, planar_angles_in_out)
    ax_scat_in_out.tick_params(axis='y', left=False, labelleft=False)
    ax_scat_in_out.spines['left'].set(linestyle='--', color='grey')

    # Get legend
    legend = ax_scat_in_rev.legend()
    legend.legendHandles[0].set_color(colour_traj)
    legend.legendHandles[0].set_sizes([6.0])
    legend.legendHandles[0].set_alpha(1.)
    legend.legendHandles[1].set_color(colour_planar)
    legend.legendHandles[1].set_sizes([6.0])
    legend.legendHandles[1].set_alpha(1.)

    def _make_angles_hist(ax_, traj_angles, planar_angles):
        ax_.hist([traj_angles, planar_angles],
                 color=[colour_traj, colour_planar], **hist_args)
        ax_.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_.spines['bottom'].set(linestyle='--', color='grey')

    # Angles in/rev histogram
    ax_hist_angles_ir = fig.add_subplot(gs[0, 0], sharex=ax_scat_in_rev)
    _make_angles_hist(ax_hist_angles_ir, traj_angles_in_rev, planar_angles_in_rev)
    ax_hist_angles_ir.set_ylabel('Density')
    ax_hist_angles_ir.set_yticks([0, 0.4, 0.8, 1.2])
    ax_hist_angles_ir.yaxis.set_label_coords(-.15, .5)
    ax_hist_angles_ir.set_title('Angles between incoming\nand reversal trajectories', fontsize=6)

    # Angles rev/out histogram
    ax_hist_angles_ro = fig.add_subplot(gs[0, 1], sharex=ax_scat_in_rev, sharey=ax_hist_angles_ir)
    _make_angles_hist(ax_hist_angles_ro, traj_angles_rev_out, planar_angles_rev_out)
    ax_hist_angles_ro.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_angles_ro.spines['left'].set(linestyle='--', color='grey')
    ax_hist_angles_ro.set_title('Angles between reversal\nand outgoing trajectories', fontsize=6)

    # Angles in/out histogram
    ax_hist_angles_io = fig.add_subplot(gs[0, 2], sharex=ax_scat_in_rev, sharey=ax_hist_angles_ir)
    _make_angles_hist(ax_hist_angles_io, traj_angles_in_out, planar_angles_in_out)
    ax_hist_angles_io.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_angles_io.spines['left'].set(linestyle='--', color='grey')
    ax_hist_angles_io.set_title('Angles between incoming\nand outgoing trajectories', fontsize=6)

    # Distances histogram
    ax_hist_dists = fig.add_subplot(gs[1, 3], sharey=ax_scat_in_rev)
    ax_hist_dists.hist(distances, orientation='horizontal', color='green', **hist_args)
    ax_hist_dists.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_dists.spines['left'].set(linestyle='--', color='grey')
    ax_hist_dists.set_xlabel('Density')
    ax_hist_dists.set_xticks([0, 2])
    ax_hist_dists.xaxis.set_label_coords(.5, -.15)

    ax_cb = fig.add_subplot(gs[0, 3])
    cax = ax_cb.inset_axes([0.04, 0.02, 0.14, 0.7], transform=ax_cb.transAxes)
    cax.spines['right'].set_visible(False)
    cb = fig.colorbar(s_ir, ax=ax_cb, cax=cax, ticks=None)
    cb.set_ticks([])
    cb.set_label('Reversal\nDuration (s)', rotation=270, labelpad=35, fontsize=5)
    cb.outline.set_visible(False)
    cb.solids.set(alpha=1)
    cax2 = ax_cb.inset_axes([0.18, 0.02, 0.14, 0.7], transform=ax_cb.transAxes)
    cax2.spines['left'].set_visible(False)
    cb2 = fig.colorbar(s2_ir, ax=ax_cb, cax=cax2)
    cb2.outline.set_visible(False)
    cb2.solids.set(alpha=1)
    cb2.set_ticks([5, 15, 25])
    ax_cb.spines['left'].set_visible(False)
    ax_cb.spines['bottom'].set_visible(False)
    ax_cb.axis('off')

    handles = legend.legendHandles
    labels = [t.get_text() for t in legend.get_texts()]
    ax_cb.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.93, 0.94),
                 bbox_transform=fig.transFigure)
    legend.remove()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_rev_dist_vs_all_angles' \
               f'_ds={args.dataset}' \
               f'_u={args.trajectory_point}' \
               f'_sw={args.smoothing_window}' \
               f'_rf={args.min_reversal_frames}' \
               f'_rd={args.min_reversal_distance}' \
               f'_ff={args.min_forward_frames}' \
               f'_fd={args.min_forward_distance}' \
               f'_mw={args.manoeuvre_window}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def _manoeuvre_angles_stats_identifiers(args: Namespace) -> str:
    return f'manoeuvre_angles_ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_sw={args.smoothing_window}' \
           f'_rf={args.min_reversal_frames}' \
           f'_rd={args.min_reversal_distance}' \
           f'_ff={args.min_forward_frames}' \
           f'_fd={args.min_forward_distance}' \
           f'_mw={args.manoeuvre_window}'


def _generate_or_load_manoeuvre_angles_stats(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _manoeuvre_stats_identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    keys = [
        'angle_prev_rev_t',
        'angle_prev_rev_n',
        'angle_rev_next_t',
        'angle_rev_next_n',
        'angle_prev_next_t',
        'angle_prev_next_n',
        'reversal_duration',
        'reversal_distance',
    ]

    res = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            res = {}
            for k in keys:
                res[k] = data[k]
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            res = None
            logger.warning(f'Could not load cache: {e}')

    if res is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        res = {k: [] for k in keys}

        # Loop over reconstructions
        for r_ref in ds.reconstructions:
            reconstruction = Reconstruction.objects.get(id=r_ref.id)
            args.reconstruction = reconstruction.id
            fps = reconstruction.trial.fps
            X_full, X_slice = get_trajectory(args)

            manoeuvres = get_manoeuvres(
                X_full,
                X_slice,
                min_reversal_frames=args.min_reversal_frames,
                min_reversal_distance=args.min_reversal_distance,
                min_forward_frames=args.min_forward_frames,
                min_forward_distance=args.min_forward_distance,
                window_size=args.manoeuvre_window,
                cut_windows_at_manoeuvres=True,
                align_components_with_traj=True
            )

            # Loop over manoeuvres
            for i, m in enumerate(manoeuvres):
                for k in keys:
                    if k == 'reversal_duration':
                        res[k].append(m[k] / fps)
                    else:
                        res[k].append(m[k])

        for k in keys:
            res[k] = np.array(res[k])
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **res)

    return res


def plot_dataset_reversal_durations_vs_angles_combined_basic():
    """
    Plot the distributions of reversal durations against the
    incoming/rev, rev/outgoing and incoming/outgoing angles for a dataset.
    """
    args = get_args(validate_source=False)
    assert args.dataset is not None

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    # Unpack results
    res = _generate_or_load_manoeuvre_angles_stats(args, rebuild_cache=False, cache_only=False)
    traj_angles_in_rev = res['angle_prev_rev_t']
    traj_angles_rev_out = res['angle_rev_next_t']
    traj_angles_in_out = res['angle_prev_next_t']
    planar_angles_in_rev = res['angle_prev_rev_n']
    planar_angles_rev_out = res['angle_rev_next_n']
    planar_angles_in_out = res['angle_prev_next_n']

    # Set up plots
    plt.rc('axes', titlesize=7, titlepad=3)  # fontsize of the title
    plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=5)  # fontsize of the legend

    gs = GridSpec(
        nrows=1,
        ncols=3,
        wspace=0,
        top=0.83,
        bottom=0.16,
        left=0.06,
        right=0.99,
    )
    fig = plt.figure(figsize=(6, 1.4))

    hist_args = dict(bins=10, density=True, rwidth=0.9)
    cmap_traj = plt.get_cmap('autumn_r')
    cmap_planar = plt.get_cmap('winter_r')
    colour_traj = cmap_traj(.6)
    colour_planar = cmap_planar(.6)

    def _make_angles_hist(ax_, traj_angles, planar_angles, label=None):
        ax_.hist([traj_angles, planar_angles],
                 color=[colour_traj, colour_planar],
                 label=label,
                 **hist_args
                 )
        ax_.set_xlabel('Angle', labelpad=-5)
        ax_.set_xlim(left=-0.1, right=np.pi + 0.1)
        ax_.set_xticks([0, np.pi])
        ax_.set_xticklabels(['0', '$\pi$'])

    # Angles in/rev histogram
    ax_hist_angles_ir = fig.add_subplot(gs[0, 0])
    _make_angles_hist(ax_hist_angles_ir, traj_angles_in_rev, planar_angles_in_rev)
    ax_hist_angles_ir.set_ylabel('Density', labelpad=0)
    ax_hist_angles_ir.set_yticks([0, 0.4, 0.8, 1.2])
    ax_hist_angles_ir.yaxis.set_label_coords(-.15, .5)
    ax_hist_angles_ir.set_title('Angles between incoming\nand reversal trajectories', fontsize=6)

    # Angles rev/out histogram
    ax_hist_angles_ro = fig.add_subplot(gs[0, 1], sharey=ax_hist_angles_ir)
    _make_angles_hist(ax_hist_angles_ro, traj_angles_rev_out, planar_angles_rev_out)
    ax_hist_angles_ro.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_angles_ro.spines['left'].set(linestyle='--', color='grey')
    ax_hist_angles_ro.set_title('Angles between reversal\nand outgoing trajectories', fontsize=6)

    # Angles in/out histogram
    ax_hist_angles_io = fig.add_subplot(gs[0, 2], sharey=ax_hist_angles_ir)
    _make_angles_hist(ax_hist_angles_io, traj_angles_in_out, planar_angles_in_out,
                      label=['Trajectory angles', 'IP angles'])
    ax_hist_angles_io.tick_params(axis='y', left=False, labelleft=False)
    ax_hist_angles_io.spines['left'].set(linestyle='--', color='grey')
    ax_hist_angles_io.set_title('Angles between incoming\nand outgoing trajectories', fontsize=6)
    ax_hist_angles_io.legend()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_rev_dist_vs_all_angles_basic' \
               f'_ds={args.dataset}' \
               f'_u={args.trajectory_point}' \
               f'_sw={args.smoothing_window}' \
               f'_rf={args.min_reversal_frames}' \
               f'_rd={args.min_reversal_distance}' \
               f'_ff={args.min_forward_frames}' \
               f'_fd={args.min_forward_distance}' \
               f'_mw={args.manoeuvre_window}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_angle_correlations():
    """
    Plot scatter plots of the angle correlations.
    """
    args = get_args(validate_source=False)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    # Loop over reconstructions
    data = {}
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps
        c = reconstruction.trial.experiment.concentration
        if c not in data:
            data[c] = {
                f'angle_{angle_from}_{angle_to}_{t_or_n}': []
                for t_or_n in ['t', 'n']
                for angle_from in ['prev', 'rev']
                for angle_to in ['rev', 'next'] if angle_from != angle_to
            }
            data[c]['reversal_duration'] = []
            data[c]['reversal_distance'] = []

        X_full, X_slice = get_trajectory(args)
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            min_reversal_distance=args.min_reversal_distance,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=True,
            align_components_with_traj=True
        )

        # Loop over manoeuvres
        for i, m in enumerate(manoeuvres):

            for t_or_n in ['t', 'n']:
                for angle_from in ['prev', 'rev']:
                    for angle_to in ['rev', 'next']:
                        if angle_from == angle_to:
                            continue
                        k = f'angle_{angle_from}_{angle_to}_{t_or_n}'
                        data[c][k].append(m[k])
            data[c]['reversal_duration'].append(m['reversal_duration'] / fps)
            data[c]['reversal_distance'].append(m['reversal_distance'])

    # Sort by concentration
    data = {k: v for k, v in sorted(list(data.items()))}
    concs = [k for k in data.keys()]

    # Set up plots
    fig, axes = plt.subplots(len(data), 6, figsize=(9, len(data) + 2))
    scatter_args = dict(s=10, marker='x', alpha=0.6)

    def _make_scat(ax_, angles_from, angles_to, angles_from_lbl, angles_to_lbl):
        s = ax_.scatter(angles_from, angles_to, **scatter_args)
        ax_.set_xlabel(angles_from_lbl, labelpad=0)
        ax_.set_ylabel(angles_to_lbl, labelpad=0)
        ax_.set_xlim(left=-0.1, right=np.pi + 0.1)
        ax_.set_ylim(bottom=-0.1, top=np.pi + 0.1)
        ax_.set_xticks([0, np.pi])
        ax_.set_yticks([0, np.pi])
        ax_.set_xticklabels(['0', '$\pi$'])
        ax_.set_yticklabels(['0', '$\pi$'])
        return s

    for i, c in enumerate(concs):
        logger.info(f'Plotting concentration {c}.')
        for j, t_or_n in enumerate(['t', 'n']):
            for k, (angles1, angles2) in enumerate(
                    [['prev_rev', 'rev_next'], ['rev_next', 'prev_next'], ['prev_rev', 'prev_next']]):
                ax = axes[i, (j * 3) + k]
                k1 = f'angle_{angles1}_{t_or_n}'
                k2 = f'angle_{angles2}_{t_or_n}'
                _make_scat(ax, data[c][k1], data[c][k2], angles1, angles2)

    fig.tight_layout()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_angle_correlations' \
               f'_ds={args.dataset}' \
               f'_rev={args.min_reversal_frames}' \
               f'_revd={args.min_reversal_distance}' \
               f'_u={args.trajectory_point}' \
               f'_sw={args.smoothing_window}' \
               f'_mw={args.manoeuvre_window}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_dataset_speeds_vs_nonp():
    """
    Plot the distributions of speeds against nonp for turns and runs in a dataset.
    Similar to the method in trajectories/planarity but using reversals to identify turn events instead of approximations.
    """
    args = get_args(validate_source=False)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    nonp_turns = []
    durations_turns = []
    distances_turns = []
    speeds_turns = []
    nonp_runs = []
    durations_runs = []
    distances_runs = []
    speeds_runs = []

    # Loop over reconstructions
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps
        X_full, X_slice = get_trajectory(args)

        # Loop over manoeuvres to get the turn statistics
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            min_forward_frames=args.min_forward_frames,
            min_forward_distance=args.min_forward_distance,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=False
        )
        for i, m in enumerate(manoeuvres):
            nonp_turns.append(m['nonp_all'])
            # durations_turns.append(m['duration_all'] / fps)
            durations_turns.append(m['reversal_duration'] / fps)
            distances_turns.append(m['distance_all'])
            speeds_turns.append(m['speed_all_signed'])

        runs = get_forward_stats(
            X_full,
            X_slice,
            min_forward_frames=args.min_forward_frames,
            min_speed=args.min_forward_speed
        )
        for i, r in enumerate(runs):
            nonp_runs.append(r['nonp'])
            durations_runs.append(r['duration'] / fps)
            distances_runs.append(r['distance'])
            speeds_runs.append(r['speed'])

    speeds_turns = np.array(speeds_turns) * fps
    speeds_runs = np.array(speeds_runs) * fps

    # Plot correlations
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(f'Speed vs non-planarity.')
    scatter_args = dict(s=10, alpha=0.4, linewidths=0)

    ax_turns = axes[0, 0]
    ax_turns.set_title(f'Turns. Window={args.manoeuvre_window * 2 / fps:.1f}s.')
    ax_turns.set_xlabel('Non-planarity')
    # ax_turns.set_ylabel('Distance (mm)')
    # s = ax_turns.scatter(nonp_turns, distances_turns, c=speeds_turns, **scatter_args)
    ax_turns.set_ylabel('Reversal duration (s)')
    s = ax_turns.scatter(nonp_turns, durations_turns, c=speeds_turns, **scatter_args)
    cb = fig.colorbar(s, ax=ax_turns)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    ax_runs = axes[0, 1]
    ax_runs.set_title(f'Runs. Min duration={args.min_forward_frames / fps:.1f}s.')
    ax_runs.set_xlabel('Non-planarity')
    ax_runs.set_ylabel('Distance (mm)')
    s = ax_runs.scatter(nonp_runs, distances_runs, c=speeds_runs, **scatter_args)
    cb = fig.colorbar(s, ax=ax_runs)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    ax_turns2 = axes[1, 0]
    ax_turns2.set_xlabel('Non-planarity')
    ax_turns2.set_ylabel('Speed (mm/s)')
    s = ax_turns2.scatter(nonp_turns, speeds_turns, c=distances_turns, **scatter_args)
    cb = fig.colorbar(s, ax=ax_turns2)
    cb.solids.set(alpha=1)
    cb.set_label('Distance (mm)', rotation=270, labelpad=15)

    ax_runs2 = axes[1, 1]
    ax_runs2.set_xlabel('Non-planarity')
    ax_runs2.set_ylabel('Speed (mm/s)')
    s = ax_runs2.scatter(nonp_runs, speeds_runs, c=distances_runs, **scatter_args)
    cb = fig.colorbar(s, ax=ax_runs2)
    cb.solids.set(alpha=1)
    cb.set_label('Distance (mm)', rotation=270, labelpad=15)

    fig.tight_layout()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_speeds_vs_nonp' \
               f'_ds={args.dataset}' \
               f'_sw={args.smoothing_window}' \
               f'_u={args.trajectory_point}' \
               f'_ff={args.min_forward_frames}' \
               f'_fs={args.min_forward_speed}' \
               f'_rf={args.min_reversal_frames}' \
               f'_mw={args.manoeuvre_window}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_dataset_turn_angles_duration():
    """
    Plot the distributions of turn angles against durations in a dataset.
    """
    args = get_args(validate_source=False)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    angles_traj = []
    angles_plan = []
    durations = []
    distances = []
    displacements = []
    speeds = []

    # Loop over reconstructions
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps
        X_full, X_slice = get_trajectory(args)

        # Loop over manoeuvres to get the turn statistics
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            min_reversal_distance=args.min_reversal_distance,
            min_forward_frames=0,  # args.min_forward_frames,
            min_forward_distance=0,  # args.min_forward_distance,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=True,
            align_components_with_traj=True
        )
        for i, m in enumerate(manoeuvres):
            angles_traj.append(m['angle_prev_next_t'])
            angles_plan.append(m['angle_prev_next_n'])
            # durations.append(m['duration_all'] / fps)
            durations.append(m['reversal_duration'] / fps)
            distances.append(m['distance_all'])
            displacements.append(m['displacement'])
            speeds.append(m['speed_all_signed'])

    speeds = np.array(speeds) * fps

    # Plot correlations
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'Turn angles vs duration. Window={args.manoeuvre_window * 2 / fps:.1f}s.')
    scatter_args = dict(s=10, alpha=0.4, linewidths=0)

    ax = axes[0, 0]
    ax.set_xlabel('Trajectory Angle')
    ax.set_ylabel('Reversal duration (s)')
    s = ax.scatter(angles_traj, durations, c=speeds, **scatter_args)
    cb = fig.colorbar(s, ax=ax)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    ax = axes[0, 1]
    ax.set_xlabel('Trajectory Angle')
    ax.set_ylabel('Distance (mm)')
    s = ax.scatter(angles_traj, distances, c=speeds, **scatter_args)
    cb = fig.colorbar(s, ax=ax)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    ax = axes[0, 2]
    ax.set_xlabel('Trajectory Angle')
    ax.set_ylabel('Displacement (mm)')
    s = ax.scatter(angles_traj, displacements, c=speeds, **scatter_args)
    cb = fig.colorbar(s, ax=ax)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    ax = axes[1, 0]
    ax.set_xlabel('sin(Planar Angle)')
    ax.set_ylabel('Reversal duration (s)')
    s = ax.scatter(np.sin(angles_plan), durations, c=speeds, **scatter_args)
    cb = fig.colorbar(s, ax=ax)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    ax = axes[1, 1]
    ax.set_xlabel('sin(Planar Angle)')
    ax.set_ylabel('Distance (mm)')
    s = ax.scatter(np.sin(angles_plan), distances, c=speeds, **scatter_args)
    cb = fig.colorbar(s, ax=ax)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    ax = axes[1, 2]
    ax.set_xlabel('Trajectory Angle')
    ax.set_ylabel('Distance (mm)')
    s = ax.scatter(np.sin(angles_plan), displacements, c=speeds, **scatter_args)
    cb = fig.colorbar(s, ax=ax)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    fig.tight_layout()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_angles_vs_duration' \
               f'_ds={args.dataset}' \
               f'_sw={args.smoothing_window}' \
               f'_u={args.trajectory_point}' \
               f'_ff={args.min_forward_frames}' \
               f'_fs={args.min_forward_speed}' \
               f'_rf={args.min_reversal_frames}' \
               f'_mw={args.manoeuvre_window}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def _manoeuvre_stats_identifiers(args: Namespace) -> str:
    postfix = ''
    if getattr(args, 'use_postures_in_pca', False):
        postfix += '_ppca=1'
    if args.manoeuvre_window < 0:
        postfix += (f'_err={args.approx_error_limit}'
                    f'_swk={args.smoothing_window_K}'
                    f'_pwv={args.planarity_window_vertices}'
                    f'_ad={args.approx_distance}'
                    f'_ah={args.approx_curvature_height}'
                    f'_aa={args.approx_max_attempts}')
    return f'manoeuvres_ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_sw={args.smoothing_window}' \
           f'_rf={args.min_reversal_frames}' \
           f'_rd={args.min_reversal_distance}' \
           f'_ff={args.min_forward_frames}' \
           f'_fd={args.min_forward_distance}' \
           f'_mw={args.manoeuvre_window}{postfix}'


def _generate_or_load_manoeuvre_stats(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _manoeuvre_stats_identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    keys = ['angles', 'durations', 'distances', 'displacements']
    res = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            res = {}
            for k in keys:
                res[k] = data[k]
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            res = None
            logger.warning(f'Could not load cache: {e}')

    if res is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        res = {k: [] for k in keys}

        # Loop over reconstructions
        for r_ref in ds.reconstructions:
            reconstruction = Reconstruction.objects.get(id=r_ref.id)
            args.reconstruction = reconstruction.id
            fps = reconstruction.trial.fps
            X_full, X_slice = get_trajectory(args)

            tumble_idxs = None
            if args.manoeuvre_window < 0:
                from wormlab3d.particles.tumble_run import find_approximation
                from wormlab3d.particles.util import calculate_trajectory_frame
                from wormlab3d.trajectories.pca import generate_or_load_pca_cache

                pcas, _ = generate_or_load_pca_cache(
                    trial_id=reconstruction.trial.id,
                    smoothing_window=args.smoothing_window,
                    window_size=args.planarity_window,
                    tracking_only=True,
                )
                e0, _, _ = calculate_trajectory_frame(X_full, pcas, args.planarity_window)
                if args.approx_error_limit is None:
                    raise ValueError('manoeuvre_window < 0 requires --approx-error-limit (for tumble detection).')
                approx, _, _, _, _ = find_approximation(
                    X=X_full,
                    e0=e0,
                    error_limit=args.approx_error_limit,
                    planarity_window_vertices=args.planarity_window_vertices,
                    distance_first=args.approx_distance,
                    height_first=args.approx_curvature_height,
                    max_iterations=args.approx_max_attempts,
                    quiet=True,
                )
                _, _, tumble_idxs, _, _, _, _, _, _, _, _ = approx
                tumble_idxs = np.asarray(tumble_idxs)

            # Loop over manoeuvres to get the turn statistics
            manoeuvres = get_manoeuvres(
                X_full,
                X_slice,
                min_reversal_frames=args.min_reversal_frames,
                min_reversal_distance=args.min_reversal_distance,
                min_forward_frames=args.min_forward_frames,
                min_forward_distance=args.min_forward_distance,
                window_size=args.manoeuvre_window,
                cut_windows_at_manoeuvres=True,
                align_components_with_traj=True,
                use_postures_in_pca=getattr(args, 'use_postures_in_pca', False),
                tumble_idxs=tumble_idxs,
            )
            for m in manoeuvres:
                res['angles'].append(m['angle_prev_next_n'])
                res['displacements'].append(m['displacement'])
                res['durations'].append(m['reversal_duration'] / fps)
                res['distances'].append(m['reversal_distance'])

        for k in keys:
            res[k] = np.array(res[k])
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **res)

    return res


def plot_dataset_ip_angles_vs_rev_duration(
        layout: str = 'paper'
):
    """
    Plot the distributions of turn angles against durations in a dataset.
    """
    args = get_args(validate_source=False)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    res = _generate_or_load_manoeuvre_stats(args, rebuild_cache=False, cache_only=False)

    durations = res['durations']
    angles = np.arcsin(np.sin(res['angles']))
    # angles = np.sin(res['angles'])

    # Divide up into bins
    d_interval = 2
    d_max = 12
    n_boxes = int(d_max / d_interval)
    data = []
    labels = []
    rhs = np.linspace(0, d_max, n_boxes + 1)
    for i, edge in enumerate(rhs):
        if i == 0:
            continue
        d_min = rhs[i - 1]
        if i == n_boxes:
            d_max = durations.max()
        else:
            d_max = rhs[i]
        angles_i = angles[(durations > d_min) & (durations <= d_max)]
        data.append(angles_i)
        labels.append(f'{d_min:.0f}-{d_max:.0f}\n({len(angles_i)})')

    # Set up plot
    if layout == 'paper':
        plt.rc('axes', titlesize=7)  # fontsize of the title
        plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('ytick.major', pad=2, size=2)
        plt.rc('legend', fontsize=5)  # fontsize of the legend
        fig, ax = plt.subplots(1, figsize=(1.536, 1.536), gridspec_kw={
            'top': 0.99,
            'bottom': 0.24,
            'left': 0.12,
            'right': 0.99,
        })
        # ax.set_title('Reversal duration vs IP angle', pad=3)
        xlabel_pad = 4
        ylabel_pad = -7
    else:
        plt.rc('axes', labelsize=8)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6.5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=6.5)  # fontsize of the y tick labels
        fig, ax = plt.subplots(1, figsize=(3.1, 2.3), gridspec_kw={
            'top': 0.99,
            'bottom': 0.22,
            'left': 0.17,
            'right': 0.99,
        })
        xlabel_pad = 6
        ylabel_pad = 7

    ax.set_xlabel('Reversal duration (s)', labelpad=xlabel_pad)
    ax.set_ylabel('IP angle', labelpad=ylabel_pad)
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, np.pi / 2])
    ax.set_yticklabels([0, '$\pi/2$'])

    ax.boxplot(data, labels=labels, widths=0.6)

    # # Add labels on the RHS
    # ax2 = ax.twinx()
    # ax2.set_ylabel('IP angle', labelpad=4, rotation=270)
    # ax2.set_ylim(ax.get_ylim())
    # ax2.set_yticks([0, 0.38, 0.7, 1])
    # ax2.set_yticklabels(['$0, \pi \\rbrace$', '$\lbrace \\frac{\pi}{8},\\frac{7\pi}{8} \\rbrace$', '$\lbrace \\frac{\pi}{4},\\frac{3\pi}{4} \\rbrace$', '$\\frac{\pi}{2}$'])

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_ip_angles_vs_rev_duration' \
               f'_ds={args.dataset}' \
               f'_sw={args.smoothing_window}' \
               f'_u={args.trajectory_point}' \
               f'_rf={args.min_reversal_frames}' \
               f'_rd={args.min_reversal_distance}' \
               f'_ff={args.min_forward_frames}' \
               f'_fd={args.min_forward_distance}' \
               f'_mw={args.manoeuvre_window}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_dataset_distances_vs_nonp():
    """
    Plot the distributions of distances against nonp for turns and runs in a dataset.
    As above but tidied up for publication.
    """
    args = get_args(validate_source=False)

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    nonp_turns = []
    durations_turns = []
    distances_turns = []
    speeds_turns = []
    nonp_runs = []
    durations_runs = []
    distances_runs = []
    speeds_runs = []

    # Loop over reconstructions
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps
        X_full, X_slice = get_trajectory(args)

        # Loop over manoeuvres to get the turn statistics
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            window_size=args.manoeuvre_window,
            cut_windows_at_manoeuvres=True
        )
        for i, m in enumerate(manoeuvres):
            nonp_turns.append(m['nonp_all'])
            durations_turns.append(m['duration_all'] / fps)
            distances_turns.append(m['distance_all'])
            speeds_turns.append(m['speed_all_abs'])

        runs = get_forward_stats(
            X_full,
            X_slice,
            min_forward_frames=args.min_forward_frames,
            min_speed=args.min_forward_speed
        )
        for i, r in enumerate(runs):
            nonp_runs.append(r['nonp'])
            durations_runs.append(r['duration'] / fps)
            distances_runs.append(r['distance'])
            speeds_runs.append(r['speed'])

    speeds_turns = np.array(speeds_turns) * fps
    speeds_runs = np.array(speeds_runs) * fps

    # Plot correlations
    # plt.rc('axes', titlesize=7)  # fontsize of the title
    # plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    # plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    # plt.rc('legend', fontsize=6)  # fontsize of the legend
    #
    # fig, axes = plt.subplots(2, figsize=(4.53, 4.62), gridspec_kw={
    #     'hspace': 0.33,
    #     'top': 0.94,
    #     'bottom': 0.08,
    #     'left': 0.09,
    #     'right': 0.88,
    # })

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(f'Speed vs non-planarity.')
    scatter_args = dict(s=10, alpha=0.4, linewidths=0)

    ax_turns = axes[0, 0]
    ax_turns.set_title(f'Turns\nWindow={args.manoeuvre_window * 2 / fps:.1f}s')
    ax_turns.set_xlabel('Non-planarity')
    ax_turns.set_ylabel('Distance (mm)')
    s = ax_turns.scatter(nonp_turns, distances_turns, c=speeds_turns, **scatter_args)
    cb = fig.colorbar(s, ax=ax_turns)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    ax_runs = axes[0, 1]
    ax_runs.set_title(f'Runs\nMin. duration={args.min_forward_frames / fps:.1f}s')
    ax_runs.set_xlabel('Non-planarity')
    ax_runs.set_ylabel('Distance (mm)')
    s = ax_runs.scatter(nonp_runs, distances_runs, c=speeds_runs, **scatter_args)
    cb = fig.colorbar(s, ax=ax_runs)
    cb.solids.set(alpha=1)
    cb.set_label('Speed (mm/s)', rotation=270, labelpad=15)

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_distances_vs_nonp' \
               f'_ds={args.dataset}' \
               f'_sw={args.smoothing_window}' \
               f'_u={args.trajectory_point}' \
               f'_ff={args.min_forward_frames}' \
               f'_fs={args.min_forward_speed}' \
               f'_rf={args.min_reversal_frames}' \
               f'_mw={args.manoeuvre_window}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def _run_stats_identifiers(args: Namespace) -> str:
    return f'ds={args.dataset}' \
           f'_u={args.trajectory_point}' \
           f'_sw={args.smoothing_window}' \
           f'_ff={args.min_forward_frames}' \
           f'_fs={args.min_forward_speed}'


def _generate_or_load_run_stats(
        args: Namespace,
        rebuild_cache: bool = False,
        cache_only: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate or load the data.
    """
    logger.info('Fetching dataset.')
    ds = Dataset.objects.get(id=args.dataset)
    cache_path = DATA_CACHE_PATH / _run_stats_identifiers(args)
    cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    keys = ['nonps', 'durations', 'distances', 'speeds']
    res = None
    if not rebuild_cache and cache_fn.exists():
        try:
            data = np.load(cache_fn)
            res = {}
            for k in keys:
                res[k] = data[k]
            logger.info(f'Loaded data from cache: {cache_fn}')
        except Exception as e:
            res = None
            logger.warning(f'Could not load cache: {e}')

    if res is None:
        if cache_only:
            raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Generating data.')
        res = {k: [] for k in keys}

        # Loop over reconstructions
        for r_ref in ds.reconstructions:
            reconstruction = Reconstruction.objects.get(id=r_ref.id)
            args.reconstruction = reconstruction.id
            fps = reconstruction.trial.fps
            X_full, X_slice = get_trajectory(args)
            runs = get_forward_stats(
                X_full,
                X_slice,
                min_forward_frames=args.min_forward_frames,
                min_speed=args.min_forward_speed
            )
            for r in runs:
                res['nonps'].append(r['nonp'])
                res['durations'].append(r['duration'] / fps)
                res['distances'].append(r['distance'])
                res['speeds'].append(r['speed'] * fps)

        for k in keys:
            res[k] = np.array(res[k])
        logger.info(f'Saving data to {cache_fn}.')
        np.savez(cache_path, **res)

    return res


def plot_dataset_run_lengths(
        distances_or_durations: str = 'distances',
        log_x: bool = False,
        log_y: bool = False,
        layout: str = 'paper'
):
    """
    Plot the distributions of run distances in a dataset.
    """
    args = get_args(validate_source=False)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    res = _generate_or_load_run_stats(args, rebuild_cache=False, cache_only=False)

    data = res[distances_or_durations]
    if distances_or_durations == 'distances':
        levy_params = (1.1021059443934766, 0.9999906365019289, 4.058541870865994, 0.4265810216580477)
    else:
        levy_params = (1.12087316881554, 0.9999999597206704, 24.15609209509127, 2.455502352507133)

    # Fit distribution
    logger.info('Fitting distribution.')
    x = np.linspace(min(data), max(data), 200)
    # levy_params = levy_stable.fit(data)
    levy_dist = levy_stable(*levy_params)
    ks_res = ks_1samp(data, levy_dist.cdf)
    print(ks_res)
    # cauchy_dist = cauchy(*cauchy.fit(distances))

    y = levy_dist.pdf(x)
    if np.argmin(y) == 82:
        y[82] = (y[81] + y[83]) / 2

    # Set up plot
    if layout == 'paper':
        plt.rc('axes', titlesize=7, titlepad=1)  # fontsize of the title
        plt.rc('axes', labelsize=6, labelpad=1)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('ytick.major', pad=2, size=2)
        plt.rc('ytick.minor', size=1)
        plt.rc('xtick.major', pad=2, size=2)
        plt.rc('legend', fontsize=5)  # fontsize of the legend
        gs = GridSpec(
            nrows=1,
            ncols=1,
            top=0.92,
            bottom=0.16,
            left=0.18,
            right=0.98,
        )
        fig = plt.figure(figsize=(1.536, 1.536))
        # ylabel_pad = -1
    else:
        plt.rc('axes', titlesize=9)  # fontsize of the title
        plt.rc('axes', labelsize=8)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=6.5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=6.5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=7)  # fontsize of the legend
        gs = GridSpec(
            nrows=1,
            ncols=1,
            top=0.91,
            bottom=0.16,
            left=0.14,
            right=0.99,
        )
        fig = plt.figure(figsize=(2.9, 2.3))
        # ylabel_pad = 2

    # Plot correlations
    logger.info('Plotting')
    ax = fig.add_subplot(gs[0, 0])

    if log_x:
        bins = np.logspace(np.log2(data.min()), np.log2(data.max()), 20, base=2)
    else:
        bins = 20

    ax.hist(data, bins=bins, density=True, rwidth=0.9)
    # ax.plot(x, cauchy_dist.pdf(x),
    #         label=f'Cauchy fit\n($x_0=${cauchy_dist.args[0]:.1f}, $\gamma=${cauchy_dist.args[1]:.1f})')
    ax.plot(x, y, label=f'Levy fit\n($\\alpha=${levy_dist.args[0]:.1f}, $\\beta=${levy_dist.args[1]:.1f})')
    ax.set_title(f'Run {distances_or_durations} ($\geq$ {args.min_forward_frames / DEFAULT_FPS:.1f}s)')
    if log_x:
        ax.set_xscale('log')
        if distances_or_durations == 'durations':
            subs = [1.6, 3.2, 6.4, 8.]  # ticks to show per decade
            ax.xaxis.set_minor_locator(LogLocator(subs=subs))
            # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_minor_formatter(FormatStrFormatter('%d'))
    if log_y:
        ax.set_yscale('log')
    if distances_or_durations == 'distances':
        ax.set_xlabel('Distance (mm)')
    else:
        ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Density')  # , labelpad=ylabel_pad)
    ax.legend()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_run_{distances_or_durations}' \
               f'_ds={args.dataset}' \
               f'_sw={args.smoothing_window}' \
               f'_ff={args.min_forward_frames}' \
               f'_fs={args.min_forward_speed}' \
               f'_N={len(data)}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def plot_single_manoeuvre(
        index: int = None,
        frame_num: int = None,
        plot_mlab_postures: bool = False
):
    """
    Plot a single manoeuvre detected in the trajectory.
    """
    args = get_args(validate_source=False)
    X_full, X_slice = get_trajectory(args)
    trial = Trial.objects.get(id=args.trial)
    signed_speeds = calculate_speeds(X_full, signed=True) * trial.fps
    worm_idxs = [500, 900]

    manoeuvres = get_manoeuvres(
        X_full,
        X_slice,
        min_reversal_frames=args.min_reversal_frames,
        window_size=args.manoeuvre_window
    )

    if index is not None:
        m = manoeuvres[index]
    elif frame_num is not None:
        frame_nums = np.array([m['centre_idx'] for m in manoeuvres])
        idx = np.argmin(np.abs(frame_nums - frame_num))
        m = manoeuvres[idx]
        index = f'f{frame_num}'
    else:
        raise RuntimeError('index or frame_num must be specified!')

    filename = f'manoeuvre_{index}_trial={args.trial}_{args.midline3d_source}' \
               f'_rev={args.min_reversal_frames}' \
               f'_ws={args.manoeuvre_window}' \
               f'_sw={args.smoothing_window}' \
               f'_frames={m["start_idx"]}-{m["end_idx"]}'

    plane_prev = make_box_from_pca(m['X_prev'], m['pca_prev'], 'orange', scale=(1, 2, 3))
    plane_next = make_box_from_pca(m['X_next'], m['pca_next'], 'green', scale=(1, 1, 1))

    plot_manoeuvre_3d(
        X_slice=X_slice[m['start_idx']:m['end_idx']],
        X_full=X_full[m['start_idx']:m['end_idx']],
        filename=filename,
        title=f'Trial={args.trial}. Frames={m["start_idx"]}-{m["end_idx"]}',
        colours=signed_speeds[m['start_idx']:m['end_idx']],
        cmap='PRGn',
        show_colourbar=True,
        worm_idxs=worm_idxs,  # 8700 - m['start_idx'],  # 0,  # [500, 900],
        planes=[plane_prev, plane_next],
        # azim=-58,  # 55,
        # elev=30,  # -5
        azim=55,
        elev=-5
    )

    if plot_mlab_postures and len(worm_idxs) > 0:
        interactive = False
        transparent_bg = True

        plot_options = {
            0: {
                # (azimuth, elevation, distance, focalpoint)
                'azimuth': 54,
                'elevation': 73,
                'distance': 1,
                'roll': -126,
            },
            1: {
                # (azimuth, elevation, distance, focalpoint)
                'azimuth': 113,
                'elevation': 134,
                'distance': 0.9,
                'roll': -0.6,
            }
        }

        for i, worm_idx in enumerate(worm_idxs):
            if i == 1:
                continue
            X = X_full[m['start_idx'] + worm_idx]
            NF = NaturalFrame(X)

            # 3D plot of eigenworm
            fig = plot_natural_frame_3d_mlab(
                NF,
                **plot_options[i],
                midline_opts={'line_width': 20},
                show_frame_arrows=False,
                show_outline=False,
                show_axis=False,
                show_pca_arrows=False,
                offscreen=not interactive,
            )

            if save_plots:
                path = LOGS_PATH / f'{START_TIMESTAMP}_{filename}_wi={worm_idx}.png'
                logger.info(f'Saving plot to {path}.')

                if not transparent_bg:
                    mlab.savefig(str(path), figure=fig)
                else:
                    fig.scene._lift()
                    img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
                    img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
                    img.save(path)
                    if not show_plots:
                        mlab.clf(fig)
                        mlab.close()

            if show_plots:
                if interactive:
                    mlab.show()
                else:
                    fig.scene._lift()
                    img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
                    mlab.clf(fig)
                    mlab.close()
                    fig_mpl = plt.figure(figsize=(10, 10))
                    ax = fig_mpl.add_subplot()
                    ax.imshow(img)
                    ax.axis('off')
                    fig_mpl.tight_layout()
                    plt.show()
                    plt.close(fig_mpl)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()

    # plot_all_manoeuvres()
    # plot_angles_and_durations()
    # plot_angles_and_durations_varying_parameters()
    # plot_manoeuvre_rate()
    # plot_dataset_distributions()
    # plot_dataset_reversal_durations_vs_angles()
    # plot_dataset_reversal_distance_vs_prev_next_angles()
    # plot_dataset_reversal_durations_vs_rev_angles()
    # plot_dataset_reversal_durations_vs_angles_combined()
    plot_dataset_reversal_durations_vs_angles_combined_basic()
    # plot_angle_correlations()
    # plot_dataset_speeds_vs_nonp()
    # plot_dataset_turn_angles_duration()
    # plot_dataset_ip_angles_vs_rev_duration(layout='thesis')
    # plot_dataset_run_lengths(distances_or_durations='distances', log_y=True, layout='thesis')
    # plot_dataset_run_lengths(distances_or_durations='durations', log_x=True, log_y=True, layout='thesis')

    # plot_single_manoeuvre(index=2, plot_mlab_postures=True)
    # plot_single_manoeuvre(frame_num=11400)
    # plot_single_manoeuvre(frame_num=15100)
    # plot_single_manoeuvre(frame_num=8700)
