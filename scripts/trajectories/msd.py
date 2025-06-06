import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset, Trial
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.brownian_particle import BrownianParticle, ActiveParticle, ConfinedParticle, BoundedParticle
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.displacement import calculate_msd, plot_msd, plot_msd_multiple, \
    calculate_displacements_parallel, DISPLACEMENT_AGGREGATION_SQUARED_SUM
from wormlab3d.trajectories.util import get_deltas_from_args

# tex_mode()

show_plots = True
save_plots = False
img_extension = 'svg'


def make_filename(method: str, args: Namespace, excludes: List[str] = None, append: str = None) -> str:
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'_{method}'

    for k in ['dataset', 'trial', 'frames', 'src', 'directionality', 'deltas', 'delta_step', 'u']:
        if k in excludes:
            continue
        if k == 'dataset' and args.dataset is not None:
            fn += f'_dataset={args.dataset}'
        elif k == 'trial' and args.trial is not None:
            fn += f'_trial={args.trial}'
        elif k == 'frames':
            frames_str_fn = ''
            if args.start_frame is not None or args.end_frame is not None:
                start_frame = args.start_frame if args.start_frame is not None else 0
                end_frame = args.end_frame if args.end_frame is not None else -1
                frames_str_fn = f'_f={start_frame}-{end_frame}'
            fn += frames_str_fn
        elif k == 'src':
            fn += f'_{args.midline3d_source}'
        elif k == 'directionality' and args.directionality is not None:
            fn += f'_dir={args.directionality}'
        elif k == 'deltas':
            fn += f'_d={args.min_delta}-{args.max_delta}'
        elif k == 'delta_step':
            fn += f'_ds={args.delta_step}'
        elif k == 'u':
            fn += f'_u={args.trajectory_point}'
        elif k == 'projection':
            fn += f'_p={args.projection}'

    if append is not None:
        fn += append

    return LOGS_PATH / (fn + '.' + img_extension)


def make_title(args: Namespace) -> str:
    frames_str_title = ''
    if args.start_frame is not None or args.end_frame is not None:
        start_frame = args.start_frame if args.start_frame is not None else 0
        end_frame = args.end_frame if args.end_frame is not None else -1
        frames_str_title = f' (frames={start_frame}-{end_frame})'

    title = f'MSD. Trial={args.trial}{frames_str_title}. Src={args.midline3d_source}. ' + \
            (f'Smoothing window={args.smoothing_window}. ' if args.smoothing_window is not None else '') + \
            (f'Pruned slowest={args.prune_slowest_ratio * 100:.1f}%. '
             if args.prune_slowest_ratio is not None else '') + \
            (f'Directionality={args.directionality}. ' if args.directionality is not None else '')

    return title


def msd():
    args = get_args()
    trajectory = get_trajectory_from_args(args)
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    msds = calculate_msd(trajectory, deltas)
    plot_msd(msds, title=make_title(args))

    if save_plots:
        plt.savefig(make_filename('msd', args))
    if show_plots:
        plt.show()


def msd_multiple():
    args = get_args()

    us = [0.1, 0.5, 0.9]
    # projections = ['3D', 'x', 'y', 'z']
    projections = ['x', 'y', 'z']
    msds = {}
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    for u in us:
        args.trajectory_point = u
        msds[u] = {}
        for p in projections:
            logger.info(f'Calculating MSD for u={u} and projection={p}.')
            args.projection = p
            trajectory = get_trajectory_from_args(args)
            msds_u = calculate_msd(trajectory, deltas)
            msds[u][p] = msds_u

    plot_msd_multiple(msds, deltas, title=make_title(args))

    if save_plots:
        plt.savefig(
            make_filename(
                'msd_multiple',
                args,
                excludes=['u'],
                append=f'_u={",".join([str(u) for u in us])}'
            )
        )
    if show_plots:
        plt.show()


def msd_wt3d_vs_reconst():
    args = get_args()

    us = [0.1, 0.5, 0.9]
    srcs = ['reconst', 'reconst', 'WT3D']
    src_files = ['039_shelf-2.hdf5', '039_shelf-2.hdf5', None]
    frames = [(0, 20000), (449, 4999), (449, 4999)]
    projections = ['x', 'y', 'z']

    msds = {}
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    for u in us:
        args.trajectory_point = u
        msds[u] = {}
        for k, src in enumerate(srcs):
            args.midline3d_source = src
            args.midline3d_source_file = src_files[k]
            args.start_frame = frames[k][0]
            args.end_frame = frames[k][1]
            msds[u][k] = {}
            for p in projections:
                logger.info(f'Calculating MSD for src={src}, frames={frames[k]} u={u} and projection={p}.')
                args.projection = p
                trajectory = get_trajectory_from_args(args)
                msds_usp = calculate_msd(trajectory, deltas)
                msds[u][k][p] = msds_usp

    deltas = deltas / 25

    linestyles = {
        'x': 'dotted',
        'y': 'dashed',
        'z': 'dashdot'
    }
    colors = {
        0: 'red',
        1: 'green',
        2: 'blue',
    }

    for u in us:
        fig, ax = plt.subplots(1, figsize=(12, 8), sharex=True, sharey=True)

        msds_u = msds[u]
        for k, src in enumerate(srcs):
            msds_us = msds_u[k]
            for p in projections:
                msds_usp = msds_us[p]
                msd_vals = np.array(list(msds_usp.values()))
                ax.plot(deltas, msd_vals, label=f'{src}, f={frames[k]}, p={p}', linestyle=linestyles[p],
                        color=colors[k], alpha=0.5)

        ax.set_title(make_title(args) + f'u={u}.')
        ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
        ax.set_xlabel('$\Delta s$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.legend()
        fig.tight_layout()

        if save_plots:
            plt.savefig(
                make_filename(
                    'msd_WT3D_vs_reconst',
                    args,
                    excludes=['frames', 'src', 'u'],
                    append=f'_u={u}'
                )
            )
        if show_plots:
            plt.show()


def msd_brownian():
    args = get_args()
    n_runs = 1

    # Use same brownian parameters for all simulations
    fps = 25
    total_time = 13 * 60
    n_steps = total_time * fps
    D = 1

    # Define deltas
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    deltas_s = deltas / fps

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title(f'MSD of Brownian Particle. D={D}, n\_steps={n_steps}, total\_time={total_time}s.')
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()

    for i in range(n_runs):
        # Generate the brownian trajectory
        p = BrownianParticle(D=D)
        trajectory = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
        for j, projection in enumerate(['x', 'y', 'z']):
            X = trajectory[:, j]
            msds = calculate_msd(X, deltas)
            msd_vals = np.array(list(msds.values()))
            ax.plot(deltas_s, msd_vals, linestyle=[':', '-.', '--'][j])

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'brownian_particle',
                args,
                excludes=['trial', 'frames', 'src', 'u'],
                append=f'_D={D}_n={n_steps}_T={total_time}'
            )
        )
    if show_plots:
        plt.show()


def msd_brownian_projections():
    args = get_args()

    # Brownian parameters
    fps = 25
    total_time = 13 * 60
    n_steps = total_time * fps
    D = 1

    # Define deltas
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    deltas_s = deltas / fps

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title(f'MSD of Brownian Particle. D={D}, n\_steps={n_steps}, total\_time={total_time}s.')
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()

    # Generate the brownian trajectory
    p = BrownianParticle(D=D)
    trajectory = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
    for j, projection in enumerate(['x', 'y', 'z']):
        X = trajectory[:, j]
        msds = calculate_msd(X, deltas)
        msd_vals = np.array(list(msds.values()))
        ax.plot(deltas_s, msd_vals, linestyle=[':', '-.', '--'][j], label=projection)
    ax.legend()

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'brownian_particle_projections',
                args,
                excludes=['trial', 'frames', 'src', 'u'],
                append=f'_D={D}_n={n_steps}_T={total_time}'
            )
        )
    if show_plots:
        plt.show()


def msd_brownian_varying_Ds():
    args = get_args()

    # Use same time window for all simulations
    fps = 25
    total_time = 10 * 60
    n_steps = total_time * fps

    # Define deltas
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    deltas_s = deltas / fps

    # Vary the diffusion coefficient
    Ds = [0.1, 1, 10, 100]

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title(f'MSD of Brownian Particle. n\_steps={n_steps}, total\_time={total_time}s.')
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()

    for D in Ds:
        # Generate the brownian trajectory
        p = BrownianParticle(D=D)
        trajectory = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
        msds = calculate_msd(trajectory, deltas)
        msd_vals = np.array(list(msds.values()))
        ax.plot(deltas_s, msd_vals, label=f'D={D}')

    ax.legend()
    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'brownian_particle',
                args,
                excludes=['trial', 'frames', 'src', 'u'],
                append=f'_D={",".join([str(D) for D in Ds])}_n={n_steps}_T={total_time}'
            )
        )
    if show_plots:
        plt.show()


def msd_active_particle():
    args = get_args()
    n_runs = 1

    # Use same parameters for all simulations
    fps = 25
    total_time = 13 * 60
    n_steps = total_time * fps
    D = 1
    momentum = 1

    # Define deltas
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    deltas_s = deltas / fps

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title(f'MSD of Active Particle. D={D}, momentum={momentum}, n\_steps={n_steps}, total\_time={total_time}s.')
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()

    for i in range(n_runs):
        # Generate the brownian trajectory
        p = ActiveParticle(D=D, momentum=momentum)
        trajectory = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
        for j, projection in enumerate(['x', 'y', 'z']):
            X = trajectory[:, j]
            msds = calculate_msd(X, deltas)
            msd_vals = np.array(list(msds.values()))
            ax.plot(deltas_s, msd_vals, linestyle=[':', '-.', '--'][j])
        msds = calculate_msd(trajectory, deltas)
        msd_vals = np.array(list(msds.values()))
        ax.plot(deltas_s, msd_vals)  # , linestyle=[':', '-.', '--'][j])

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'active_particle',
                args,
                excludes=['trial', 'frames', 'src', 'u'],
                append=f'_D={D}_n={n_steps}_T={total_time}'
            )
        )
    if show_plots:
        plt.show()


def msd_active_particles():
    args = get_args()
    n_runs_per_momentum = 5

    # Use same parameters for all simulations
    fps = 25
    total_time = 13 * 60
    n_steps = total_time * fps
    D = 1
    momenta = [0, 0.25, 0.5, 0.75, 1]
    cmaps = [plt.get_cmap(k) for k in ['Blues_r', 'Oranges_r', 'Greens_r', 'Reds_r', 'Purples_r']]

    # Define deltas
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    deltas_s = deltas / fps

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title(f'MSD of Active Particles. D={D}, n\_steps={n_steps}, total\_time={total_time}s.')
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()

    for i, momentum in enumerate(momenta):
        for j in range(n_runs_per_momentum):
            p = ActiveParticle(D=D, momentum=momentum)
            trajectory = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
            msds = calculate_msd(trajectory, deltas)
            msd_vals = np.array(list(msds.values()))
            if j == 0:
                label = f'momentum={momentum:.2f}'
            else:
                label = None
            ax.plot(deltas_s, msd_vals, label=label, alpha=0.6,
                    color=cmaps[i](j / n_runs_per_momentum))

    ax.legend()
    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'active_particles',
                args,
                excludes=['trial', 'frames', 'src', 'u'],
                append=f'_D={D}_ms={",".join([f"{m:.2f}" for m in momenta])}_n={n_steps}_T={total_time}'
            )
        )
    if show_plots:
        plt.show()


def msd_bounded_particle():
    args = get_args()
    n_runs = 10

    # Use same parameters for all simulations
    fps = 25
    total_time = 15 * 60
    n_steps = total_time * fps
    D = 1e-4
    momentum = 0.95
    box_size = 3
    bounds=np.array([[-box_size/2,box_size/2]]*3)

    # Define deltas
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    deltas_s = deltas / fps

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title(f'MSD of Bounded Particle. Box size = {box_size}\n'
                 f'D={D}, momentum={momentum:.2f}, n\_steps={n_steps}, total\_time={total_time}s.')
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()

    for i in range(n_runs):
        # Generate the brownian trajectory
        p = BoundedParticle(
            D=D,
            momentum=momentum,
            bounds=bounds
        )
        X = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
        msds = calculate_msd(X, deltas)
        msd_vals = np.array(list(msds.values()))
        ax.plot(deltas_s, msd_vals)

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'bounded_particle',
                args,
                excludes=['trial', 'frames', 'src', 'u'],
                append=f'_D={D}_m={momentum:.2f}_n={n_steps}_T={total_time}_bs={box_size}'
            )
        )
    if show_plots:
        plt.show()


def msd_confined_particle():
    args = get_args()
    n_runs = 25

    # Use same parameters for all simulations
    fps = 25
    total_time = 60 * 60
    n_steps = total_time * fps
    D = 1e-4
    momentum = 0.95
    unconfined_duration_mean = 30
    unconfined_duration_variance = 0.1
    unconfined_momentum = 0.99
    confined_duration_mean = 5
    confined_duration_variance = 0
    confined_momentum = 0.0001
    D_confined = 1e-9

    # Define deltas
    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    deltas_s = deltas / fps

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title(f'MSD of Confined Particle.\n'
                 f'D\_unconfined={D}, D\_confined={D_confined}, momentum={momentum:.2f}, n\_steps={n_steps}, total\_time={total_time}s.\n'
                 f'Unconfined duration $\sim \mathcal{{N}}({unconfined_duration_mean:.2f}, {unconfined_duration_variance:.2f})$. \n'
                 f'Confined duration $\sim \mathcal{{N}}({confined_duration_mean:.2f}, {unconfined_duration_variance:.2f})$.', )
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()

    for i in range(n_runs):
        # Generate the brownian trajectory
        p = ConfinedParticle(
            D=D,
            momentum=momentum,
            unconfined_duration_mean=unconfined_duration_mean,
            unconfined_duration_variance=unconfined_duration_variance,
            unconfined_momentum=unconfined_momentum,
            confined_duration_mean=confined_duration_mean,
            confined_duration_variance=confined_duration_variance,
            confined_momentum=confined_momentum,
            D_confined=D_confined,
        )
        trajectory = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
        msds = calculate_msd(trajectory, deltas)
        msd_vals = np.array(list(msds.values()))
        ax.plot(deltas_s, msd_vals)

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'confined_particle',
                args,
                excludes=['trial', 'frames', 'src', 'u'],
                append=f'_Du={D}_Dc={D_confined}_m={momentum:.2f}_n={n_steps}_T={total_time}'
                       f'_u={unconfined_duration_mean:.2f},{unconfined_duration_variance:.2f}'
                       f'_c={unconfined_duration_mean:.2f},{unconfined_duration_variance:.2f}'
            )
        )
    if show_plots:
        plt.show()


def msd_dataset():
    args = get_args()

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    deltas_ts = deltas / 25

    # Unset midline source args and use tracking data only (longer)
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.reconstruction = None
    args.tracking_only = True

    # Calculate the displacements for all trials
    displacements = {}
    all_displacements = {delta: [] for delta in deltas}
    for trial in ds.include_trials:
        logger.info(f'Calculating displacements for trial={trial.id}.')
        args.trial = trial.id

        # Group results by concentration
        c = trial.experiment.concentration
        if c not in displacements:
            displacements[c] = {delta: [] for delta in deltas}

        # Calculate displacements for trial
        trajectory = get_trajectory_from_args(args)
        d = calculate_displacements_parallel(trajectory, deltas, aggregation=DISPLACEMENT_AGGREGATION_SQUARED_SUM)
        for delta in deltas:
            displacements[c][delta].extend(d[delta])
            all_displacements[delta].extend(d[delta])

    # Sort by concentration
    displacements = {k: v for k, v in sorted(list(displacements.items()))}
    concs = list(displacements.keys())

    # Calculate msds
    msds = {}
    for c, displacements_c in displacements.items():
        msds[c] = {
            delta: np.mean(displacements_c[delta])
            for delta in deltas
        }
    msds_all_traj = {
        delta: np.mean(all_displacements[delta])
        for delta in deltas
    }
    msds_all_conc = {
        delta: np.mean([msds_c[delta] for c, msds_c in msds.items()])
        for delta in deltas
    }

    # Calculate gradients
    grads = {}
    for c, msds_c in msds.items():
        msd_vals = np.array(list(msds_c.values()))
        grads[c] = np.gradient(np.log(msd_vals), np.log(deltas))
    msd_vals_all_traj = np.array(list(msds_all_traj.values()))
    grads_all_traj = np.gradient(np.log(msd_vals_all_traj), np.log(deltas))
    msd_vals_all_conc = np.array(list(msds_all_conc.values()))
    grads_all_conc = np.gradient(np.log(msd_vals_all_conc), np.log(deltas))

    # Set up plots and colours
    fig, axes = plt.subplots(2, figsize=(12, 14), sharex=True)
    cmap = plt.get_cmap('jet')
    colours = cmap(np.linspace(0, 1, len(concs)))

    # Plot MSD combined results
    ax = axes[0]
    msd_vals_all_traj = np.array(list(msds_all_traj.values()))
    ax.plot(deltas_ts, msd_vals_all_traj, label='Trajectory average',
            alpha=0.8, c='gray', linestyle=':', linewidth=2, zorder=100)
    msd_vals_all_conc = np.array(list(msds_all_conc.values()))
    ax.plot(deltas_ts, msd_vals_all_conc, label='Concentration average',
            alpha=0.6, c='black', linestyle='--', linewidth=2, zorder=100)

    # Plot MSD for each concentration
    for i, (c, msds_c) in enumerate(msds.items()):
        msd_vals = np.array(list(msds_c.values()))
        ax.plot(deltas_ts, msd_vals, label=f'c={c:.2f}%', alpha=0.6, c=colours[i])

    # Complete MSD plot
    title = f'MSD. Dataset={args.dataset}. u={args.trajectory_point}.' + \
            (f'Smoothing window={args.smoothing_window}. ' if args.smoothing_window is not None else '')
    ax.set_title(title)
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    ax.legend()

    # Plot (log-log) gradients
    ax = axes[1]
    ax.plot(deltas_ts, grads_all_traj, label='Trajectory average',
            alpha=0.8, c='gray', linestyle=':', linewidth=2, zorder=100)
    ax.plot(deltas_ts, grads_all_conc, label='Concentration average',
            alpha=0.6, c='black', linestyle='--', linewidth=2, zorder=100)

    # Plot grads for each concentration
    for i, (c, grads_c) in enumerate(grads.items()):
        ax.plot(deltas_ts, grads_c, label=f'c={c:.2f}%', alpha=0.6, c=colours[i])

    # Complete grads plot
    ax.set_title('Gradients (log-log)')
    ax.set_ylabel('grad')
    ax.set_xlabel('$\Delta s$')
    ax.grid()
    ax.legend()

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                'msd_dataset',
                args,
                excludes=['trial', 'frames', 'src'],
            )
        )
    if show_plots:
        plt.show()


def msd_dataset_trials():
    """
    MSD plot for all trials in a dataset.
    """
    args = get_args()

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))
    deltas_ts = deltas / 25

    # Unset midline source args and use tracking data only (longer)
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.reconstruction = None
    args.tracking_only = True

    # Calculate the displacements for all trials
    all_displacements = {delta: [] for delta in deltas}
    conc_displacements = {}
    trial_displacements = {}
    trials = {}
    for trial in ds.include_trials:
        logger.info(f'Calculating displacements for trial={trial.id}.')
        args.trial = trial.id
        trials[trial.id] = trial

        # Group results by concentration
        c = trial.experiment.concentration
        if c not in conc_displacements:
            conc_displacements[c] = {delta: [] for delta in deltas}
        if c not in trial_displacements:
            trial_displacements[c] = {}
        trial_displacements[c][trial.id] = {}

        # Calculate displacements for trial
        trajectory = get_trajectory_from_args(args)
        d = calculate_displacements_parallel(trajectory, deltas, aggregation=DISPLACEMENT_AGGREGATION_SQUARED_SUM)
        for delta in deltas:
            trial_displacements[c][trial.id][delta] = d[delta]
            conc_displacements[c][delta].extend(d[delta])
            all_displacements[delta].extend(d[delta])

    # Sort by concentration
    conc_displacements = {k: v for k, v in sorted(list(conc_displacements.items()))}
    trial_displacements = {k: v for k, v in sorted(list(trial_displacements.items()))}
    concs = list(conc_displacements.keys())

    # Calculate msds
    msds_trials = {}
    msds_conc = {c: {} for c in concs}
    for c in concs:
        msds_conc[c] = {
            delta: np.mean(conc_displacements[c][delta])
            for delta in deltas
        }
        msds_trials[c] = {}
        for trial_id, t_displacements in trial_displacements[c].items():
            msds_trials[c][trial_id] = {
                delta: np.mean(t_displacements[delta])
                for delta in deltas
            }
    msds_all_traj = {
        delta: np.mean(all_displacements[delta])
        for delta in deltas
    }
    msds_all_conc = {
        delta: np.mean([msds_c[delta] for c, msds_c in msds_conc.items()])
        for delta in deltas
    }

    # Calculate gradients
    grads_trials = {}
    grads_concs = {}
    for c in concs:
        msd_vals_c = np.array(list(msds_conc[c].values()))
        grads_concs[c] = np.gradient(np.log(msd_vals_c), np.log(deltas))
        grads_trials[c] = {}
        for trial_id, t_displacements in msds_trials[c].items():
            msd_vals_t = np.array(list(t_displacements.values()))
            grads_trials[c][trial_id] = np.gradient(np.log(msd_vals_t), np.log(deltas))
    msd_vals_all_traj = np.array(list(msds_all_traj.values()))
    grads_all_traj = np.gradient(np.log(msd_vals_all_traj), np.log(deltas))
    msd_vals_all_conc = np.array(list(msds_all_conc.values()))
    grads_all_conc = np.gradient(np.log(msd_vals_all_conc), np.log(deltas))

    # Make a separate plot for each concentration
    for c in concs:
        msds = msds_trials[c]
        grads = grads_trials[c]

        # Set up plots and colours
        fig, axes = plt.subplots(2, figsize=(12, 14), sharex=True)
        cmap = plt.get_cmap('jet')
        colours = cmap(np.linspace(0, 1, len(msds)))

        # Plot MSD combined results
        ax = axes[0]
        msd_vals_all_traj = np.array(list(msds_all_traj.values()))
        ax.plot(deltas_ts, msd_vals_all_traj, label='Dataset average 1',
                alpha=0.7, c='brown', linestyle=':', linewidth=2, zorder=80)
        msd_vals_all_conc = np.array(list(msds_all_conc.values()))
        ax.plot(deltas_ts, msd_vals_all_conc, label='Dataset average 2',
                alpha=0.5, c='brown', linestyle='--', linewidth=2, zorder=80)
        msd_vals_conc = np.array(list(msds_conc[c].values()))
        ax.plot(deltas_ts, msd_vals_conc, label='Concentration average',
                alpha=0.6, c='black', linestyle='--', linewidth=2, zorder=100)

        # Plot MSD for each trial
        for i, (trial_id, msds_t) in enumerate(msds.items()):
            msd_vals = np.array(list(msds_t.values()))
            ax.plot(deltas_ts, msd_vals, label=f'trial={trial_id} ({trials[trial_id].duration:%M:%S})', alpha=0.6,
                    c=colours[i])

        # Complete MSD plot
        title = f'MSD. Concentration={c:.2f}\%. Dataset={args.dataset}. u={args.trajectory_point}.' + \
                (f'Smoothing window={args.smoothing_window}. ' if args.smoothing_window is not None else '')
        ax.set_title(title)
        ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
        ax.set_xlabel('$\Delta s$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.legend()

        # Plot (log-log) gradients
        ax = axes[1]
        ax.plot(deltas_ts, grads_all_traj, label='Dataset average 1',
                alpha=0.7, c='brown', linestyle=':', linewidth=2, zorder=80)
        ax.plot(deltas_ts, grads_all_conc, label='Dataset average 2',
                alpha=0.5, c='brown', linestyle='--', linewidth=2, zorder=80)
        ax.plot(deltas_ts, grads_concs[c], label='Concentration average',
                alpha=0.6, c='black', linestyle='--', linewidth=2, zorder=100)

        # Plot grads for each trial
        for i, (trial_id, grads_t) in enumerate(grads.items()):
            ax.plot(deltas_ts, grads_t, label=f'trial={trial_id} ({trials[trial_id].duration:%M:%S})', alpha=0.6,
                    c=colours[i])

        # Complete grads plot
        ax.set_title('Gradients (log-log)')
        ax.set_ylabel('grad')
        ax.set_xlabel('$\Delta s$')
        ax.grid()
        ax.legend()

        fig.tight_layout()

        if save_plots:
            plt.savefig(
                make_filename(
                    f'msd_dataset_conc={c:.2f}',
                    args,
                    excludes=['trial', 'frames', 'src'],
                )
            )
        if show_plots:
            plt.show()


def msd_multiple_trials():
    """
    MSD plot for all trials in a dataset.
    """
    args = get_args()
    deltas, delta_ts = get_deltas_from_args(args)

    # Define trial_ids
    # trial_ids = [111, 72, 106, 110, 239, 112, 80, 103, 168]
    # trial_ids = [111, 72, 106, 110, 239, 112, 80, 103, 168, 37]
    trial_ids = [162, 73, 114, 103, 76, 35, 168, 37]

    # Unset midline source args and use tracking data only (longer)
    args.midline3d_source = None
    args.midline3d_source_file = None
    args.reconstruction = None
    args.tracking_only = True

    # Calculate the displacements for all trials
    all_displacements = {delta: [] for delta in deltas}
    trial_displacements = {}
    trials = {}
    for trial_id in trial_ids:
        trial = Trial.objects.get(id=trial_id)
        logger.info(f'Calculating displacements for trial={trial.id}.')
        args.trial = trial.id
        trials[trial.id] = trial
        trial_displacements[trial.id] = {}

        # Calculate displacements for trial
        trajectory = get_trajectory_from_args(args)
        d = calculate_displacements_parallel(trajectory, deltas, aggregation=DISPLACEMENT_AGGREGATION_SQUARED_SUM)
        for delta in deltas:
            trial_displacements[trial.id][delta] = d[delta]
            all_displacements[delta].extend(d[delta])

    # Sort the trials by concentration
    trial_ids.sort(key=lambda tid: trials[tid].duration)
    trial_ids.sort(key=lambda tid: trials[tid].experiment.concentration)

    # Calculate msds
    msds = {}
    for trial_id, t_displacements in trial_displacements.items():
        msds[trial_id] = {
            delta: np.mean(t_displacements[delta])
            for delta in deltas
        }
    msds_all_traj = {
        delta: np.mean(all_displacements[delta])
        for delta in deltas
    }

    # Set up plots and colours
    fig, ax = plt.subplots(1, figsize=(8, 4))
    cmap = plt.get_cmap('brg')
    colours = cmap(np.linspace(0, 1, len(msds)))

    # Plot MSD combined results
    msd_vals_all_traj = np.array(list(msds_all_traj.values()))
    ax.plot(delta_ts, msd_vals_all_traj, label='Average',
            alpha=0.8, c='black', linestyle='--', linewidth=3, zorder=80)

    # Plot MSD for each trial
    for i, trial_id in enumerate(trial_ids):
        msd_vals = np.array(list(msds[trial_id].values()))
        ax.plot(
            delta_ts,
            msd_vals,
            label=f'Trial #{trial_id} '
                  f'(c={trials[trial_id].experiment.concentration:.2f}% '
                  f'T={trials[trial_id].duration:%M:%S})',
            alpha=0.5,
            c=colours[i]
        )

    # Complete MSD plot
    ax.set_ylabel('MSD$=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta\ (s)$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    ax.legend(bbox_to_anchor=(1.04, 1))

    # Highlight flattening-off region
    ylim = ax.get_ylim()[1] * 1.1
    ax.set_ylim(top=ylim)
    ax.fill_between(np.arange(30, 100), ylim, color='red', alpha=0.3, zorder=-1, linewidth=0)

    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename(
                f'msd_multiple_trials={",".join([str(tid) for tid in trial_ids])}',
                args,
                excludes=['trial', 'frames', 'src'],
            ),
            transparent=True
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # msd()
    # msd_multiple()
    # msd_wt3d_vs_reconst()
    # msd_brownian()
    # msd_brownian_projections()
    # msd_brownian_varying_Ds()
    # msd_active_particle()
    # msd_active_particles()
    msd_bounded_particle()
    # msd_confined_particle()
    # msd_dataset()
    # msd_dataset_trials()
    # msd_multiple_trials()
