import os
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCES
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory

show_plots = True
save_plots = True
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot helicity across a clip.')
    parser.add_argument('--trial', type=int, help='Trial by id.')
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--source', type=str, choices=M3D_SOURCES, help='Midline3D source.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--resample-points', type=int, default=-1, help='Resample the curve points.')
    parser.add_argument('--smoothing-window', type=int, default='time', help='Label x-axis with time or frame number.')
    parser.add_argument('--x-label', type=str, default='time', help='Label x-axis with time or frame number.')
    parser.add_argument('--x-label-start', type=str, default='trial',
                        help='Start x-axis labelling at "zero" or to match the "trial".')
    parser.add_argument('--K-max', type=float, help='Fix the maximum curvature for the colourmap.')

    args = parser.parse_args()

    assert args.reconstruction is not None or args.trial is not None, 'Trial or reconstruction must be specified!'

    print_args(args)

    return args


def curvature_kymogram(
        layout: str = 'debug'
):
    """
    Plot a kymogram of the curvature over time.
    """
    args = parse_args()

    common_args = {
        'reconstruction_id': args.reconstruction,
        'trial_id': args.trial,
        'midline_source': args.source,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'resample_points': args.resample_points,
        'smoothing_window': args.smoothing_window
    }
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    reconstruction = Reconstruction.objects.get(id=meta['reconstruction'])
    trial = reconstruction.trial
    K = np.abs(Z)
    logger.info(f'Maximum curvature = {K.max():.3f}')
    N = len(Z)
    if args.x_label_start == 'trial':
        label0 = args.start_frame if args.start_frame is not None else 0
    else:
        label0 = 0
    if args.x_label == 'time':
        ts = np.arange(label0, label0 + N) / trial.fps
    else:
        ts = np.arange(N) + label0

    # Plot
    if layout == 'paper':
        plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('legend', fontsize=5)  # fontsize of the legend
        plt.rc('xtick.major', pad=2, size=2)
        plt.rc('ytick.major', pad=1, size=2)
        fig, ax = plt.subplots(1, figsize=(3.5, 1.3), gridspec_kw={
            'top': 0.99,
            'bottom': 0.15,
            'left': 0.06,
            'right': 0.87,
        })
        ylabel_pad = 1
        ylabel_fontsize = 7
        xlabel_pad = 1
        ht_fontsize = 6.5
        cax = ax.inset_axes([1.03, 0.1, 0.04, 0.8], transform=ax.transAxes)
    else:
        fig, ax = plt.subplots(1, figsize=(6, 3))
        ax.set_title(f'Curvature\n'
                     f'Trial {trial.id} ({trial.date:%Y-%m-%d} #{trial.trial_num}).\n'
                     f'Reconstruction {reconstruction.id} ({reconstruction.source}).')
        ylabel_pad = 10
        ylabel_fontsize = 12
        xlabel_pad = None
        ht_fontsize = 12
        cax = ax.inset_axes([1.03, 0.1, 0.02, 0.8], transform=ax.transAxes)

    im = ax.imshow(K.T, aspect='auto', cmap='Reds', origin='lower', extent=(ts[0], ts[-1], 0, 1),
                   vmin=0, vmax=args.K_max if args.K_max is not None and args.K_max > 0 else K.max())
    cb = fig.colorbar(im, ax=ax, cax=cax)
    cb.set_label('Curvature (mm$^{-1}$)', rotation=270, labelpad=8, fontsize=5)
    cb.set_ticks([0, 5, 10, 15])
    ht_args = dict(transform=ax.transAxes, horizontalalignment='right', fontweight='bold', fontsize=ht_fontsize)
    ax.text(-0.02, 0.98, 'T', verticalalignment='top', **ht_args)
    ax.text(-0.02, 0.01, 'H', verticalalignment='bottom', **ht_args)
    ax.set_ylabel('$\kappa$', fontsize=ylabel_fontsize, labelpad=ylabel_pad)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([])
    if args.x_label == 'time':
        ax.set_xlabel('Time (s)', labelpad=xlabel_pad)
    else:
        ax.set_xlabel('Frame #', labelpad=xlabel_pad)

    if layout == 'general':
        fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_curvature_kymogram' \
                           f'_t={trial.id:03d}' \
                           f'_f={meta["start_frame"]}-{meta["end_frame"]}' \
                           f'_r={reconstruction.id}_{reconstruction.source}' \
                           f'_sw={args.smoothing_window}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    curvature_kymogram(layout='paper')
