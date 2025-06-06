import os
from argparse import Namespace, ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mayavi import mlab

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Eigenworms, Dataset
from wormlab3d.postures.cpca import load_cpca_from_file
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d, plot_natural_frame_3d_mlab
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory

plot_n_components = 7
show_plots = True
save_plots = True
img_extension = 'svg'
eigenworm_length = 1
eigenworm_scale = 64
cmap = cm.get_cmap(MIDLINE_CMAP_DEFAULT)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot an eigenworm basis.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset by id.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--n-components', type=int, default=10,
                        help='Number of eigenworms to use (basis dimension).')
    parser.add_argument('--restrict-concs', type=lambda s: [float(item) for item in s.split(',')],
                        help='Restrict to specified concentrations.')
    parser.add_argument('--cpca-file', type=str,
                        help='Load CPCA from file.')
    args = parser.parse_args()

    targets = np.array([getattr(args, k) is not None for k in ['reconstruction', 'dataset', 'cpca_file']],
                       dtype=bool)
    assert targets.sum() == 1, 'One of --reconstruction, --dataset or --cpca-file must be defined.'

    print_args(args)

    return args


def _plot_eigenworms(
        eigenworms: Eigenworms,
        title: str,
        filename: str
):
    n_rows = 3
    n_cols = plot_n_components
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    gs = GridSpec(n_rows, n_cols)

    for i in range(plot_n_components):
        component = eigenworms.components[i]
        NF = NaturalFrame(component * eigenworm_scale, length=eigenworm_length)
        N = NF.N
        ind = np.arange(N)
        fc = cmap((np.arange(N) + 0.5) / N)

        # 3D plot of eigenworm
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ax = plot_natural_frame_3d(
            NF,
            show_frame_arrows=True,
            n_frame_arrows=20,
            arrow_scale=0.2,
            show_pca_arrows=False,
            ax=ax
        )
        ax.set_title(f'Component = {i}')

        # Psi polar plot
        ax = fig.add_subplot(gs[1, i], projection='polar')
        ax.set_title('$\psi=arg(m_1+i m_2)$')
        for j in range(N - 1):
            ax.plot(NF.psi[j:j + 2], ind[j:j + 2], c=fc[j])
        ax.set_rticks([])
        thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
        ax.set_xticks(thetaticks)
        ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
        ax.xaxis.set_tick_params(pad=-3)

        # Curvature plot
        ax = fig.add_subplot(gs[2, i])
        if i == 0:
            kappa_share_ax = ax
        else:
            ax.sharey(kappa_share_ax)
        ax.set_title('$|\kappa|=|m_1+i m_2|$')
        for j in range(N - 1):
            ax.plot(ind[j:j + 2], NF.kappa[j:j + 2], c=fc[j])
        ax.set_xticks([0, ind[-1]])
        ax.set_xticklabels(['H', 'T'])

    fig.suptitle(title.replace('.\n', '. '))
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_eigenworms_{filename}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    plt.close(fig)


def _plot_eigenworms_basic(
        eigenworms: Eigenworms,
        filename: str
):
    plot_config = {
        0: {
            'azim': -118,
            'elev': 15,
            'zoom': 0.8,
            'arrow_scale': 0.1,
        },
        1: {
            'azim': -30,
            'elev': -165,
            'zoom': 0.8,
            'arrow_scale': 0.1,
        },
        2: {
            'azim': -60,
            'elev': -160,
            'zoom': 0.6,
            'arrow_scale': 0.08,
        },
        3: {
            'azim': -160,
            'elev': 30,
            'zoom': 1,
            'arrow_scale': 0.12,
        },
        4: {
            'azim': -20,
            'elev': -165,
            'zoom': 1,
            'arrow_scale': 0.12,
        }
    }

    default_plot_options = {
        'arrow_opts': {
            'linewidth': 2
        },
        'midline_opts': {
            's': 30
        }
    }

    for i in range(plot_n_components):
        if i not in plot_config:
            continue

        component = eigenworms.components[i]
        NF = NaturalFrame(component * eigenworm_scale, length=eigenworm_length)

        # 3D plot of eigenworm
        fig = plot_natural_frame_3d(
            NF,
            show_frame_arrows=True,
            n_frame_arrows=16,
            show_pca_arrows=False,
            **plot_config[i],
            **default_plot_options
        )
        ax = fig.gca()
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_zticks([])
        ax.set_zticklabels([])
        ax.grid(False)

        if save_plots:
            path = LOGS_PATH / f'{START_TIMESTAMP}_eigenworms_c={i}_{filename}.{img_extension}'
            logger.info(f'Saving plot to {path}.')
            plt.savefig(path)

        if show_plots:
            plt.show()

        plt.close(fig)


def _plot_eigenworms_basic_mlab(
        eigenworms: Eigenworms,
        filename: str,
        interactive: bool = True,
        transparent_bg: bool = True,
):
    """
    Use mayavi to plot the eigenworms.
    """
    plot_config = {
        0: {
            'azimuth': 70,
            'elevation': 45,
            'roll': -135
        },
        1: {
            'azimuth': -110,
            'elevation': 40,
            'roll': -5,
        },
        2: {
            'azimuth': -160,
            'elevation': 50,
            'roll': 80,
        },
        3: {
            'azimuth': 125,
            'elevation': 140,
            'roll': -15,
            'distance': 1.5,
        },
        4: {
            'azimuth': 165,
            'elevation': 65,
            'roll': 100,
            'distance': 1.7,
        }
    }

    default_plot_options = {
        'azimuth': -60,
        'elevation': 60,
        'roll': -45,
        'distance': 1.8,
        'arrow_scale': 0.12,
        'arrow_opts': {
            'radius_shaft': 0.02,
            'radius_cone': 0.1,
            'length_cone': 0.2,
        },
        'midline_opts': {
            'line_width': 8
        }
    }

    for i in range(plot_n_components):
        if i not in plot_config:
            continue
        logger.info(f'Plotting component {i}.')

        component = eigenworms.components[i]
        NF = NaturalFrame(component * eigenworm_scale, length=eigenworm_length)

        # 3D plot of eigenworm
        fig = plot_natural_frame_3d_mlab(
            NF,
            show_frame_arrows=True,
            n_frame_arrows=16,
            show_pca_arrows=False,
            show_outline=False,
            show_axis=False,
            offscreen=not interactive,
            **{**default_plot_options, **plot_config[i]}
        )

        if save_plots:
            path = LOGS_PATH / f'{START_TIMESTAMP}_eigenworms_c={i}_{filename}.{img_extension}'
            logger.info(f'Saving plot to {path}.')

            if not transparent_bg:
                mlab.savefig(str(path), figure=fig)
            else:
                fig.scene._lift()
                img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
                img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
                img.save(path)
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


def _plot_eigenvalues(
        eigenworms: Eigenworms,
        title: str,
        filename: str
):
    fig, axes = plt.subplots(2)
    ax = axes[0]
    ax.set_title('Explained variance')
    ax.plot(np.cumsum(eigenworms.explained_variance))
    ax = axes[1]
    ax.set_title('Explained variance ratio')
    ax.plot(np.cumsum(eigenworms.explained_variance_ratio))

    fig.suptitle(title)
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_eigenvalues_{filename}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    plt.close(fig)


def _plot_eigenvalues_basic(
        eigenworms: Eigenworms,
        filename: str,
        format: str = 'paper'
):
    vr = np.cumsum([0, *eigenworms.explained_variance_ratio[:plot_n_components]])
    xs = np.arange(len(vr))
    if format == 'paper':
        highlight_idxs = [4, 5]  # 1-based index!
    else:
        highlight_idxs = [4, 5]  # 1-based index!

    NPs = []
    Hs = []
    for i in range(plot_n_components):
        component = eigenworms.components[i]
        NF = NaturalFrame(component * eigenworm_scale, length=eigenworm_length)
        NPs.append(NF.non_planarity())
        Hs.append(NF.helicity())
    Hs = np.array(Hs)

    # Make plots
    if format == 'paper':
        plt.rc('axes', labelsize=6)  # fontsize of the X label
        plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
        plt.rc('xtick.major', pad=2, size=2)
        plt.rc('ytick.major', pad=2, size=2)
        fig, ax = plt.subplots(1, figsize=(1.94, 1.2), gridspec_kw={
            'left': 0.166,
            'right': 0.843,
            'top': 0.983,
            'bottom': 0.173,
        })
        ax.set_xlabel('Component', labelpad=0)
        ax.set_ylabel('Cumulative variance', labelpad=2)
        # ax.set_yticks([0, 0.4, 0.8, 1.0])
        # ax.set_yticklabels([0, 0.4, 0.8, None])
        ax.set_yticks(vr)
        ax.set_yticklabels([f'{v:.2f}' if i < 6 else None for i, v in enumerate(vr)])
        cv_colour = 'black'
        scat_size = 35
        redline_width = 1.5

    else:
        plt.rc('axes', labelsize=9)  # fontsize of the X label
        plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
        plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
        plt.rc('xtick.major', pad=2, size=3)
        plt.rc('ytick.major', pad=2, size=3)
        fig, ax = plt.subplots(1, figsize=(4, 2.1), gridspec_kw={
            'left': 0.14,
            'right': 0.86,
            'top': 0.98,
            'bottom': 0.18,
        })
        ax.set_xlabel('Component', labelpad=5)
        ax.set_ylabel('Cumulative variance', labelpad=8)
        ax.set_yticks(vr)
        ax.set_yticklabels([f'{v:.2f}' if i < 6 else None for i, v in enumerate(vr)])
        cv_colour = 'C0'
        scat_size = 50
        redline_width = 2

    ax.grid()
    ax.scatter(xs[1:], vr[1:], zorder=20, s=scat_size, c=cv_colour)
    ax.plot(xs, vr, zorder=5, alpha=0.5, linestyle=':', color=cv_colour)
    ax.set_xlim(0, plot_n_components + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([2, 4, 6])
    ax.spines['right'].set_visible(False)

    for highlight_idx in highlight_idxs:
        ax.hlines(xmin=0, xmax=highlight_idx, y=vr[highlight_idx],
                  color='red', linewidth=redline_width, zorder=9)
        ax.vlines(ymin=0, ymax=vr[highlight_idx], x=highlight_idx,
                  color='red', linewidth=redline_width, zorder=9)
        # ax.text(-0.1, vr[highlight_idx], f'{vr[highlight_idx]:.2f}',
        #         color='red', fontsize=8, ha='right', va='center', transform=ax.transData)

    ax_nonp = ax.twinx()
    if format == 'paper':
        ax_nonp.set_ylabel('Non-planarity / Helicity', rotation=270, labelpad=7)
    else:
        ax_nonp.set_ylabel('Non-planarity / Helicity', rotation=270, labelpad=15)
    ax_nonp.scatter(xs[1:], NPs, zorder=8, s=50, alpha=0.7, color='orange', marker='x')
    ax_nonp.plot(xs[1:], NPs, zorder=4, alpha=0.3, color='orange', linestyle='--')
    ax_nonp.set_yticks([0, 0.01, 0.02])
    ax_nonp.set_yticklabels([0, 0.01, 0.02])

    ax_nonp.spines['right'].set_linewidth(1.5)
    ax_nonp.spines['right'].set_linestyle((0, (3, 2)))
    ax_nonp.spines['right'].set_alpha(0.8)
    ax_nonp.spines['right'].set_color('orange')

    # Helicity
    ax_hel = ax.twinx()
    ax_hel.set_yticks([])
    ax_hel.spines['right'].set_visible(False)
    h_lim = np.abs(Hs).max() * 1.1
    ax_hel.set_ylim(bottom=-h_lim, top=h_lim)
    n_fade_lines = 100
    fade_lines_pos = np.linspace(0, Hs.max(), n_fade_lines)
    fade_lines_neg = np.linspace(0, Hs.min(), n_fade_lines)
    alpha_max = 1

    if format == 'paper':
        hel_col_pos = 'grey'
        hel_col_neg = 'grey'
    else:
        hel_col_pos = 'purple'
        hel_col_neg = 'green'

    for i, H in enumerate(Hs):
        x_bounds = np.array([xs[i + 1] - 0.25, xs[i + 1] + 0.25])
        h = np.ones(2) * H
        for j in range(n_fade_lines):
            if H > 0:
                ax_hel.fill_between(
                    x_bounds,
                    np.ones_like(h) * fade_lines_pos[j],
                    h,
                    where=h > fade_lines_pos[j],
                    color=hel_col_pos,
                    alpha=alpha_max / n_fade_lines,
                    linewidth=0,
                    zorder=-100,
                )
            else:
                ax_hel.fill_between(
                    x_bounds,
                    h,
                    np.ones_like(h) * fade_lines_neg[j],
                    where=h < fade_lines_neg[j],
                    color=hel_col_neg,
                    alpha=alpha_max / n_fade_lines,
                    linewidth=0,
                    zorder=-100,
                )

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_eigenvalues_basic_{filename}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()

    plt.close(fig)


def _plot_reconstruction(
        eigenworms: Eigenworms,
        X_full: np.ndarray,
        idx: int,
        title: str,
        filename: str
):
    # Reconstruct a worm using the basis
    X = X_full[idx]
    NF_original = NaturalFrame(X)
    coeffs = eigenworms.transform(np.array([NF_original.mc]))
    mc = eigenworms.inverse_transform(coeffs)
    NF_reconstructed = NaturalFrame(
        mc[0],
        X0=NF_original.X_pos[0],
        T0=NF_original.T[0],
        M0=NF_original.M1[0],
    )

    NF_original_args = {'c': 'red', 'linestyle': '--', 'alpha': 0.8, 'label': 'Original'}
    NF_reconst_args = {'c': 'blue', 'linestyle': ':', 'alpha': 0.8, 'label': 'Reconstructed'}
    N = NF_original.N
    ind = np.arange(N)

    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(3, 2)
    gs2 = GridSpec(3, 2, wspace=0.25, hspace=0.2, left=0.1, right=0.95, bottom=0.05, top=0.9)

    # 3D plots of eigenworms
    for i, NF in enumerate([NF_original, NF_reconstructed]):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ax = plot_natural_frame_3d(
            NF,
            show_frame_arrows=True,
            n_frame_arrows=20,
            arrow_scale=0.2,
            show_pca_arrows=False,
            ax=ax
        )
        ax.set_title(['Original', 'Reconstruction'][i])

    # Kappa
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title('$|\kappa|=|m_1|+|m_2|$')
    ax.plot(ind, NF_original.kappa, **NF_original_args)
    ax.plot(ind, NF_reconstructed.kappa, **NF_reconst_args)
    ax.set_xticks([0, ind[-1]])
    ax.set_xticklabels(['H', 'T'])
    ax.legend()

    # Psi
    ax = fig.add_subplot(gs2[1, 1], projection='polar')
    ax.set_title('$\psi=arg(m_1+i m_2)$')
    ax.plot(NF_original.psi, ind, **NF_original_args)
    ax.plot(NF_reconstructed.psi, ind, **NF_reconst_args)
    ax.set_rticks([])
    thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
    ax.set_xticks(thetaticks)
    ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
    ax.xaxis.set_tick_params(pad=-3)

    # Coefficients
    ax = fig.add_subplot(gs[2, :])
    ax.set_title('Coefficient magnitudes')
    ax.plot(np.abs(coeffs)[0])

    fig.suptitle(title + f' Idx={idx}.')
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_eigen-reconstruction_{filename}_idx={idx}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    plt.close(fig)


def sweep_dataset_concentrations():
    """
    Sweep over all concentrations in a dataset and plot the eigenworms and values for each subset.
    """
    args = parse_args()
    assert args.dataset is not None
    dataset = Dataset.objects.get(id=args.dataset)
    concs = sorted(list(dataset.metas['concentration'].keys()))

    for c in concs:
        ew = generate_or_load_eigenworms(
            dataset_id=args.dataset,
            n_components=args.n_components,
            restrict_concs=[float(c), ],
            regenerate=False
        )

        title = f'Dataset {dataset.id}. Concentration {c}.\nNum worms={ew.n_samples}. Num points={ew.n_features}.'
        filename = f'ds={dataset.id}_c={c}_M={ew.n_samples}_N={ew.n_features}'

        _plot_eigenworms(ew, title, filename)
        _plot_eigenvalues_basic(ew, filename)


def main():
    args = parse_args()
    X_full = None

    if args.dataset is not None:
        ew = generate_or_load_eigenworms(
            dataset_id=args.dataset,
            n_components=args.n_components,
            restrict_concs=args.restrict_concs,
            regenerate=False
        )
        dataset = Dataset.objects.get(id=args.dataset)
        # X_full = dataset.X_all

        title = f'Dataset {dataset.id}.'
        filename = f'ds={dataset.id}_ew={ew.id}'

        if args.restrict_concs is not None:
            title += f' Concentrations [' + ', '.join([f'{c}%' for c in args.restrict_concs]) + '].'
            filename += f'_concs=' + ','.join([f'{c}' for c in args.restrict_concs])

    elif args.reconstruction is not None:
        ew = generate_or_load_eigenworms(
            reconstruction_id=args.reconstruction,
            n_components=args.n_components,
            regenerate=False
        )
        reconstruction = Reconstruction.objects.get(id=args.reconstruction)
        X_full, meta = get_trajectory(reconstruction_id=args.reconstruction)

        title = f'Trial {reconstruction.trial.id}.\n' \
                f'Reconstruction={reconstruction.id} ({reconstruction.source}).'

        filename = f'trial={reconstruction.trial.id:03d}_' \
                   f'reconstruction={reconstruction.id}_{reconstruction.source}_' \
                   f'ew={ew.id}'

    elif args.cpca_file is not None:
        path = Path(args.cpca_file)
        fn = path.parts[-1]
        assert path.exists(), 'CPCA file not found!'
        ew = Eigenworms()
        ew.cpca = load_cpca_from_file(path)
        ew.components = ew.cpca.components_
        ew.n_samples = ew.cpca.n_samples_
        ew.n_features = ew.cpca.n_features_
        ew.n_components = ew.cpca.n_components_
        title = f'Basis: {fn}.'
        filename = f'basis={fn}'

    title += f'\nNum worms={ew.n_samples}. Num points={ew.n_features}.'
    filename += f'_M={ew.n_samples}_N={ew.n_features}'

    # _plot_eigenworms_basic(ew, filename)
    # _plot_eigenworms_basic_mlab(ew, filename, interactive=False)
    # _plot_eigenworms(ew, title, filename)
    # _plot_eigenvalues(ew, title, filename)
    _plot_eigenvalues_basic(ew, filename, format='paper')
    # if X_full is not None:
    #     _plot_reconstruction(ew, X_full, 500, title, filename)


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # sweep_dataset_concentrations()
    main()
