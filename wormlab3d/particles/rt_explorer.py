import gc
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import Any, Dict, Optional

import numpy as np
import torch
from copulas.multivariate import GaussianMultivariate
from progress.bar import Bar
from scipy.interpolate import PchipInterpolator
from torch import nn

from wormlab3d import N_WORKERS, logger
from wormlab3d.data.model import Dataset
from wormlab3d.midlines3d.mf_methods import normalise
from wormlab3d.particles.particle_explorer import orthogonalise
from wormlab3d.particles.tumble_run import generate_or_load_ds_statistics

PARTICLE_PARAMETER_KEYS = ['thetas', 'phis']


def simulate_particle_trajectory(
        n_steps: int,
        durations: torch.Tensor,
        speeds: torch.Tensor,
        e0s: torch.Tensor,
        pauses: torch.Tensor,
) -> torch.Tensor:
    """
    Simulate an individual particle trajectory.
    """
    step = 0
    i = 0
    e0 = e0s[0]
    e0_exp = torch.zeros((n_steps, 3))

    while step < n_steps:
        # Run
        run_steps = durations[i]
        run_speed = speeds[i]
        if step + run_steps > n_steps:
            e0_exp[step:] = e0 * run_speed
            break
        e0_exp[step:step + run_steps] = e0 * run_speed

        # Tumble
        e0 = e0s[i]

        # Pause
        pause_steps = pauses[i]

        # Increment
        i += 1
        step += run_steps + pause_steps

    return e0_exp


def simulate_particle_trajectory_wrapper(args):
    return simulate_particle_trajectory(*args)


class RTExplorer(nn.Module):
    thetas: torch.Tensor
    phis: torch.Tensor

    def __init__(
            self,
            dataset: Dataset,
            approx_args: Dict[str, Any],
            batch_size: int = 20,
            x0: torch.Tensor = None,
            state0: torch.Tensor = None,
            phi_factor: float = 1.,
            nonp_pause_type: Optional[str] = None,
            nonp_pause_max: float = 0.,
            n_angle_bins: int = 10,
            quiet: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.approx_args = approx_args
        self.batch_size = batch_size
        self.quiet = quiet

        # Squash or expand the sampled nonplanar angles
        self.phi_factor = phi_factor

        # Should nonplanar turns induce a longer pause than planar turns
        self.nonp_pause_type = nonp_pause_type
        self.nonp_pause_max = nonp_pause_max

        # Number of angle bins for the angles model
        self.n_angle_bins = n_angle_bins

        self._init_particle(x0)
        self._init_state(state0)
        self._init_run_model()
        self._init_angles_model()

    def _log(self, msg: str, level: str = 'info'):
        if not self.quiet:
            getattr(logger, level)(msg)

    def _progress(self, msg: str, n_steps: int) -> Optional[Bar]:
        bar = None
        if not self.quiet:
            bar = Bar(msg, max=n_steps)
            bar.check_tty = False
        return bar

    def _init_particle(self, x0: Optional[torch.Tensor] = None):
        """
        Initialise the particle position and orientation.
        """

        # Starting position
        if x0 is None:
            x0 = torch.zeros((self.batch_size, 3))
        else:
            assert x0.shape == (self.batch_size, 3)
            x0 = x0.to(torch.float64)
        self.register_buffer('x', x0)

        # Heading
        e0 = normalise(torch.rand(self.batch_size, 3, dtype=torch.float64))
        self.register_buffer('e0', e0)

        # Orientation
        e1 = normalise(orthogonalise(torch.rand(self.batch_size, 3, dtype=torch.float64), self.e0))
        self.register_buffer('e1', e1)
        e2 = normalise(torch.cross(e0, e1))
        self.register_buffer('e2', e2)

    def _init_state(
            self,
            state0: Optional[torch.Tensor] = None
    ):
        """
        Initialise the state and state parameters.
        """
        if state0 is None:
            state0 = torch.zeros((self.batch_size,), dtype=torch.uint8)
        else:
            assert state0.shape == (self.batch_size,)
            state0 = state0.to(torch.uint8)
        self.register_buffer('state', state0)

        for pk in PARTICLE_PARAMETER_KEYS:
            self.register_buffer(pk, torch.zeros(self.batch_size))

    def _init_run_model(self):
        """
        Initialise the run model.
        """
        logger.info('Initialising run model.')

        # Generate or load tumble/run values
        ds_stats = generate_or_load_ds_statistics(
            ds=self.dataset,
            rebuild_cache=False,
            **self.approx_args
        )
        # stats = trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles
        durations = ds_stats[1][0]
        speeds = ds_stats[2][0]

        # Fit a copula to the durations and speeds
        data = np.column_stack((durations, speeds))
        copula = GaussianMultivariate()
        copula.fit(data)
        self.run_model = copula

    def _init_angles_model(self):
        """
        Initialise the angles model.
        """
        logger.info('Initialising angles model.')

        # Generate or load tumble/run values
        ds_stats = generate_or_load_ds_statistics(
            ds=self.dataset,
            rebuild_cache=False,
            **self.approx_args
        )
        # stats = trajectory_lengths, durations, speeds, planar_angles, nonplanar_angles, twist_angles
        thetas = ds_stats[3][0]
        phis = ds_stats[4][0]
        data = np.array([thetas, phis]).T

        # Put all data in the top right quadrant
        data = np.abs(data)

        # Convert data to polar coordinates
        r = np.linalg.norm(data, axis=1)
        psi = np.arctan2(data[:, 1], data[:, 0])
        assert np.all(psi >= 0) and np.all(psi <= np.pi / 2), 'psi must be between 0 and pi/2!'
        data_p = np.array([r, psi]).T

        # Split the data into wedges based on the angles
        angle_bins = np.linspace(0, np.pi / 2, self.n_angle_bins + 1)
        psi_bin_idxs = np.digitize(psi, angle_bins)
        psi_bin_idxs -= 1
        data_slices_by_psi = [data_p[psi_bin_idxs == i] for i in range(self.n_angle_bins)]

        def make_cdf(vals: np.ndarray):
            vals = np.sort(vals)
            k = len(vals)
            if k == 0:
                vals = np.array([0, 1e-4])
            elif k == 1:
                vals = np.concatenate(([np.min(vals) * 0.9], vals))
            k = len(vals)
            y = np.arange(0, k) / (k - 1)
            interp = PchipInterpolator(vals, y, extrapolate=False)
            interp_inv = PchipInterpolator(y, vals, extrapolate=False)

            def interp_func(x, inverse=False):
                if inverse:
                    return interp_inv(x)
                y = interp(x)
                y = np.where(x < vals[0], 0, np.where(x > vals[-1], 1, y))
                return y

            return interp_func

        # Calculate the psi distribution for all r values
        cdfs_psi_all = make_cdf(data_p[:, 1])

        # For each of the psi data slices, make a cdf for the distribution of r values
        cdfs_r = []
        for i, data_slice in enumerate(data_slices_by_psi):
            cdfs_r.append(make_cdf(data_slice[:, 0]))

        def angles_model(n_samples: int):
            # Sample some new data
            z1 = np.random.rand(n_samples)
            z2 = np.random.rand(n_samples)
            psi_vals = cdfs_psi_all(z1, inverse=True)
            r_vals_all = np.zeros((self.n_angle_bins, n_samples))
            for i, cdf in enumerate(cdfs_r):
                r_vals_all[i] = cdf(z2, inverse=True)
            pbi = np.digitize(psi_vals, angle_bins) - 1

            # Interpolate the r values across neighbouring bins depending on their position in the bin
            psi_bin_positions = (psi_vals - angle_bins[pbi]) / (angle_bins[pbi + 1] - angle_bins[pbi])
            pos_idx = np.arange(n_samples)
            r_vals_left = r_vals_all[np.clip(pbi - 1, a_min=0, a_max=None), pos_idx]
            r_vals_middle = r_vals_all[pbi, pos_idx]
            r_vals_right = r_vals_all[np.clip(pbi + 1, a_min=0, a_max=self.n_angle_bins - 1), pos_idx]
            r_vals = (r_vals_middle * (1 - np.abs(psi_bin_positions - 0.5))
                      + r_vals_left * np.clip(0.5 - psi_bin_positions, a_min=0, a_max=None)
                      + r_vals_right * np.clip(psi_bin_positions - 0.5, a_min=0, a_max=None))

            # Turn the data back into cartesian
            fake_data = np.array([r_vals * np.cos(psi_vals), r_vals * np.sin(psi_vals)]).T

            # Randomly flip the x and y values
            flip_x = np.random.rand(n_samples) > 0.5
            flip_y = np.random.rand(n_samples) > 0.5
            fake_data[:, 0] *= np.where(flip_x, -1, 1)
            fake_data[:, 1] *= np.where(flip_y, -1, 1)

            return fake_data

        self.angles_model = angles_model

    def forward(
            self,
            T: float,
            dt: float
    ) -> Dict[str, Any]:
        """
        Simulate a batch of particle explorers.
        """
        start_time = time.time()
        n_steps = int(T / dt)
        ts = torch.arange(n_steps) * dt
        nonp_pause_max = self.nonp_pause_max / dt

        # Generate the run durations and speeds
        self._log('Generating run durations and speeds.')
        durations = torch.zeros((self.batch_size, 0), dtype=torch.int32)
        speeds = torch.zeros((self.batch_size, 0))
        min_particle_runtime = 0
        while min_particle_runtime < n_steps:
            d, s = torch.from_numpy(self.run_model.sample(self.batch_size).values.T)
            d = (d / dt).to(torch.int32)
            s = s * dt
            durations = torch.cat([durations, d[..., None]], dim=1)
            speeds = torch.cat([speeds, s[..., None]], dim=1)
            min_particle_runtime = durations.sum(dim=1).min()
        max_tumbles = durations.shape[1]

        # Generate the tumble angles
        self._log('Generating tumble angles.')
        sample_attempts = 0
        max_attempts = 10
        while True:
            thetas, phis = torch.from_numpy(self.angles_model(self.batch_size * max_tumbles).T)
            thetas = torch.atan2(torch.sin(thetas), torch.cos(thetas))
            if (not torch.isnan(thetas).any() and not torch.isinf(thetas).any()
                    and not torch.isnan(phis).any() and not torch.isinf(phis).any()):
                break
            elif sample_attempts == max_attempts:
                raise RuntimeError('Failed to sample numerical angles from the copula!')
            sample_attempts += 1
        thetas = thetas.reshape(self.batch_size, max_tumbles)
        phis *= self.phi_factor
        phis = phis.reshape(self.batch_size, max_tumbles)

        # Generate the directions of the runs
        self._log('Generating tumbles.')
        gc.collect()  # probably in the wrong place but seems to fix the random crashes
        e0s = torch.zeros((self.batch_size, max_tumbles + 1, 3), dtype=torch.float64)
        e0s[:, 0] = self.e0.clone()
        for i in range(max_tumbles):
            self._rotate_frames(thetas[:, i], 'planar')
            self._rotate_frames(phis[:, i], 'nonplanar')
            e0s[:, i + 1] = self.e0.clone()

        # Calculate the pause durations based on how extreme the nonplanar angle is
        self._log('Calculating pause durations.')
        if self.nonp_pause_type is not None:
            if self.nonp_pause_type == 'linear':
                p = 1
            elif self.nonp_pause_type == 'quadratic':
                p = 2
            else:
                raise RuntimeError(f'Unrecognised pause type "{self.nonp_pause_type}".')
            pauses = ((phis.abs() / (torch.pi / 2))**p * nonp_pause_max).round().to(torch.int32)
        else:
            pauses = torch.zeros_like(phis).to(torch.int32)

        # Simulate the particle exploration
        self._log('Simulating particle exploration.')
        if N_WORKERS > 1:
            with Pool(processes=N_WORKERS) as pool:
                res = pool.map(
                    simulate_particle_trajectory_wrapper,
                    [
                        (n_steps, durations[i], speeds[i], e0s[i], pauses[i])
                        for i in range(self.batch_size)
                    ]
                )
                e0_exp = torch.zeros((self.batch_size, n_steps, 3))
                for i in range(self.batch_size):
                    e0_exp[i] = res[i]
        else:
            e0_exp = torch.zeros((self.batch_size, n_steps, 3))
            bar = Bar('Simulating', max=self.batch_size)
            bar.check_tty = False
            for i in range(self.batch_size):
                e0_exp[i] = simulate_particle_trajectory(
                    n_steps=n_steps,
                    durations=durations[i],
                    speeds=speeds[i],
                    e0s=e0s[i],
                    pauses=pauses[i],
                )
                bar.next()
            bar.finish()
        X = torch.cumsum(e0_exp, dim=1)

        # Calculate the tumble times
        self._log('Calculating tumble times.')

        # Distribute the pauses between the runs
        p2 = torch.cat((torch.zeros(self.batch_size, 1), pauses), dim=1)
        distributed_pauses = (p2[:, :-1] + p2[:, 1:]) / 2
        distributed_pauses += torch.randn(
            *distributed_pauses.shape) * 0.0001  # add some noise so that half steps get rounded in both directions
        distributed_pauses = distributed_pauses.round().to(torch.int32)

        # Add the run durations to the distributed pauses so the vertex indices fall in the middle of each pause
        vertex_idxs = torch.cumsum(durations + distributed_pauses, dim=1)

        # Crop to within the simulation time and convert to time units
        n_tumbles = [(vertex_idxs[i] < n_steps).sum() for i in range(self.batch_size)]
        vertex_idxs = [vertex_idxs[i][:n_tumbles[i]] for i in range(self.batch_size)]
        tumble_ts = [idxs * dt for idxs in vertex_idxs]
        durations = [durations[i, :n_tumbles[i]].to(torch.float64) * dt for i in range(self.batch_size)]

        # Generate the states - 0 for run, 1 for tumble
        self._log('Calculating states.')
        states = torch.zeros((self.batch_size, n_steps))
        for i in range(self.batch_size):
            states[i, vertex_idxs[i]] = 1

        # Calculate the sequences - somewhat redundant, but keeps things consistent with the 3-state model
        self._log('Calculating sequences.')
        sequences = []
        for i in range(self.batch_size):
            sequences.append(torch.unique_consecutive(states[i]))

        # Calculate intervals between turns - the real run durations
        intervals = []
        for i in range(self.batch_size):
            tt = torch.cat((torch.zeros(1), tumble_ts[i]))
            intervals.append(tt[1:] - tt[:-1])

        # Calculate average speeds between turns, absorbing pauses into the r
        avg_speeds = []
        for i in range(self.batch_size):
            vertices = torch.cat((torch.zeros(1, 3), X[i, vertex_idxs[i]]))
            distances = (vertices[1:] - vertices[:-1]).norm(dim=-1)
            avg_speeds.append(distances / intervals[i])

        # Prune the angles to those actually used in each run
        thetas = [thetas[i, :n_tumbles[i]] for i in range(self.batch_size)]
        phis = [phis[i, :n_tumbles[i]] for i in range(self.batch_size)]

        sim_time = time.time() - start_time
        self._log(f'Time: {timedelta(seconds=sim_time)}')

        return dict(
            ts=ts,
            tumble_ts=tumble_ts,
            X=X,
            states=states,
            durations_0=durations,
            durations_1=None,
            thetas=thetas,
            phis=phis,
            intervals=intervals,
            speeds=avg_speeds
        )

    def _sample_angles(self, which: str, n: int) -> torch.Tensor:
        """
        Sample angle parameters from the distributions.
        """
        if which == 'thetas':
            dist = self.theta_dist
        elif which == 'phis':
            dist = self.phi_dist
        else:
            raise RuntimeError(f'Unrecognised angle parameter key "{which}".')
        angles = dist.sample((self.batch_size * n,)).reshape(self.batch_size, n)
        angles[angles.isnan() | angles.isinf()] = 0
        if which == 'thetas':
            # Put into range +/- pi
            angles = torch.atan2(torch.sin(angles), torch.cos(angles))
        else:
            # Put into range +/- pi/2
            angles = torch.atan(torch.tan(angles))
        return angles

    def _rotate_frames(self, angles: torch.Tensor, which: str):
        """
        Rotate the frames in a planar or non-planar direction.
        """
        if which == 'planar':
            u = self.e2
        else:
            u = self.e1

        # Convert rotation axis and angle to 3x3 rotation matrix
        # (See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)
        I = torch.eye(3, dtype=torch.float64)[None, ...].repeat(self.batch_size, 1, 1)
        cosA = torch.cos(angles)[:, None, None]
        sinA = torch.sin(angles)[:, None, None]
        outer = torch.einsum('bi,bj->bij', u, u)
        cross = torch.cross(u[..., None].repeat(1, 1, 3), I)

        R = cosA * I \
            + sinA * cross \
            + (1 - cosA) * outer

        # Rotate frame vectors
        self.e0 = normalise(torch.einsum('bij,bj->bi', R, self.e0))
        if which == 'planar':
            self.e1 = normalise(torch.einsum('bij,bj->bi', R, self.e1))

        # Recalculate frame
        self.e1 = normalise(orthogonalise(self.e1, self.e0))
        self.e2 = normalise(torch.cross(self.e0, self.e1))
