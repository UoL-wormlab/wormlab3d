from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from flask import request

from app.util.encoders import base64img
from app.views.api import bp_api
from app.views.page.reconstructions import CAM_PARAMETER_KEYS
from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d.data.model import Reconstruction, Frame
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF, Midline3D
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache


@bp_api.route('/reconstruction/<string:_id>/trajectory', methods=['GET'])
def get_trajectory_data(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    X, meta = get_trajectory(
        reconstruction_id=_id,
        depth=-1,
        trajectory_point=-1,
    )

    # Replace NaN's with Nones to allow json decoding on the frontend
    Xo = X.astype(np.object)
    Xo[np.isnan(X)] = None

    # Prune timestamps
    timestamps = _get_timestamps(reconstruction, meta['frame_nums'])
    timestamps = timestamps[:len(Xo)]

    response = {
        'timestamps': timestamps,
        'X': Xo.T.tolist(),
    }
    return response


@bp_api.route('/reconstruction/<string:_id>/stats', methods=['GET'])
def get_stats(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    ts = TrialState(reconstruction=reconstruction)
    key = request.args.get('key')
    assert key in ts.stats
    return {
        'timestamps': _get_timestamps(reconstruction),
        'values': ts.stats[key],
    }


@bp_api.route('/reconstruction/<string:_id>/cameras', methods=['GET'])
def get_cameras_parameters(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    ts = TrialState(reconstruction=reconstruction)

    key = request.args.get('key')
    assert key in CAM_PARAMETER_KEYS

    if key == 'Intrinsics':
        p = ts.get('cam_intrinsics')  # (T, 3, 4)
        titles = ['fx', 'fy', 'cx', 'cy']

    elif key == 'Rotations':
        p = ts.get('cam_rotation_preangles')  # (T, 3, 3, 2)
        p = np.arctan2(p[:, :, :, 0], p[:, :, :, 1])
        titles = ['theta', 'phi', 'psi']

    elif key == 'Translations':
        p = ts.get('cam_translations')  # (T, 3, 3)
        titles = ['x', 'y', 'z']

    elif key == 'Distortions':
        p = ts.get('cam_distortions')  # (T, 3, 5)
        titles = ['k1', 'k2', 'p1', 'p2', 'k3']

    elif key == 'Shifts':
        p = ts.get('cam_shifts')  # (T, 3, 1)
        titles = ['ds']

    p = p.transpose(2, 1, 0)  # (4, 3, T)

    return {
        'timestamps': _get_timestamps(reconstruction),
        'data': p.tolist(),
        'shape': tuple(p.shape),
        'titles': titles
    }


@bp_api.route('/reconstruction/<string:_id>/render-stats', methods=['GET'])
def get_render_stats(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    ts = TrialState(reconstruction=reconstruction)

    vals = {}
    for k in ['sigmas', 'exponents', 'intensities']:
        vals[k] = {'points': {}, 'sfs': {}}
        ps = ts.get(k)
        for i, d in enumerate(range(ts.parameters.depth_min, ts.parameters.depth)):
            vals[k]['points'][d] = ps[:, i].tolist()
        sfs = ts.get(f'camera_{k}')
        for i in range(3):
            vals[k]['sfs'][i] = sfs[:, i].tolist()

    return {
        'timestamps': _get_timestamps(reconstruction),
        **vals
    }


@bp_api.route('/reconstruction/<string:_id>/posture', methods=['GET'])
def get_posture(_id):
    reconstruction = Reconstruction.objects.get(id=_id)
    frame_num = request.args.get('frame_num', type=int)
    frame = reconstruction.get_frame(frame_num)
    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    masks = None

    if reconstruction.source == M3D_SOURCE_MF:
        D = request.args.get('depth', type=int)
        D_min = reconstruction.mf_parameters.depth_min
        idx_offset = 2**D_min - 1
        ts = TrialState(reconstruction=reconstruction)
        from_idx = sum([2**d2 for d2 in range(D)]) - idx_offset
        to_idx = from_idx + 2**D
        d_idx = D - D_min

        # Get 3D posture
        all_points = ts.get('points')[frame_num]
        points = all_points[from_idx:to_idx].T

        # Get 2D projections
        all_projections = ts.get('points_2d')[frame_num]  # (N, 3, 2)
        points_2d = np.round(all_projections[from_idx:to_idx]).astype(np.int32)

        # Get body sigma
        N4 = int(2**D / 4)
        sigma = ts.get('sigmas')[frame_num][d_idx]
        min_sigma = 0.04
        slopes = (sigma - min_sigma) / N4 * np.arange(N4) + min_sigma
        sigmas = np.concatenate([
            slopes,
            np.ones(2 * N4) * sigma,
            slopes[::-1]
        ])

        # Get masks
        masks_target = ts.get('masks_target')[frame_num]

        # Colour map
        colours = np.array([cmap(d) for d in np.linspace(0, 1, 2**D)])
        colours = np.round(colours * 255).astype(np.uint8)

        # Prepare images
        images = _generate_annotated_images(frame, points_2d, colours)

        # Prepare masks with overlaid vertices scaled by sigmas.
        masks = []
        for i, mask_array in enumerate(masks_target):
            z = (mask_array * 255).astype(np.uint8)
            z = cv2.cvtColor(z, cv2.COLOR_GRAY2RGB)

            # Overlay 2d projection
            p2d = points_2d[:, i]
            for j, p in enumerate(p2d):
                z = cv2.circle(z, p, int(sigmas[j] * 100), color=colours[j].tolist(), thickness=-1,
                               lineType=cv2.LINE_AA)

            img_str = base64img(z, image_mode='RGB')
            masks.append(img_str)

        # Fetch curvatures
        K = ts.get('curvatures')[frame_num] * (2**D - 1)
        m1 = K[..., 0]
        m2 = K[..., 1]
        curvatures = {
            'k': np.linalg.norm(K, axis=-1).tolist(),
            'psi': np.arctan2(m1, m2).tolist(),
            'm1': m1.tolist(),
            'm2': m2.tolist(),
            'xs': np.linspace(0, 1, 2**D + 2)[1:-1].tolist()
        }
    else:
        m3d = Midline3D.objects.get(
            frame=frame,
            source=reconstruction.source,
            source_file=reconstruction.source_file,
        )
        points = m3d.X.T

        # Get 2D projections
        points_2d = np.round(m3d.get_prepared_2d_coordinates()).astype(np.int32)
        points_2d = points_2d.transpose(1, 0, 2)

        # Colour map
        colours = np.array([cmap(i) for i in np.linspace(0, 1, points_2d.shape[0])])
        colours = np.round(colours * 255).astype(np.uint8)

        # Prepare images
        images = _generate_annotated_images(frame, points_2d, colours)

        # No curvatures (for now)
        curvatures = []

    response = {
        'images': images,
        'posture': points.tolist(),
        'curvatures': curvatures,
    }
    if masks is not None:
        response['masks'] = masks

    return response


@bp_api.route('/reconstruction/<string:_id>/worm-lengths', methods=['GET'])
def get_worm_lengths(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    D = reconstruction.mf_parameters.depth
    D_min = reconstruction.mf_parameters.depth_min
    ts = TrialState(reconstruction=reconstruction)
    l = ts.get('length')  # (T, D)
    lengths = {}
    for i, d in enumerate(range(D_min, D)):
        lengths[d] = l[:, i].tolist()

    return {
        'timestamps': _get_timestamps(reconstruction),
        'lengths': lengths
    }


@bp_api.route('/reconstruction/<string:_id>/planarity', methods=['GET'])
def get_planarity(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)

    pcas, meta = generate_or_load_pca_cache(
        reconstruction_id=reconstruction.id,
        window_size=1,
    )

    r = pcas.explained_variance_ratio.T
    planarities = 1 - r[2] / np.sqrt(r[1] * r[0])

    return {
        'timestamps': _get_timestamps(reconstruction),
        'planarities': planarities.tolist()
    }


def _get_timestamps(reconstruction: Reconstruction, frame_nums: List[int] = None):
    if frame_nums is None:
        frame_nums = np.arange(reconstruction.start_frame, reconstruction.end_frame)
    timestamps = [n / reconstruction.trial.fps for n in frame_nums]
    return timestamps


def _generate_annotated_images(frame: Frame, points_2d: np.ndarray, colours: np.ndarray) -> List[str]:
    """
    Prepare images with overlaid midlines as connecting lines between vertices.
    """
    images = []
    for i, img_array in enumerate(frame.images):
        z = (img_array * 255).astype(np.uint8)
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2RGB)

        # Overlay 2d projection
        p2d = points_2d[:, i]
        for j, p in enumerate(p2d):
            z = cv2.drawMarker(
                z,
                p,
                color=colours[j].tolist(),
                markerType=cv2.MARKER_CROSS,
                markerSize=3,
                thickness=1,
                line_type=cv2.LINE_AA
            )
            if j > 0:
                cv2.line(
                    z,
                    p2d[j - 1],
                    p2d[j],
                    color=colours[j].tolist(),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

        img_str = base64img(z, image_mode='RGB')
        images.append(img_str)

    return images
