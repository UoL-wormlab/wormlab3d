import itertools
from collections import OrderedDict
from typing import List, Union

import numpy as np
from scipy.optimize import minimize
from wormlab3d import logger, CAMERA_IDXS
from wormlab3d.data.model.cameras import Cameras
from wormlab3d.data.model.object_point import ObjectPoint


def triangulate(
        image_points: List[np.ndarray],
        cameras: Cameras,
        x0: Union[list, np.ndarray] = None,
        matching_threshold: float = 1000,
) -> List[ObjectPoint]:
    """
    Triangulate a list of 2D image points obtained from each camera view to produce a list of 3D object
    points which correspond up to some tolerance.
    """
    assert len(image_points) == 3
    assert isinstance(cameras, Cameras)
    if isinstance(image_points, OrderedDict):
        image_points = list(image_points.values())
    for image_points_cam in image_points:
        assert isinstance(image_points_cam, list)
        for pts in image_points_cam:
            assert isinstance(pts, list)
            assert len(pts) == 2
    if x0 is not None:
        if not isinstance(x0, np.ndarray):
            x0 = np.array(x0)
        assert x0.shape == (3,)

    logger.debug(f'Triangulating. '
                 f'Cameras id={cameras.id}. ' +
                 (f'Reprojection error={cameras.reprojection_error:.4f}. '
                  if cameras.reprojection_error is not None else '') +
                 f'Trial={cameras.trial.id if cameras.trial is not None else "-"}.')
    cams_model = cameras.get_camera_model_triplet()

    if x0 is None:
        x0 = np.array((1., 1., 500.))

    def f(x, pts):
        return np.sqrt(
            sum([
                np.linalg.norm(cams_model[c].project_to_2d(x) - pts[c])**2
                for c in CAMERA_IDXS
            ])
        )

    point_keys = [list(range(len(pts))) for pts in image_points]
    point_key_combinations = list(itertools.product(*point_keys))
    res_3d = []
    best_err = np.inf
    for point_key_combination in point_key_combinations:
        points_2d = [
            image_points[0][point_key_combination[0]],
            image_points[1][point_key_combination[1]],
            image_points[2][point_key_combination[2]],
        ]
        # logger.debug(f'Trying point combination: {points_2d}')
        res = minimize(
            f,
            x0=x0.copy(),
            args=(points_2d,),
            method='BFGS',
            options={
                'maxiter': 1000,
                'gtol': 1e-8,
                'disp': False,  # LOG_LEVEL == 'DEBUG',
            },
            tol=cameras.reprojection_error  # probably will never reach this
        )

        obj_point = ObjectPoint()
        obj_point.cameras = cameras
        obj_point.point_3d = list(res.x)
        obj_point.error = res.fun
        obj_point.source_point_idxs = point_key_combination
        res_3d.append(obj_point)
        best_err = min(best_err, obj_point.error)

    # Filter out any point combinations with too large of an error
    res_3d = [r for r in res_3d if r.error < matching_threshold]
    res_3d.sort(key=lambda r: r.error)

    # Raise exception if we can't find any points
    if len(res_3d) == 0:
        raise ValueError(f'Found zero 3D points below threshold, best: {best_err:.2f} > {matching_threshold}.')

    # Re-project the final 3d object points back to new 2d image points
    for r in res_3d:
        r.reprojected_points_2d = [
            list(cams_model[c].project_to_2d(r.point_3d))
            for c in CAMERA_IDXS
        ]
        # logger.debug(f'Found point: {r.point_3d}')
        # logger.debug(f'Error: {r.error}')
        # logger.debug(f'Source point idxs: {r.source_point_idxs}')
        # logger.debug(f'Reprojected points 2d: {r.reprojected_points_2d}')

    return res_3d
