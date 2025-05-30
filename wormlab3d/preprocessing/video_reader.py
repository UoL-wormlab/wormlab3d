from subprocess import CalledProcessError
from typing import List, Tuple

import cv2
import numpy as np
import pims

from wormlab3d import logger
from wormlab3d.data.annex import fetch_from_annex, is_annexed_file
from wormlab3d.preprocessing.contour import CONT_THRESH_RATIO_DEFAULT, contour_mask, find_contours, \
    MAX_CONTOURING_ATTEMPTS, contour_centre, MAX_CONTOURS_ALLOWED, MIN_REQ_THRESHOLD
from wormlab3d.preprocessing.create_bg_lp import Accumulate


class VideoReader:
    """
    Video reader class. This can read either any format supported by opencv or pims.
    Works as an iterator over video frames or can be indexed directly.
    Provides methods to:
        1. Extract contours from frames.
        2. Find centre points of any detected objects.
        3. Generate a background image from the video using a low pass temporal filter.
    """

    def __init__(
            self,
            video_path: str,
            background_image_path: str = None,
            contour_thresh_ratio: float = CONT_THRESH_RATIO_DEFAULT
    ):
        # If the video is a link try and fetch it from the annex
        if is_annexed_file(video_path):
            fetch_from_annex(video_path)

        try:
            # standard video reader
            self.video = pims.PyAVReaderTimed(video_path)
        except Exception as e:
            logger.error(f'{type(e)}, {e}')
            # TODO add specialist for seq files once we have test data
            # generic video reader
            self.video = pims.open(video_path)

        # Open background image
        if background_image_path is not None:
            if is_annexed_file(background_image_path):
                try:
                    fetch_from_annex(background_image_path)
                except CalledProcessError as e:
                    logger.error(f'Could not fetch from annex: {e}')
            self.background = cv2.imread(background_image_path, cv2.IMREAD_GRAYSCALE)
            if self.background is None:
                logger.error(f'Cannot open background image: {background_image_path}')
        else:
            self.background = None

        self.current_frame: int = -1
        self.contour_thresh_ratio = contour_thresh_ratio
        logger.debug(f'VideoReader(video={video_path}, bg={background_image_path})')

    @property
    def fps(self):
        """Frames per second"""
        return self.video.frame_rate

    @property
    def frame_size(self):
        shape = self.video.frame_shape
        if len(shape) == 3:
            shape = shape[:-1]
        return shape

    def __iter__(self):
        return self

    def __next__(self) -> pims.Frame:
        try:
            next_frame = self.current_frame + 1
            img = self[next_frame]
            self.current_frame = next_frame
        except IndexError:
            raise StopIteration()
        return img

    def __getitem__(self, idx):
        img = self.video[idx]
        grey = self._as_grey(img)
        return grey

    def __len__(self):
        return len(self.video)

    def set_frame_num(self, idx: int):
        assert 0 <= idx <= len(self), 'Target frame number out of range for video stream!'
        self.current_frame = idx

    @staticmethod
    @pims.pipeline
    def _as_grey(frame: pims.Frame) -> pims.Frame:
        dtype = frame.dtype
        frame = pims.as_grey(frame)
        return frame.astype(dtype)

    @staticmethod
    @pims.pipeline
    def _invert(frame: pims.Frame) -> pims.Frame:
        frame = frame.copy()
        frame.data = np.invert(frame.data)
        return frame

    def get_image(self, invert: bool = False, subtract_background: bool = False) -> pims.Frame:
        """
        Fetch the image from the current video frame and optionally invert it and subtract the background.
        """
        image = self[self.current_frame].copy()

        # Invert image (white worms on black background)
        if invert:
            image = self._invert(image)

        # Subtract background
        if subtract_background:
            assert self.background is not None, 'No background image available to subtract.'
            assert self.background.shape == image.shape, 'Image size from video does not match background image size!'
            if invert:
                bg_inv = self._invert(self.background)
            image = cv2.subtract(image.copy(), bg_inv)

        return image

    def find_contours(self, subtract_background: bool = True, cont_threshold_ratio: float = None) \
            -> Tuple[List[np.ndarray], int]:
        """
        Find the contours in the image.
        Note - if the background is not subtracted this doesn't work very well.
        """
        image = self.get_image(invert=True, subtract_background=subtract_background)

        # Get max brightness
        max_brightness = image.max()

        # Find the contours
        contours = []
        if cont_threshold_ratio is None:
            cont_threshold_ratio = self.contour_thresh_ratio

        attempts = 0
        while len(contours) == 0 or len(contours) > MAX_CONTOURS_ALLOWED:
            threshold = max(MIN_REQ_THRESHOLD, int(max_brightness * cont_threshold_ratio))
            contours, mask = contour_mask(
                image,
                thresh=threshold,
                maxval=max_brightness
            )
            if len(contours):
                mask_dil = cv2.dilate(mask, None, iterations=5)
                contours = find_contours(
                    image=mask_dil,
                    max_area=np.inf
                )

            # If no contours found, decrease the threshold and try again
            if len(contours) == 0:
                cont_threshold_ratio -= 0.05

                # Allow one less-than-minimum threshold to be processed so we save to the database and can avoid repeating it.
                if threshold <= MIN_REQ_THRESHOLD:
                    logger.warning(f'Threshold ({threshold}) now too low. Min required = {MIN_REQ_THRESHOLD}.')
                    break

            # If too many contours found, increase the threshold and try again
            if len(contours) > MAX_CONTOURS_ALLOWED:
                cont_threshold_ratio += 0.05

            # Bail if too many attempts
            attempts += 1
            if attempts > MAX_CONTOURING_ATTEMPTS:
                logger.warning(
                    f'Could not find any contours in image! '
                    f'Max attempts exceeded ({MAX_CONTOURING_ATTEMPTS}).'
                )
                break

        return contours, threshold

    def find_objects(self, cont_threshold_ratio: float = None) -> Tuple[np.ndarray, float]:
        """
        Finds contours in the current frame and returns all centre point coordinates plus
        the final threshold value used to find them (it is automatically adjusted as needed).
        """
        contours, final_threshold = self.find_contours(
            subtract_background=True,
            cont_threshold_ratio=cont_threshold_ratio
        )
        centres = []
        for c in contours:
            centres.append(contour_centre(c))
        return centres, final_threshold

    def get_background(self) -> np.ndarray:
        """
        Create a background image by low-pass filtering the video.
        todo: redo the low-pass filter so it is more understandable
        """
        logger.info('Generating background')
        # Create the filter
        a = Accumulate(self.frame_size)
        self.current_frame = -1
        for image in self:
            a.push(image)
        bg = a.get()
        return bg

    def close(self):
        self.video.close()
