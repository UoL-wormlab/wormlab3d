import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from wormlab3d import logger, CAMERA_IDXS, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model.trial import Trial
from wormlab3d.preprocessing.cropper import crop_image
from wormlab3d.toolkit.util import parse_target_arguments

show_plots = False
save_plots = True
invert = True
use_uncompressed_videos = True


def plot_crops():
    """
    Plots a triplet of frames in:
        1) Original - with 2d centre points indicated,
        2) Cropped images around centre points.
        3) Inverted and background-subtracted, full size.
        4) Inverted, background-subtracted and cropped.
        5) Prepared images as loaded from filesystem (if available).
    """
    args = parse_target_arguments()
    assert args.trial is not None, '--trial must be defined!'
    assert args.frame_num is not None, '--frame-num must be defined!'

    # Fetch the trial, the video readers and the cameras
    trial = Trial.objects.get(id=args.trial)
    reader = trial.get_video_triplet_reader(use_uncompressed_videos=use_uncompressed_videos)
    frame = trial.get_frame(args.frame_num)
    crop_size = (trial.crop_size, trial.crop_size)

    # Set the frame number, fetch the images from each video and find objects in all 3
    reader.set_frame_num(args.frame_num)
    images = reader.get_images()
    images_inv_no_bg = reader.get_images(invert=True, subtract_background=True)

    # Get the centre points from the frame if available
    if frame.centre_3d is not None:
        logger.info('Frame has 3d centre precomputed, using the reprojections.')
        centres_2d = frame.centre_3d.reprojected_points_2d
        centres_2d = np.array(centres_2d)
        centres_2d = centres_2d[:, np.newaxis, :]
    elif len(frame.centres_2d):
        logger.info('Frame has 2d centres precomputed, using these.')
        centres_2d = frame.centres_2d
    else:
        logger.info('Frame has no 2d centres precomputed, computing new ones.')
        centres_2d, thresholds = reader.find_objects()

    # Check for prepared images
    has_prepared_images = False
    if len(frame.images) == 3:
        logger.info('Frame has prepared images ready.')
        has_prepared_images = True

    # Plot the results
    fig, axes = plt.subplots(4 + int(has_prepared_images), 3)
    fig.suptitle(
        f'{trial.date:%Y%m%d} Trial #{trial.trial_num}. '
        f'Frame #{args.frame_num}.'
    )
    for c in CAMERA_IDXS:
        if len(centres_2d[c]) > 0:
            centre_pts = np.stack(centres_2d[c])
        else:
            centre_pts = np.zeros((0, 2))

        # Original images
        ax = axes[0, c]
        ax.set_title(trial.videos[c])
        ax.imshow(images[c], vmin=0, vmax=255, cmap='gray')
        ax.scatter(x=centre_pts[:, 0], y=centre_pts[:, 1], color='blue', s=10, alpha=0.8)

        # Cropped images
        ax = axes[1, c]
        if c == 1:
            ax.set_title(f'Cropped to ({crop_size})')
        if len(centres_2d[c]) > 0:
            crop = crop_image(
                images[c],
                centre_2d=centres_2d[c][0],
                size=crop_size,
                fix_overlaps=True
            )
            ax.imshow(crop, vmin=0, vmax=255, cmap='gray')

        # Inverted and background-subtracted images
        ax = axes[2, c]
        if c == 1:
            ax.set_title('Inverted, background-subtracted')
        ax.imshow(images_inv_no_bg[c], vmin=0, vmax=255, cmap='gray')
        ax.scatter(x=centre_pts[:, 0], y=centre_pts[:, 1], color='yellow', s=10, alpha=0.8)

        # Cropped final images
        ax = axes[3, c]
        if c == 1:
            ax.set_title(f'Cropped to ({crop_size})')
        if len(centres_2d[c]) > 0:
            crop_inv_no_bg = crop_image(
                images_inv_no_bg[c],
                centre_2d=centres_2d[c][0],
                size=crop_size,
                fix_overlaps=True
            )
            ax.imshow(crop_inv_no_bg, vmin=0, vmax=255, cmap='gray')

        if has_prepared_images:
            # Prepared images
            ax = axes[4, c]
            if c == 1:
                ax.set_title(f'Prepared image')
            ax.imshow(frame.images[c], vmin=0, vmax=1, cmap='gray')

    plt.show()


def plot_crop_singles(args):
    """
    Plots the prepared crop single images from each camera.
    """
    # args = parse_target_arguments()

    # Fetch the trial, the video readers and the cameras
    trial = Trial.objects.get(id=args.trial)
    frame = trial.get_frame(args.frame_num)

    for c in CAMERA_IDXS:
        img = frame.images[c]
        if invert:
            img = 1 - img
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Convert to PIL image
        img = Image.fromarray(img, 'RGB')

        if save_plots:
            save_path = LOGS_PATH / f'{START_TIMESTAMP}' \
                                    f'_trial={trial.id}' \
                                    f'_frame={frame.frame_num}' \
                                    f'_c={c}.png'
            logger.info(f'Saving image to {save_path}.')
            img.save(save_path)

        if show_plots:
            img.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from wormlab3d.toolkit.plot_utils import interactive_plots
    # interactive_plots()
    # plot_crops()

    args = parse_target_arguments()
    for frame_num in range(780, 820):
        args.frame_num = frame_num
        plot_crop_singles(args)
