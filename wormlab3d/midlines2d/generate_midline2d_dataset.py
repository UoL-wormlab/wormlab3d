from bson import ObjectId

from wormlab3d import logger
from wormlab3d.data.model import Midline2D
from wormlab3d.data.model.dataset import DatasetMidline2D
from wormlab3d.midlines2d.args import DatasetMidline2DArgs
from wormlab3d.nn.generate_dataset import build_pipeline, build_dataset


def generate_midline2d_dataset(args: DatasetMidline2DArgs, fix_frames: bool = False) -> DatasetMidline2D:
    """
    Generate a 2D midline dataset.
    """
    logger.info(
        f'Generating dataset: ---- \n' +
        '\n'.join(args.get_info()) +
        '\n-----------------------------------\n'
    )
    failed_midline_ids = []

    def validate_midline(midline_id: ObjectId) -> bool:
        midline = Midline2D.objects.no_cache().get(id=midline_id)
        frame = midline.frame
        logger.debug(f'Checking frame id={frame.id}')
        try:
            if fix_frames:
                ok = True
                if not frame.centres_2d_available():
                    logger.warning('Frame does not have 2d centre points available for all views, generating now.')
                    centres_2d, centres_2d_thresholds = frame.generate_centres_2d()
                    frame.update_centres_2d(centres_2d, centres_2d_thresholds)
                    if frame.centres_2d_available():
                        frame.save()
                    else:
                        ok = False
                    frame.reload()

                if ok and frame.centre_3d is None:
                    logger.warning('Frame does not have 3d centre point available, generating now.')
                    res = frame.generate_centre_3d(
                        error_threshold=50,
                        try_experiment_cams=True,
                        try_all_cams=False,
                        only_replace_if_better=True,
                        store_bad_result=True,
                        ratio_adj_orig=0.1,
                        ratio_adj_exp=0.1,
                    )
                    if res:
                        frame.save()
                    else:
                        ok = False
                    frame.reload()

                if ok and len(frame.images) != 3:
                    logger.warning(f'Frame does not have prepared images, generating now.')
                    frame.generate_prepared_images()
                    frame.save()
                    frame.reload()

            assert len(frame.images) == 3, 'Frame does not have prepared images.'
            assert midline.frame.centre_3d is not None, 'Frame does not have 3d centre point available.'
            assert len(midline.get_prepared_coordinates()) > 1, 'Midline coordinates empty after crop.'
            return True
        except AssertionError as e:
            failed_midline_ids.append(midline_id)
            if fix_frames:
                logger.error(f'Failed to prepare frame: {e}')
            else:
                logger.error(f'Frame is not ready: {e}')
        return False

    # Fetch manually-annotated midlines, grouped by tags (each might appear for multiple tags)
    # The query starts matching on the midline2d collection.
    logger.info('Querying database.')
    pipeline = [
        {'$match': {'user': {'$exists': True}}},
        *build_pipeline(args)
    ]
    cursor = Midline2D.objects().aggregate(pipeline)

    # Build the dataset
    DS = build_dataset(cursor, args, validate_midline)

    n_failed = len(failed_midline_ids)
    if n_failed > 0:
        logger.error(f'Failed to include {n_failed}/{DS.size_all + n_failed} matching midlines: {failed_midline_ids}.')

    # Save dataset
    logger.debug('Saving dataset.')
    DS.save()

    return DS
