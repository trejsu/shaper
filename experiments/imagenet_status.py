import argparse
import logging
import os
import time

TIME_PATH = '/tmp/imagenet-start-time'

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    images = os.listdir(ARGS.images_dir)
    num_images = len(images)
    drawings = os.listdir(ARGS.drawings_dir)
    num_drawings = len(drawings)
    with open(TIME_PATH, "r") as t:
        start_time = t.readlines()
    start_time = float(start_time[0])
    elapsed = time.time() - start_time
    completed = num_drawings * 100 / (num_images * ARGS.n)
    if completed < 100:
        log.info('Drawing in progress')
        log.info(f'{completed}% completed.')
        if num_drawings == 0:
            num_drawings += 1
        # todo: sth wrong with time left
        time_left = (num_images * ARGS.n * elapsed) / num_drawings
        log.info(f'Estimated time left: {format_time(time_left)}')
    else:
        log.info('Drawing completed.')


def format_time(t):
    hours = t / 3600
    if hours > 1:
        return str(hours) + ' hours'
    minutes = t / 60
    if minutes > 1:
        return str(minutes) + ' minutes'
    return str(t) + ' seconds'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-dir', type=str, help='Directory with input images', required=True)
    parser.add_argument('--drawings-dir', type=str, help='Directory to save drawings', required=True)
    parser.add_argument('--n', type=int, help='Number of shapes to draw', required=True)
    ARGS = parser.parse_args()
    log.info(ARGS)
    main()
