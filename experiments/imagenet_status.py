import argparse
import logging
import os
import time
from pathlib import Path

TIME_PATH = os.path.join(str(Path.home()), 'imagenet-start-time')

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    num_images = len(os.listdir(ARGS.images_dir))
    num_drawings = len(os.listdir(ARGS.drawings_dir))
    total_num_drawings = num_images * ARGS.n
    completed = num_drawings * 100 / total_num_drawings
    if completed < 100:
        drawing_status(completed, num_drawings, total_num_drawings)
    else:
        classification_status(num_images, total_num_drawings)


def classification_status(num_images, total_num_drawings):
    log.info('Drawing completed.')
    log.info('Classification in progress...')
    with open(TIME_PATH, "r") as t:
        start_time = t.readlines()
    classification_start_time = float(start_time[1])
    elapsed = time.time() - classification_start_time
    estimated = ((num_images + total_num_drawings) * 1.1) / ARGS.cpu
    completed = (elapsed / estimated) * 100
    log.info(f'About {completed:.2f}% completed.')
    log.info(f'Time elapsed: {format_time(elapsed)}')
    log.info(f'Estimated time left: {format_time(estimated - elapsed)}')


def drawing_status(completed, num_drawings, total_num_drawings):
    log.info('Drawing in progress')
    log.info(f'{num_drawings}/{total_num_drawings} ({completed}%) completed.')

    with open(TIME_PATH, "r") as t:
        start_time = t.readlines()

    drawing_start_time = float(start_time[0])
    elapsed = time.time() - drawing_start_time
    if num_drawings == 0:
        num_drawings += 1
    estimated = (total_num_drawings * elapsed) / num_drawings
    log.info(f'Time elapsed: {format_time(elapsed)}')
    log.info(f'Estimated time left: {format_time(estimated - elapsed)}')


def format_time(t):
    hours = t / 3600
    if hours > 1:
        return str(int(hours)) + ' hours ' + format_time(t % 3600)
    minutes = t / 60
    if minutes > 1:
        return str(int(minutes)) + ' minutes ' + format_time(t % 60)
    return str(int(t)) + ' seconds'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-dir', type=str, help='Directory with input images', required=True)
    parser.add_argument('--drawings-dir', type=str, help='Directory to save drawings', required=True)
    parser.add_argument('--n', type=int, help='Number of shapes to draw', required=True)
    parser.add_argument('--cpu', type=int, help='Number of CPUs to use', required=True)
    ARGS = parser.parse_args()
    log.info(ARGS)
    main()
