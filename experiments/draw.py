import argparse
import logging
import os
import time
from pathlib import Path

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DRAW_COMMANDS_PATH = os.path.join(str(Path.home()), 'draw-commands.txt')
DRAW_CMD_TEMPLATE = 'python {} --input {} --output {}-%d.jpg --n {} --resize {} --output-size {} --alpha {} {}\n'
TIME_PATH = os.path.join(str(Path.home()), 'imagenet-start-time')


def main():
    save_start_time()
    draw()


def draw():
    prepare_commands_for_drawing()
    parallel_cmd = f'parallel -j {ARGS.cpu} < {DRAW_COMMANDS_PATH}'
    log.info('Starting drawing...')
    os.system(parallel_cmd)
    log.info('Drawing completed')


def prepare_commands_for_drawing():
    shaper_main_path = os.path.join(CURRENT_DIR + '/..', 'main.py')
    log.info(f'shaper main path: {shaper_main_path}')

    imgs = os.listdir(ARGS.images_dir)
    log.info(f'found {len(imgs)} images to draw')

    if os.path.exists(DRAW_COMMANDS_PATH):
        os.remove(DRAW_COMMANDS_PATH)

    with open(DRAW_COMMANDS_PATH, "a") as commands:
        for img in imgs:
            inpt = os.path.join(ARGS.images_dir, img)
            output = os.path.join(ARGS.drawings_dir, img.split('.')[0])
            commands.write(
                DRAW_CMD_TEMPLATE.format(shaper_main_path, inpt, output, ARGS.n, ARGS.resize, ARGS.output_size,
                                         ARGS.alpha,
                                         '' if ARGS.background is None else f'--background {ARGS.background}'))


def save_start_time():
    if os.path.exists(TIME_PATH):
        os.remove(TIME_PATH)

    with open(TIME_PATH, "a") as t:
        t.write(str(time.time()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-dir', type=str, help='Directory with input images', required=True)
    parser.add_argument('--drawings-dir', type=str, help='Directory to save drawings', required=True)
    parser.add_argument('--n', type=int, help='Number of shapes to draw', required=True)
    parser.add_argument('--cpu', type=int, help='Number of CPUs to use', required=True)
    parser.add_argument('--resize', type=int, default=300)
    parser.add_argument('--output-size', type=int, default=300)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--background', type=str)
    ARGS = parser.parse_args()
    log.info(ARGS)
    main()
