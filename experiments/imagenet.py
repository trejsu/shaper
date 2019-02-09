import argparse
import logging
import os

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

COMMANDS_PATH = '/tmp/draw-commands.txt'
CSV_HEADERS = 'path,name,n,top1_class,top1_percent,top2_class,top2_percent,top3_class,top3_percent,top4_class,' \
              'top4_percent,top5_class,top5_percent'
DRAW_CMD_TEMPLATE = 'python {} --input {} --output {}-%d.jpg --n {} --resize {} --output-size {}\n'
DARKNET_CMD_TEMPLATE = 'printf \'{}\' | ./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg ' \
                       'darknet19.weights > /tmp/darknet-output.txt'


def main():
    num_images = prepare_commands_for_drawing()

    parallel_cmd = f'parallel -j {ARGS.cpu} < {COMMANDS_PATH}'
    log.info('Starting drawing...')
    os.system(parallel_cmd)
    log.info('Drawing completed')

    classify(num_images)

    if os.path.exists(ARGS.result_csv_path):
        old_results = ARGS.result_csv_path + '.old'
        log.warning(f'Found old csv with results, renaming to {old_results}')
        os.rename(ARGS.result_csv_path, old_results)

    with open(ARGS.result_csv_path, "a") as csv:
        csv.write(f'{CSV_HEADERS}\n')


def classify(num_images):
    drawings = os.listdir(ARGS.drawings_dir)
    log.info(f'Found {len(drawings)} drawings, should be {num_images * ARGS.n}')

    drawings_string = ''
    for drawing in drawings:
        drawings_string += os.path.join(ARGS.drawings_dir, drawing) + '\n'

    darknet_cmd = DARKNET_CMD_TEMPLATE.format(drawings_string)
    log.info(f'Command to classify images: {darknet_cmd}')

    log.info('Starting classification...')
    os.system(f'cd {ARGS.darknet_path} && {darknet_cmd} && cd {CURRENT_DIR}')
    log.info(f'Classification completed')


def prepare_commands_for_drawing():
    shaper_main_path = os.path.join(CURRENT_DIR + '/..', 'main.py')
    log.info(f'shaper main path: {shaper_main_path}')

    imgs = os.listdir(ARGS.images_dir)
    log.info(f'found {len(imgs)} images to draw')

    if os.path.exists(COMMANDS_PATH):
        os.remove(COMMANDS_PATH)

    with open(COMMANDS_PATH, "a") as commands:
        for img in imgs:
            inpt = os.path.join(ARGS.images_dir, img)
            output = os.path.join(ARGS.drawings_dir, img.split('.')[0])
            commands.write(
                DRAW_CMD_TEMPLATE.format(shaper_main_path, inpt, output, ARGS.n, ARGS.resize, ARGS.output_size))

    return len(imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-dir', type=str, help='Directory with input images', required=True)
    parser.add_argument('--drawings-dir', type=str, help='Directory to save drawings', required=True)
    parser.add_argument('--n', type=int, help='Number of shapes to draw', required=True)
    parser.add_argument('--cpu', type=int, help='Number of CPUs to use', required=True)
    parser.add_argument('--resize', type=int, default=300)
    parser.add_argument('--output-size', type=int, default=300)
    parser.add_argument('--result-csv-path', type=str, help='Output path to csv classification results', required=True)
    parser.add_argument('--darknet-path', type=str, help='Path to darknet classifier', required=True)
    ARGS = parser.parse_args()
    log.info(ARGS)
    main()
