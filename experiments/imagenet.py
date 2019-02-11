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
CLASSIFY_COMMANDS_PATH = os.path.join(str(Path.home()), 'classify-commands.txt')
CSV_HEADERS = 'path,name,n,top1_class,top1_percent,top2_class,top2_percent,top3_class,top3_percent,top4_class,' \
              'top4_percent,top5_class,top5_percent'
DRAW_CMD_TEMPLATE = 'python {} --input {} --output {}-%d.jpg --n {} --resize {} --output-size {}\n'
DARKNET_CMD_TEMPLATE = 'printf \'{}\' | ./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg ' \
                       'darknet19.weights | sed \'s/Enter Image Path: //\' > {}'
DARKNET_OUTPUT_DRAWINGS_PATH = os.path.join(str(Path.home()), 'darknet-output-drawings-%d.txt')
DARKNET_OUTPUT_ORIGINALS_PATH = os.path.join(str(Path.home()), 'darknet-output-originals-%d.txt')
TIME_PATH = os.path.join(str(Path.home()), 'imagenet-start-time')


def main():
    remove_old_data()
    save_start_time()
    num_images = draw()
    save_classification_start_time()
    drawings_chunks = classify_images(
        img_dir=ARGS.drawings_dir,
        num=num_images * ARGS.n,
        output_file=DARKNET_OUTPUT_DRAWINGS_PATH
    )
    write_drawings_classification_results_to_csv(drawings_chunks)
    originals_chunks = classify_images(
        img_dir=ARGS.images_dir,
        num=num_images,
        output_file=DARKNET_OUTPUT_ORIGINALS_PATH
    )
    write_originals_classification_results_to_csv(originals_chunks)
    log.info(f'Results saved under {ARGS.result_csv_path}')


def classify_images(img_dir, num, output_file):
    images = os.listdir(img_dir)
    num_images = len(images)
    log.info(f'Found {num_images} images, should be {num}')
    chunk_len = 100 if num_images > 100 * ARGS.cpu else num_images // ARGS.cpu + 1
    log.info(f'Chunk length =  {chunk_len}.')
    images_chunks = [images[i:i + chunk_len] for i in range(0, num_images, chunk_len)]
    log.info(f'Divided images to classify into {len(images_chunks)} parts.')
    classify(chunks=images_chunks, images_dir=img_dir, output_file=output_file)
    return images_chunks


def draw():
    num_images = prepare_commands_for_drawing()
    parallel_cmd = f'parallel -j {ARGS.cpu} < {DRAW_COMMANDS_PATH}'
    log.info('Starting drawing...')
    os.system(parallel_cmd)
    log.info('Drawing completed')
    return num_images


def write_drawings_classification_results_to_csv(chunks):
    if os.path.exists(ARGS.result_csv_path):
        old_results = ARGS.result_csv_path + '.old'
        log.warning(f'Found old csv with results, renaming to {old_results}')
        os.rename(ARGS.result_csv_path, old_results)

    with open(ARGS.result_csv_path, "a") as csv:
        csv.write(f'{CSV_HEADERS}\n')
        write_drawings_results(csv, chunks)


def write_drawings_results(csv, chunks):
    top1_cls, top1_perc, top2_cls, top2_perc, top3_cls, top3_perc, top4_cls, top4_perc, top5_cls, top5_perc = \
        extract_results(output_file=DARKNET_OUTPUT_DRAWINGS_PATH, num_chunks=len(chunks))
    for chunk_index, chunk in enumerate(chunks):
        for img_index in range(len(chunk)):
            drawing = chunk[img_index]
            path = os.path.join(ARGS.drawings_dir, drawing)
            name = drawing.split('-')[0]
            n = drawing.split('-')[1].split('.')[0]
            csv_line = ','.join([path, name, n, top1_cls[chunk_index][img_index], top1_perc[chunk_index][img_index],
                                 top2_cls[chunk_index][img_index], top2_perc[chunk_index][img_index],
                                 top3_cls[chunk_index][img_index], top3_perc[chunk_index][img_index],
                                 top4_cls[chunk_index][img_index], top4_perc[chunk_index][img_index],
                                 top5_cls[chunk_index][img_index], top5_perc[chunk_index][img_index]]) + '\n'
            csv.write(csv_line)


def write_originals_classification_results_to_csv(chunks):
    top1_cls, top1_perc, top2_cls, top2_perc, top3_cls, top3_perc, top4_cls, top4_perc, top5_cls, top5_perc = \
        extract_results(output_file=DARKNET_OUTPUT_ORIGINALS_PATH, num_chunks=len(chunks))

    with open(ARGS.result_csv_path, "a") as csv:
        for chunk_index, chunk in enumerate(chunks):
            for img_index in range(len(chunk)):
                img = chunk[img_index]
                path = os.path.join(ARGS.images_dir, img)
                name = img.split('.')[0]
                n = '0'
                csv_line = ','.join([path, name, n, top1_cls[chunk_index][img_index],
                                     top1_perc[chunk_index][img_index], top2_cls[chunk_index][img_index],
                                     top2_perc[chunk_index][img_index], top3_cls[chunk_index][img_index],
                                     top3_perc[chunk_index][img_index], top4_cls[chunk_index][img_index],
                                     top4_perc[chunk_index][img_index], top5_cls[chunk_index][img_index],
                                     top5_perc[chunk_index][img_index]]) + '\n'
                csv.write(csv_line)


def extract_results(output_file, num_chunks):
    results = [[], [], [], [], [], [], [], [], [], []]
    for i in range(num_chunks):
        chunk_results = extract_results_for_one_chunk(output_file % i)
        for j in range(10):
            results[j].append(chunk_results[j])
    return results


def extract_results_for_one_chunk(output_file):
    log.info(f'Extracting results for {output_file}')
    with open(output_file, "r") as darknet_output:
        darknet = darknet_output.readlines()
    percentages = [line.split('%')[0].strip() for line in darknet]
    classes = [line.split(': ')[1][:-1] for line in darknet]
    assert len(percentages) % 5 == 0
    assert len(classes) % 5 == 0
    return classes[0::5], percentages[0::5], classes[1::5], percentages[1::5], classes[2::5], percentages[2::5], \
           classes[3::5], percentages[3::5], classes[4::5], percentages[4::5]


def classify(chunks, images_dir, output_file):
    if os.path.exists(CLASSIFY_COMMANDS_PATH):
        os.remove(CLASSIFY_COMMANDS_PATH)

    with open(CLASSIFY_COMMANDS_PATH, "a") as commands:
        for i, chunk in enumerate(chunks):
            images_string = ''
            for img in chunk:
                images_string += os.path.join(images_dir, img) + '\\n'
            cmd = DARKNET_CMD_TEMPLATE.format(images_string, output_file % i)
            commands.write(f'cd {ARGS.darknet_path} && {cmd} && cd {CURRENT_DIR}\n')

    parallel_cmd = f'parallel -j {ARGS.cpu} < {CLASSIFY_COMMANDS_PATH}'
    log.info('Starting classification...')
    os.system(parallel_cmd)
    log.info('Classification completed')


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
                DRAW_CMD_TEMPLATE.format(shaper_main_path, inpt, output, ARGS.n, ARGS.resize, ARGS.output_size))

    return len(imgs)


def save_start_time():
    if os.path.exists(TIME_PATH):
        os.remove(TIME_PATH)

    with open(TIME_PATH, "a") as t:
        t.write(str(time.time()) + '\n')


def save_classification_start_time():
    with open(TIME_PATH, "a") as t:
        t.write(str(time.time()))


def remove_old_data():
    if os.path.exists(DARKNET_OUTPUT_DRAWINGS_PATH):
        os.remove(DARKNET_OUTPUT_DRAWINGS_PATH)
    if os.path.exists(DARKNET_OUTPUT_ORIGINALS_PATH):
        os.remove(DARKNET_OUTPUT_ORIGINALS_PATH)


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
