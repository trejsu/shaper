import argparse
import logging
import os
from pathlib import Path

import pandas as pd

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFY_COMMANDS_PATH = os.path.join(str(Path.home()), 'classify-commands.txt')
CSV_HEADER = 'img,true_class,top1_class,top1_prob,top2_class,top2_prob,top3_class,top3_prob,top4_class,' \
             'top4_prob,top5_class,top5_prob,top1,top2,top3,top4,top5\n'
DARKNET_CMD_TEMPLATE = 'printf \'{}\' | ./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg ' \
                       'darknet19.weights | sed \'s/Enter Image Path: //\' > {}'
DARKNET_OUTPUT_PATH = os.path.join(str(Path.home()), 'darknet-output-%d.txt')


def main():
    remove_old_data()
    chunks = classify_images(
        img_dir=ARGS.images_dir,
        output_file=DARKNET_OUTPUT_PATH
    )
    write_classification_results_to_csv(chunks)
    log.info(f'Results saved under {ARGS.result_csv_path}')


def classify_images(img_dir, output_file):
    images = os.listdir(img_dir)
    num_images = len(images)
    log.info(f'Found {num_images} images')
    chunk_len = 100 if num_images > 100 * ARGS.cpu else num_images // ARGS.cpu + 1
    log.info(f'Chunk length = {chunk_len}.')
    images_chunks = [images[i:i + chunk_len] for i in range(0, num_images, chunk_len)]
    log.info(f'Divided images to classify into {len(images_chunks)} parts.')
    classify(chunks=images_chunks, images_dir=img_dir, output_file=output_file)
    return images_chunks


def score_predictions(name, top_cls, img_to_cls):
    true_cls = img_to_cls[name]
    return true_cls, str(int(true_cls in top_cls[:1])), str(int(true_cls in top_cls[:2])), \
           str(int(true_cls in top_cls[:3])), str(int(true_cls in top_cls[:4])), str(int(true_cls in top_cls[:5]))


def write_classification_results_to_csv(chunks):
    if os.path.exists(ARGS.result_csv_path):
        old_results = ARGS.result_csv_path + '.old'
        log.warning(f'Found old csv with results, renaming to {old_results}')
        os.rename(ARGS.result_csv_path, old_results)

    top1_cls, top1_prob, top2_cls, top2_prob, top3_cls, top3_prob, top4_cls, top4_prob, top5_cls, top5_prob = \
        extract_results(output_file=DARKNET_OUTPUT_PATH, num_chunks=len(chunks))

    df = pd.read_csv(ARGS.img_cls_mapping)
    log.info(f'Loaded img to class mapping data frame with {len(df.index)} rows.')
    img_to_cls = dict(zip(df['img'], df['class']))

    with open(ARGS.result_csv_path, "a") as csv:
        csv.write(CSV_HEADER)
        for i, chunk in enumerate(chunks):
            for j in range(len(chunk)):
                img = chunk[j]
                name = img.split('.')[0]
                true_cls, top1, top2, top3, top4, top5 = score_predictions(name, [top1_cls[i][j], top2_cls[i][j],
                                                                                  top3_cls[i][j], top4_cls[i][j],
                                                                                  top5_cls[i][j]], img_to_cls)

                csv_line = ','.join([name, true_cls, top1_cls[i][j], top1_prob[i][j], top2_cls[i][j], top2_prob[i][j],
                                     top3_cls[i][j], top3_prob[i][j], top4_cls[i][j], top4_prob[i][j], top5_cls[i][j],
                                     top5_prob[i][j], top1, top2, top3, top4, top5]) + '\n'
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
    probs = [line.split('%')[0].strip() for line in darknet]
    classes = [line.split(': ')[1][:-1] for line in darknet]
    assert len(probs) % 5 == 0
    assert len(classes) % 5 == 0
    return classes[0::5], probs[0::5], classes[1::5], probs[1::5], classes[2::5], probs[2::5], \
           classes[3::5], probs[3::5], classes[4::5], probs[4::5]


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


def remove_old_data():
    if os.path.exists(DARKNET_OUTPUT_PATH):
        os.remove(DARKNET_OUTPUT_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-dir', type=str, help='Directory with input images', required=True)
    parser.add_argument('--cpu', type=int, help='Number of CPUs to use', required=True)
    parser.add_argument('--result-csv-path', type=str, help='Output path to csv classification results', required=True)
    parser.add_argument('--darknet-path', type=str, help='Path to darknet classifier', required=True)
    parser.add_argument('--img-cls-mapping', type=str, help='Path to mapping between image names and labels',
                        required=True)
    ARGS = parser.parse_args()
    log.info(ARGS)
    main()
