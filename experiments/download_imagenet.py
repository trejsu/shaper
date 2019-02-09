import argparse
import logging
import os

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    log.info('Choosing random images...')
    os.system(f'shuf -n {ARGS.n} {ARGS.urls_file} > /tmp/samples.txt')
    os.system('cat /tmp/samples.txt | awk \'{ print $2 }\' > /tmp/urls.txt')
    log.info('Downloading...')
    os.system(f'wget -i /tmp/urls.txt -P {ARGS.download_dir} --tries=1 --timeout=3')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, help='Directory to download images', required=True)
    parser.add_argument('--urls-file', type=str, help='File with imagenet urls', required=True)
    parser.add_argument('--n', type=int, help='Number of images to download', required=True)
    ARGS = parser.parse_args()
    log.info(ARGS)
    main()
