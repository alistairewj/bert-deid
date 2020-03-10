"""
Compare two sets of gold standard annotations.
Allows selection of one of the two as the correct annotation,
which is then stored in the output folder.

Alternatively, the review option can be selected, and both
annotations are output to a review folder.
"""

from pathlib import Path

import os
import sys
import csv
import argparse

parser = argparse.ArgumentParser(description='Convert ann files to gs files')
parser.add_argument('-p', '--path', type=str,
                    default='/data/deid-gs/rr-set-to-fix/ann',
                    help='path containing files to compare')
parser.add_argument('-c', '--compare', type=str,
                    default='/db/git/rr-set-to-fix/deid',
                    help='path containing files to compare')
parser.add_argument('-t', '--text', type=str,
                    default='/db/git/rr-set-to-fix/txt',
                    help='path with text files')
parser.add_argument('-o', '--output', type=str,
                    default='/db/git/rr-set-to-fix/fixed',
                    help='output path')
parser.add_argument('-r', '--review', type=str,
                    default='/db/git/rr-set-to-fix/review',
                    help='review path for outputting uncertain anns')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_text_files(folder, extension):
    files = os.listdir(folder)
    if extension[0] != '.':
        extension = '.' + extension
    files = [f[0:-len(extension)] for f in files if f.endswith(extension)]
    return files


def main():
    args = parser.parse_args()

    base_path = Path(args.base_path)
    extension = 'gs'
    files = load_text_files(base_path, extension)

    cmp_path = Path(args.compare)
    cmp_ext = 'phi'
    cmp_files = load_text_files(cmp_path, cmp_ext)
    files = [f for f in files if f in set(cmp_files)]

    out_path = Path(args.output)
    out_ext = 'gs'

    txt_path = Path(args.text)
    review_path = Path(args.review)

    def get_record(i):
        with open(base_path / '{}.{}'.format(files[i], extension), 'r') as fp:
            text_top = []
            reader = csv.reader(fp)
            for row in reader:
                text_top.append(row)

        with open(cmp_path / '{}.{}'.format(files[i], cmp_ext), 'r') as fp:
            text_bot = []
            reader = csv.reader(fp)
            for row in reader:
                text_bot.append(row)

        with open(txt_path / '{}.{}'.format(files[i], 'txt'), 'r') as fp:
            text = ''.join(fp.readlines())

        return text, text_top, text_bot

    if len(files) == 0:
        return
    files.sort()

    # initialize with first record
    i = 0
    text, text_top, text_bot = get_record(i)

    while i < len(files):
        os.system('clear')

        # display top report
        print('top:')
        for l in text_top[1:]:
            print(','.join(l).replace('\n', ' '), end='\t')
            start, stop = int(l[2]), int(l[3])
            begin, end = max(start-30, 0), min(stop+30, len(text))
            # print with color highlight
            print(text[begin:start].replace('\n', ' '), end='')
            print(bcolors.WARNING, end='')  # red
            print(text[start:stop].replace('\n', ' '), end='')
            print(bcolors.ENDC, end='')  # end color
            print(text[stop:end].replace('\n', ' '), end='\n')
        print('')
        print('\n')

        # display bottom with context
        print('bot:')
        for l in text_bot[1:]:
            print(','.join(l).replace('\n', ' '), end='\t')
            start, stop = int(l[2]), int(l[3])
            begin, end = max(start-30, 0), min(stop+30, len(text))
            # print with color highlight
            print(text[begin:start].replace('\n', ' '), end='')
            print(bcolors.FAIL, end='')  # red
            print(text[start:stop].replace('\n', ' '), end='')
            print(bcolors.ENDC, end='')  # end color
            print(text[stop:end].replace('\n', ' '), end='\n')
        print('')

        sys.stdout.flush()

        # capture input character
        c = input('(t)op, (b)ottom, (s)ave, (u)ndo, (q)uit: ')

        out_fn = '{}.{}'.format(files[i], out_ext)

        if c.upper() == 'Q':
            break

        elif c.upper() == 'U':
            # go back to previous report
            pass

        elif c.upper() == 'T':
            # retain the top annotations
            with open(out_path / out_fn, 'w') as fp:
                writer = csv.writer(fp)
                for row in text_bot:
                    writer.writerow(row)

        elif c.upper() == 'B':
            # retain the bottom annotations
            with open(out_path / out_fn, 'w') as fp:
                writer = csv.writer(fp)
                for row in text_bot:
                    writer.writerow(row)

        elif c.upper() == 'S':
            # write to another folder for later review
            with open(review_path / f'{out_fn}.a', 'w') as fp:
                writer = csv.writer(fp)
                for row in text_top:
                    writer.writerow(row)
            with open(review_path / f'{out_fn}.b', 'w') as fp:
                writer = csv.writer(fp)
                for row in text_bot:
                    writer.writerow(row)

        i += 1
        text, text_top, text_bot = get_record(i)


if '__main__' == __name__:
    main()
