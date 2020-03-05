import glob
import os
import shutil


def main():
    if os.path.exists('docs/images'):
        shutil.rmtree('docs/images')
    if os.path.exists('docs/modules.rst'):
        os.remove('docs/modules.rst')
    for f in glob.glob('docs/hic3defdr*.rst'):
        os.remove(f)


if __name__ == '__main__':
    main()
