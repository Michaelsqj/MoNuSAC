import shutil
import os


def movefile(src, dst, number):
    filenames = sorted(os.listdir(os.path.join(src, 'imgs')))
    filenames = filenames[:number]
    for file in filenames:
        shutil.move(os.path.join(src, 'imgs', file), os.path.join(dst, 'imgs', file))
        shutil.move(os.path.join(src, 'masks', file), os.path.join(dst, 'masks', file))


if __name__ == '__main__':
    dst = r'E:\MoNUSAC\MyModel\dataset\val'
    src = r'E:\MoNUSAC\MyModel\dataset\train'
    movefile(src, dst, 50)
