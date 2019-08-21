from __future__ import print_function
import sys, os
import argparse
import subprocess
import mxnet
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from pascal_voc import PascalVoc

def load_pascal(image_set, devkit_path, shuffle=False, class_names=None):
    """
    wrapper function for loading pascal voc dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """
    assert image_set, "No image_set specified"

    return PascalVoc(image_set, devkit_path, shuffle, is_train=True, class_names=class_names)

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='pascal', type=str)
    parser.add_argument('--set', dest='set', help='train, val, trainval, test',
                        default='trainval', type=str)
    parser.add_argument('--target', dest='target', help='output list file',
                        default=None,
                        type=str)
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default=None, help='string of comma separated names, or text filename')
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=os.path.join(curr_path, '..', 'data', 'VOCdevkit'),
                        type=str)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list',
                        type=bool, default=True)
    parser.add_argument('--annotations', dest='annotations_file', help='annotations file',
                        type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.class_names is not None:
        assert args.target is not None, 'for a subset of classes, specify a target path. Its for your own safety'
    if args.dataset == 'pascal':
        db = load_pascal(args.set, args.root_path, args.shuffle, args.class_names)
        print("saving list to disk...")
        db.save_imglist(args.target, root=args.root_path)
    else:
        raise NotImplementedError("No implementation for dataset: " + args.dataset)

    print("List file {} generated...".format(args.target))

    im2rec_path = os.path.join(mxnet.__path__[0], 'tools/im2rec.py')

    # final validation - sometimes __path__ (or __file__) gives 'mxnet/python/mxnet' instead of 'mxnet'
    if not os.path.exists(im2rec_path):
        im2rec_path = os.path.join(os.path.dirname(os.path.dirname(mxnet.__path__[0])), 'tools/im2rec.py')
    subprocess.check_call(["python", im2rec_path,
        os.path.abspath(args.target), os.path.abspath(args.root_path),
        "--pack-label", "--num-thread", "12"])

    print("Record file {} generated...".format(args.target.split('.')[0] + '.rec'))
