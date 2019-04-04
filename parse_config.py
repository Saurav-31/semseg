import argparse

def parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-data", required=True, help="Root directory for Input data")
    parser.add_argument("-model", required=False, help="Model For pre-training")
    parser.add_argument("-max_epochs", default=5, type=int, help="Number of epochs for training")
    parser.add_argument("-batch_size", default=16, type=int, help="Batch Size for training")
    parser.add_argument("-gpu", required=True, type=int, help="Gpu to be used")
    parser.add_argument("-save_path", required=True, help="Path where to save the model")
    parser.add_argument("-lr", default=0.001, required=False, help="Learning Rate")
    parser.add_argument("-scale", default=False, required=True, help="Scale Image or Not (Boolean)")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument('-num_gpus', default=1, type=int, help="Number of gpu to be used")
    parser.add_argument('-val_split', default=0.1, type=float, help="Number of gpu to be used")
    parser.add_argument('-normalize', default=False, type=bool, help="Normalize the images and masks")
    args = vars(parser.parse_args())

    inp_im_size = {'resnet50': 224, 'resnet150': 224, 'vgg16': 224, 'inceptionv3': 299}
    args['imsize'] = inp_im_size[args['model']]
    args['batch_size'] = args['batch_size'] * args['num_gpus']
    return args
