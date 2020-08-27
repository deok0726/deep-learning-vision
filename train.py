import argparse
import glob
# import pdb; pdb.set_trace();
from model import ModelFactory
from solver import Solver
from loss import get_loss_fn
from SR_datasets import *

description='Video Super Resolution pytorch implementation'

parser = argparse.ArgumentParser(description=description)

parser.add_argument('-m', '--model', metavar='M', type=str, default='TDAN',
                    help='network architecture. Default TDAN')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=4,
                    help='interpolation scale. Default 3')
parser.add_argument('--train-set', metavar='T', type=str, default='/hd/youtube_dl_face/cropped_videos_frames/resize_small2/',
                    help='data set for training. Default train')
parser.add_argument('--test-set', metavar='NAME', type=str, default='/hd/youtube_dl_face/cropped_videos_frames/youtube_face_valid_small_nopaddingneeded/',
                    help='dataset for testing. Default Temple from SPMC  dataset')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=1,
                    help='batch size used for training. Default 100')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float, default=1e-4,
                    help='learning rate used for training. Default 1e-3')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=600,
                    help='number of training epochs. Default 600')
parser.add_argument('-f', '--fine-tune', dest='fine_tune', action='store_true',
                    help='fine tune the model under check_point dir,\
                    instead of training from scratch. Default False')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help='print training information. Default False')
parser.add_argument('-g', '--gpu', metavar='G', type=str, default='1',
                    help='GPU numbers')
parser.add_argument('-cp', '--checkpoint', metavar='CP', type=str, default='/hd/checkpoints/tdan',
                    help='network architecture. Default False')
args = parser.parse_args()

def get_full_path(scale, train_set):
    """
    Get full path of data based on configs and target path
    example: data/interpolation/test/set5/3x
    """
    scale_path = str(scale) + 'x'
    return os.path.join('preprocessed_data', train_set, scale_path)
    
def display_config():
    print('############################################################')
    print('# Video Super Resolution - Pytorch implementation          #')

    print('############################################################')
    print('')
    print('-------YOUR SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" %(str(arg), str(getattr(args, arg))))
    print('')


def main():
    display_config()
    print('Start train.py main()')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # train_dir = sorted(glob.glob(args.train_set))
    # test_dir = sorted(glob.glob(args.test_set))

    # train_dir = sorted(glob.glob(args.train_set + '/*'))
    # test_dir = sorted(glob.glob(args.test_set + '/*'))     
    train_dataset = VSR_Dataset(dir=args.train_set, trans=transforms.Compose([RandomCrop(48, args.scale), DataAug(), ToTensor()]))

    model_factory = ModelFactory()
    model = model_factory.create_model(args.model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('parameter is : ', 1.0 * params / (1000 * 1000))
    
    loss_fn = get_loss_fn(model.name)
    check_point = os.path.join(args.checkpoint, model.name, str(args.scale) + 'x')
    if not os.path.exists(check_point):
        os.makedirs(check_point)
        
    solver = Solver(model, check_point, model.name, loss_fn=loss_fn, batch_size=args.batch_size,
                    num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                    fine_tune=args.fine_tune, verbose=args.verbose)
    
    val_dataset = VSR_Dataset(dir=args.test_set, trans=transforms.Compose([ToTensor()]))
    
    solver.train(train_dataset, val_dataset)
    print('after solver.train')

    # 3번째 코드
    # for train_fol, test_fol in zip(train_dir, test_dir):
    #     train_dataset  = VSR_Dataset(dir=train_fol, trans =transforms.Compose([RandomCrop(48, args.scale), DataAug(), ToTensor()]))
    #     model_factory = ModelFactory()
    #     model = model_factory.create_model(args.model)
    #     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     print(1.0 * params / (1000 * 1000))
        
    #     loss_fn = get_loss_fn(model.name)

    #     check_point = os.path.join(args.checkpoint, model.name, str(args.scale) + 'x')
    #     if not os.path.exists(check_point):
    #         os.makedirs(check_point)

    #     solver = Solver(model, check_point, model.name, loss_fn=loss_fn, batch_size=args.batch_size,
    #                     num_epochs=args.num_epochs, learning_rate=args.learning_rate,
    #                     fine_tune=args.fine_tune, verbose=args.verbose)

    #     print('Training...')
    #     print(train_dataset.dir_HR)
    #     print(train_dataset.dir_LR)
        
    #     val_dataset  = VSR_Dataset(dir=test_fol, trans =transforms.Compose([ToTensor()]))
    #     solver.train(train_dataset, val_dataset)

    # 2번째 코드
    # for train_fol in train_dir:
    #     train_dataset  = VSR_Dataset(dir=train_fol, trans =transforms.Compose([RandomCrop(48, args.scale), DataAug(), ToTensor()]))
    #     model_factory = ModelFactory()
    #     model = model_factory.create_model(args.model)
    #     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     print(1.0 * params / (1000 * 1000))
        
    #     loss_fn = get_loss_fn(model.name)

    #     check_point = os.path.join(args.checkpoint, model.name, str(args.scale) + 'x')
    #     if not os.path.exists(check_point):
    #         os.makedirs(check_point)

    #     solver = Solver(model, check_point, model.name, loss_fn=loss_fn, batch_size=args.batch_size,
    #                     num_epochs=args.num_epochs, learning_rate=args.learning_rate,
    #                     fine_tune=args.fine_tune, verbose=args.verbose)

    #     print('Training...')

    # for test_fol in test_dir:
    #     val_dataset  = VSR_Dataset(dir=test_fol, trans =transforms.Compose([ToTensor()]))
    #     solver.train(train_dataset, val_dataset)
    
        
    # 원본 코드
    # train_dataset  = VSR_Dataset(dir=args.train_set, trans =transforms.Compose([RandomCrop(48, args.scale), DataAug(), ToTensor()]))
    # model_factory = ModelFactory()
    # model = model_factory.create_model(args.model)
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(1.0 * params / (1000 * 1000))
    
    # loss_fn = get_loss_fn(model.name)

    # check_point = os.path.join(args.checkpoint, model.name, str(args.scale) + 'x')
    # if not os.path.exists(check_point):
    #     os.makedirs(check_point)

    # solver = Solver(model, check_point, model.name, loss_fn=loss_fn, batch_size=args.batch_size,
    #                 num_epochs=args.num_epochs, learning_rate=args.learning_rate,
    #                 fine_tune=args.fine_tune, verbose=args.verbose)

    # print('Training...')
    # val_dataset  = VSR_Dataset(dir=args.test_set, trans =transforms.Compose([ToTensor()]))
    # solver.train(train_dataset, val_dataset)
if __name__ == '__main__':
    main()

