import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from absl import app
from absl import flags
from absl import logging

from tensorflow.python.ops import custom_gradient # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops # pylint:disable=g-direct-tensorflow-import

def get_variable_by_name(var_name):
    """Given a variable name, retrieves a handle on the tensorflow Variable."""

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

    def _filter_fn(item):
        try:
            return var_name == item.op.name
        except AttributeError:
            # Collection items without operation are ignored.
            return False

    candidate_vars = list(filter(_filter_fn, global_vars))

    if len(candidate_vars) >= 1:
        # Filter out non-trainable variables.
        candidate_vars = [v for v in candidate_vars if v.trainable]
    else:
        raise ValueError("Unsuccessful at finding variable {}.".format(var_name))

    if len(candidate_vars) == 1:
        return candidate_vars[0]
    elif len(candidate_vars) > 1:
        raise ValueError(
            "Unsuccessful at finding trainable variable {}. "
            "Number of candidates: {}. "
            "Candidates: {}".format(var_name, len(candidate_vars), candidate_vars))
    else:
        # The variable is not trainable.
        return None

custom_gradient.get_variable_by_name = get_variable_by_name
import tensorflow.compat.v1 as tf

# Verbose off 
tf.logging.set_verbosity(tf.logging.ERROR)

tf.disable_eager_execution()
import dataloader
import det_model_fn
import hparams_config
import time 
import sys 


flags.DEFINE_string('eval_name', default=None, help='Eval job name')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
' containing attributes to use as hyperparameters.')
flags.DEFINE_integer('num_cores', default=8, help='Number of TPU cores for training')
flags.DEFINE_integer('eval_batch_size', 1, 'global evaluation batch size')
flags.DEFINE_integer('eval_samples', 5000, 'Number of samples for eval.')
flags.DEFINE_string('val_file_pattern', None, 'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string('val_json_file', None, 'COCO validation JSON containing golden bounding boxes. If None, use the '
'ground truth from the dataloader. Ignored if testdev_dir is not None.')
flags.DEFINE_string('mode', 'train', 'Mode to run: train or eval (default: train)')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_string('model_name', 'efficientdet-d1', 'Model name.')

FLAGS = flags.FLAGS


def main(_):
    # get console command 
    command = 'python ' + ' '.join(sys.argv)
    print(command)
    
    now = time.localtime()
    print("[Python Script 'demo_main.py' start] @ %04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

    CLASS_TABLE = {'CL'  : (1, 'Crack Longitudinal', '균열-길이'),
                   'CC'  : (2, 'Crack Circumferential', '균열-원주'),
                   'SD'  : (3, 'Surface Damage', '표면손상'),
                   'BK'  : (4, 'Broken Pipe', '파손'),
                   'LP'  : (5, 'Lateral Protruding', '연결관-돌출'),
                   'JF'  : (6, 'Joint Faulty', '이음부-손상'),
                   'JD'  : (7, 'Joint Displaced', '이음부-단차'),
                   'DS'  : (8, 'Deposits Silty', '토사퇴적'),
                   'ETC' : (9, 'Etc.', '기타결함'),
                    'PJ' : (10,'Pipe Joint', '이음부'),
                    'IN' : (11,'Inside', '하수관로 내부'),
                    'CAR': (12, 'Outside-Car', '하수관로 외부-자동차'),
                    'INV': (13, 'Outside-Invert', '하수관로 외부-인버트'),
                    'HL' : (14, 'Outside-Manhole', '하수관로 외부-맨홀')}
    # Parse and override hparams
    config = hparams_config.get_detection_config(FLAGS.model_name)
    config.override(FLAGS.hparams)
    # Parse image size in case it is in string format.
    config.image_size = 512
    num_shards = FLAGS.num_cores
    params = dict(
            config.as_dict(),
            model_name=FLAGS.model_name,            
            model_dir=FLAGS.model_dir,
            num_examples_per_epoch=FLAGS.num_examples_per_epoch,
            num_shards=num_shards,
            val_json_file=FLAGS.val_json_file,
            mode=FLAGS.mode)
            
    config_proto = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
    
    model_dir = FLAGS.model_dir
    model_fn_instance = det_model_fn.get_model_fn(FLAGS.model_name)
    max_instances_per_image = config.max_instances_per_image
    if FLAGS.eval_samples:
        eval_steps = int((FLAGS.eval_samples + FLAGS.eval_batch_size - 1) //
                                         FLAGS.eval_batch_size)
    else:
        eval_steps = None
    
    if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)

    config_file = os.path.join(model_dir, 'config.yaml')
    if not tf.io.gfile.exists(config_file):
        tf.io.gfile.GFile(config_file, 'w').write(str(config))

    eval_input_fn = dataloader.InputReader(
        FLAGS.val_file_pattern,
        is_training=False,
        max_instances_per_image=max_instances_per_image)
    
    run_config = tf.estimator.RunConfig(
            model_dir=model_dir,
            session_config=config_proto,
    )

    def get_estimator(global_batch_size):
        params['num_shards'] = 1
        params['batch_size'] = global_batch_size // params['num_shards']
        return tf.estimator.Estimator(
                model_fn=model_fn_instance, config=run_config, params=params)

    eval_est = get_estimator(FLAGS.eval_batch_size)
    # eval_recall = get_estimator(FLAGS.eval_batch_size)

    now = time.localtime()
    print("[Model Ready & Evaluation Start] @ %04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('start eval_est.evaluate')
    eval_results = eval_est.evaluate(eval_input_fn, steps=eval_steps)
    # eval_results_recall = eval_recall(eval_input_fn, steps=eval_steps)
    print('after dataloader.InputReader')
    # Filter evaluation results 
    per_class_results = [('%s'%(key.split('/')[-1]), value) for key, value in eval_results.items() if 'per_class/' in key]
    model_loss = [(key, value) for key, value in eval_results.items() if 'loss' in key]
    
    def key_length(x):
        return CLASS_TABLE[x[0]][0]
    per_class_results = sorted(per_class_results, key = key_length)

    # Print per class iou 0.5 loss
    print('-'*90)
    # iStr = ' {:<3d} | {:<25} | AP @ [ IoU= 0.50 ] = {:0.3f} | {:<30}'
    iStr = ' {:<3d} | {:<25} | AP @ [ IoU= 0.50 ] = {:0.3f}'
    for titleStr, mean_s in per_class_results:
        korname = CLASS_TABLE[titleStr][2]
        fullname = CLASS_TABLE[titleStr][1]
        abv = titleStr
        engname = '%s(%s)'%(abv, fullname)
        num = CLASS_TABLE[titleStr][0]
        value = mean_s
        # print(iStr.format(num, engname, value, korname))
        print(iStr.format(num, engname, value))
    
    # # Print per class iou 0.5 loss
    # print('-'*80)
    # iStr = ' {:<23} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    # iouStr = '{:0.2f}'.format(0.5)
    # areaRng = 'all'
    # maxDets = 100
    # for titleStr, mean_s in per_class_results:
    #     print(iStr.format(titleStr, iouStr, areaRng, maxDets, mean_s))
    
    
    # Print losses
    print('-'*90)
    for titleStr, mean_s in model_loss:
        print('{:<15} = {:0.4e}'.format(titleStr, mean_s))
    now = time.localtime()
    print("[Evaluation Finish!] @ %04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

if __name__ == '__main__':
    app.run(main)


