#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
from utils.repeater import Repeater
from utils.repeater import get_config

from multiprocessing import current_process


if __name__ == '__main__':
    settings = [
        ('gpu_id', int, -1),

        ('n_epochs', int, 10),
        ('batch_size', int, 256),
        ('verbose', int, 0),

        ('data', str, 'mnist'),
        ('unimodal_normal', bool, False),
        ('target_class', int, '1,2,8,9'),
        ('novelty_ratio', float, .0),

        ('model', str, 'vae'),
        ('btl_size', int, 100),
        ('n_layers', int, 10),
        
        ('use_rapp', bool, False),
        ('start_layer_index', int, 0),
        ('end_layer_index', int, -1),

        ('n_trials', int, 5),        
        ('output_file', str, 'output.csv'),
    ]
    config = get_config(settings)

    assert len(config.use_rapp) < 2
    assert len(config.n_trials) < 2
    assert len(config.verbose) < 2

    config.n_trials = [i + 1 for i in range(max(config.n_trials))]
    print(config)

    def main_wrapper(config):
        import os
        import time

        print(current_process().name)
        config.gpu_id = config.gpu_id[int(current_process().name.split('-')[-1]) % len(config.gpu_id)]

        from novelty_detection import main

        time.sleep(config.sleep)
        
        try:
            (base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr), train_history, valid_history, test_history = main(config)
            print(config)
            print(current_process().name, 'BASE AUROC: %.4f AUPR: %.4f' % (base_auroc, base_aupr))
            if config.use_rapp:
                print(current_process().name, 'RaPP SAP AUROC: %.4f AUPR: %.4f' % (sap_auroc, sap_aupr))
                print(current_process().name, 'RaPP NAP AUROC: %.4f AUPR: %.4f' % (nap_auroc, nap_aupr))

            return config, ((base_auroc, base_aupr),
                            (sap_auroc, sap_aupr),
                            (nap_auroc, nap_aupr),
                            train_history,
                            valid_history,
                            test_history,
                            )
        except Exception as e:
            print(e)
            return config, ((.0, .0), (.0, .0), (.0, .0), [], [], [])

    repeater = Repeater(config, main_wrapper, interval=60 if config.use_rapp[0] else 0)
    results = repeater.run()

    from utils.reporter import Reporter

    reporter = Reporter()
    for r in results:
        config, ((base_auroc, base_aupr),
                 (sap_auroc, sap_aupr),
                 (nap_auroc, nap_aupr),
                 train_history,
                 valid_history,
                 test_history,
                 ) = r

        reporter.add(vars(config), {
            'base_auroc': base_auroc,
            'base_aupr': base_aupr,
            'sap_auroc': sap_auroc,
            'sap_aupr': sap_aupr,
            'nap_auroc': nap_auroc,
            'nap_aupr': nap_aupr,
            'train_history': ';'.join(['%.4e' % e for e in train_history]),
            'valid_history': ';'.join(['%.4e' % e for e in valid_history]),
            'auroc_history': ';'.join(['%.4f' % e[0] for e in test_history]),
            'aupr_history': ';'.join(['%.4f' % e[1] for e in test_history]),
        })

    reporter.export(config.output_file)
