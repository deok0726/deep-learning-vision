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
import torch
import numpy as np

from torch import optim
from ignite.engine import Engine, Events

from utils.metric import *
from utils.data_loaders import get_loaders, get_input_size


class NoveltyDetecter():

    def __init__(self, config):
        self.config = config

    def test(self,
             model,
             dset_manager,
             train_loader,
             valid_loader,
             test_loader,
             use_rapp=False):
        from reconstruction_aggregation import get_diffs

        model.eval()

        with torch.no_grad():
            _train_x, _ = dset_manager.get_transformed_data(train_loader)
            _valid_x, _ = dset_manager.get_transformed_data(valid_loader)
            _test_x, _test_y = dset_manager.get_transformed_data(test_loader)

            if self.config.unimodal_normal:
                _test_y = np.where(np.isin(_test_y, [self.config.target_class]), False, True)
            else:
                _test_y = np.where(np.isin(_test_y, [self.config.target_class]), True, False)

            train_diff_on_layers = get_diffs(_train_x, model)
            valid_diff_on_layers = get_diffs(_valid_x, model)
            test_diff_on_layers = get_diffs(_test_x, model)

        from utils.metric import get_d_norm_loss, get_recon_loss

        _, base_auroc, base_aupr, _ = get_recon_loss(
            valid_diff_on_layers[0],
            test_diff_on_layers[0],
            _test_y,
            f1_quantiles=[],
        )

        if use_rapp:
            _, base_auroc, base_aupr, _ = get_recon_loss(
                valid_diff_on_layers[0],
                test_diff_on_layers[0],
                _test_y,
                f1_quantiles=[],
            )

            _, sap_auroc, sap_aupr, _ = get_d_loss(
                train_diff_on_layers,
                valid_diff_on_layers,
                test_diff_on_layers,
                _test_y,
                gpu_id=self.config.gpu_id,
                start_layer_index=self.config.start_layer_index,
                end_layer_index=self.config.n_layers + 1 - self.config.end_layer_index,
                norm_type=2,
                f1_quantiles=[],
            )

            _, nap_auroc, nap_aupr, _ = get_d_norm_loss(
                train_diff_on_layers,
                valid_diff_on_layers,
                test_diff_on_layers,
                _test_y,
                gpu_id=self.config.gpu_id,
                start_layer_index=self.config.start_layer_index,
                end_layer_index=self.config.n_layers + 1 - self.config.end_layer_index,
                norm_type=2,
                f1_quantiles=[],
            )

            return (base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr)
        return (base_auroc, base_aupr), (0, 0), (0, 0)

    def train(self, model, dset_manager, train_loader, valid_loader, test_loader, test_every_epoch=False):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        trainer = Engine(model.step)
        trainer.model, trainer.optimizer, trainer.config = model, optimizer, self.config
        trainer.train_history = []
        trainer.test_history = []

        evaluator = Engine(model.validate)
        evaluator.model, evaluator.config, evaluator.lowest_loss = model, self.config, np.inf
        evaluator.valid_history = []

        model.attach(trainer, evaluator, self.config)

        def run_validation(engine, evaluator, valid_loader):
            evaluator.run(valid_loader, max_epochs=1)

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, run_validation, evaluator, valid_loader
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def append_train_loss_history(engine):
            engine.train_history += [float(engine.state.metrics['recon'])]

        @evaluator.on(Events.EPOCH_COMPLETED)
        def append_valid_loss_history(engine):
            from copy import deepcopy
            loss = float(engine.state.metrics['recon'])
            if loss < engine.lowest_loss:
                engine.lowest_loss = loss
                engine.best_model = deepcopy(engine.model.state_dict())

            engine.valid_history += [loss]

        def run_test(
            engine,
            detector,
            model,
            dset_manager,
            train_loader,
            valid_loader,
            test_loader,
            use_rapp=False,
        ):
            (base_auroc, base_aupr), _, _ = detector.test(
                model,
                dset_manager,
                train_loader,
                valid_loader,
                test_loader,
                use_rapp=use_rapp,
            )
            if detector.config.verbose >= 1:
                print('BASE AUROC: %.4f AUPR: %.4f' % (base_auroc, base_aupr))
            engine.test_history += [(base_auroc, base_aupr)]

        if test_every_epoch:
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED,
                run_test,
                self,
                model,
                dset_manager,
                train_loader,
                valid_loader,
                test_loader,
                self.config.use_rapp,
            )

        _ = trainer.run(train_loader, max_epochs=self.config.n_epochs)
        model.load_state_dict(evaluator.best_model)

        return trainer.train_history, evaluator.valid_history, trainer.test_history



def get_config():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument('--gpu_id', type=int, default=0)

    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--data', type=str, default='mnist')
    p.add_argument('--unimodal_normal', action='store_true', default=False)
    p.add_argument('--target_class', type=int, default=0)
    p.add_argument('--novelty_ratio', type=float, default=.0)

    p.add_argument('--model', type=str, default='vae')
    p.add_argument('--btl_size', type=int, default=100)
    p.add_argument('--n_layers', type=int, default=10)

    p.add_argument('--use_rapp', action='store_true', default=False)
    p.add_argument('--start_layer_index', type=int, default=0)
    p.add_argument('--end_layer_index', type=int, default=-1)

    config = p.parse_args()

    return config


def main(config):
    from model_builder import get_model
    
    config.input_size = get_input_size(config)
    model = get_model(config)

    dset_manager, train_loader, valid_loader, test_loader = get_loaders(config)
    detecter = NoveltyDetecter(config)

    if config.verbose >= 1:
        print(config)

    if config.verbose >= 2:
        print(model)

    train_history, valid_history, test_history = detecter.train(model,
                                                                dset_manager,
                                                                train_loader,
                                                                valid_loader,
                                                                test_loader,
                                                                test_every_epoch=not config.use_rapp
                                                                )
    (base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr) = detecter.test(
        model,
        dset_manager,
        train_loader,
        valid_loader,
        test_loader,
        use_rapp=config.use_rapp,
    )

    return (base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr), train_history, valid_history, test_history


if __name__ == '__main__':
    config = get_config()
    
    (base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr), _, _, _ = main(config)
    print('BASE AUROC: %.4f AUPR: %.4f' % (base_auroc, base_aupr))
    if config.use_rapp:
        print('RaPP SAP AUROC: %.4f AUPR: %.4f' % (sap_auroc, sap_aupr))
        print('RaPP NAP AUROC: %.4f AUPR: %.4f' % (nap_auroc, nap_aupr))

