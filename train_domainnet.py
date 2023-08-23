import os
import copy
import fire
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from timm.models import create_model


from utils import public_utils
from FedIns.models import vision_transformer
from utils import domainnet_data_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(
        # base para
        seed: int = 1,
        batch: int = 32,
        iters: int = 300,
        lr: float = 1e-2,
        wk_iters: int = 10,
        logout: bool = False,
        data_base_path: str = 'your/domainnet/datapath',
        # ssf_pool
        top_k: int = 3,
        pool_size: int = 25,
        training_loss_weight: float = 0.1,
    ):
    set_seed(seed)
    
    # data and client
    client_name = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']
    client_num = len(client_name)
    client_weights = [1/client_num for i in range(client_num)]
    train_loaders, val_loaders, test_loaders = domainnet_data_utils.prepare_data(data_base_path, batch)

    # server and clients model
    server_model = create_model('vit_base_patch16_224', pretrained=True, num_classes=10, tuning_mode='ssf', pool_size=pool_size).to(device)
    public_utils.para_frozen(server_model, frozen_mode=1)
    client_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    # query model
    queryModel = create_model('vit_base_patch16_224', pretrained=True, num_classes=10, tuning_mode='None', block_type=1).to(device)
    public_utils.para_frozen(queryModel, -1)

    # ssfpool training
    best_acc = 0.
    loss_fun = nn.CrossEntropyLoss()
    for a_iter in range(0, iters):
        optimizers = [optim.SGD(params=client_models[idx].parameters(), lr=lr) for idx in range(client_num)]
        # train
        for wi in range(wk_iters):
            for client_idx, model in enumerate(client_models):
                train_loss, train_acc = public_utils.pool_train(queryModel, model, train_loaders[client_idx], optimizers[client_idx], loss_fun, device, top_k, training_loss_weight)
        # aggregate
        public_utils.communication(server_model, client_models, client_weights, client_num)
        # test
        test_acc_mean = []
        for client_idx, datasite in enumerate(client_name):
            _, test_acc = public_utils.pool_test(queryModel, client_models[client_idx], test_loaders[client_idx], loss_fun, device, top_k, training_loss_weight)
            test_acc_mean.append(test_acc)
        if best_acc < np.mean(test_acc_mean):
            best_acc = np.mean(test_acc_mean)
        print(f'Test Epoch:{a_iter} | Test Acc Mean: {np.mean(test_acc_mean):.{4}f} | Best Acc: {best_acc:.{4}f}')
        
        if logout:
            with open('./domainnet_res.txt', 'a+') as f_out:
                f_out.write(f'Test Epoch:{a_iter} | Test Acc Mean: {np.mean(test_acc_mean):.{4}f} | Best Acc: {best_acc:.{4}f}\n')


if __name__ == '__main__':
    fire.Fire(main)