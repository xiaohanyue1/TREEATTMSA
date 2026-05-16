import torch
import argparse
import random
import numpy as np
import os
import logging
import torch.nn as nn
from dataloader import MMDataLoader
from tools.metric import MetricsTop
from models.model import LGMSAT
from config import ConfigRegression
from train import GO

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', default='mosi', type=str, choices=['mosi', 'mosei', 'sims'])
    parser.add_argument('--modelName', default='autograph', type=str)
    parser.add_argument("--train_mode", default="regression", type=str, choices=["regression", "classification"])
    parser.add_argument('--batch_size', default=64, type=int, choices=[32, 64])
    # parser.add_argument('--epochs', default=[100, 60, 100], type=list)
    parser.add_argument('--num_modals', default=3, type=int)
    parser.add_argument('--seed', default=242, type=int)
    parser.add_argument("--ln_rate", default=8e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument('--en_pretrained_path',
                        default='/data/student_code/XHY/base/bert-base-uncased')
    parser.add_argument('--ch_pretrained_path',
                        default='/data/student_code/XHY/base/bert-base-chinese')
    parser.add_argument('--dropout', default=0.5, type=float)


    args = parser.parse_args()

    return args

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_log(args):
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger

def main():
    main_config = parser()
    args = ConfigRegression(main_config).get_config()


    logger = set_log(args)
    set_random_seed(args.seed)
    dataloader = MMDataLoader(args)
    model = LGMSAT(args).to(device)

    # 计算模型参数量
    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer

    logger.info(f'The model has {count_parameters(model)} trainable parameters')

    mlalo = GO(args)
    metrics = MetricsTop().getMetics(args.datasetName)
    # train
    test_accu2 = []
    test_f1 = []
    test_acc_5 = []
    test_acc_7 = []
    test_mae = []
    test_corr = []
    for epoch in range(1, args.epochs + 1):
        tes_accu2, tes_f1, tes_acc5, tes_acc7, tes_mae, tes_corr \
            = mlalo.do_train(args, dataloader, model, metrics, epoch)
        # train_accu.append(tra_accu)
        test_accu2.append(tes_accu2)
        test_f1.append(tes_f1)
        test_acc_5.append(tes_acc5)
        test_acc_7.append(tes_acc7)
        test_mae.append(tes_mae)
        test_corr.append(tes_corr)

    # max_train_accu = max(train_accu)
    # average_train_accu = sum(train_accu) / len(train_accu)
    max_test_accu2 = max(test_accu2)
    aver_test_accu2 = sum(test_accu2) / len(test_accu2)
    max_test_f1 = max(test_f1)
    aver_test_f1 = sum(test_f1) / len(test_f1)
    max_test_acc5 = max(test_acc_5)
    aver_test_acc5 = sum(test_acc_5) / len(test_acc_5)
    max_test_acc7 = max(test_acc_7)
    aver_test_acc7 = sum(test_acc_7) / len(test_acc_7)
    min_test_mae = min(test_mae)
    max_test_corr = max(test_corr)
    # print(f'max_train_accuracy:{max_train_accu}')
    # print(f"average_train_accuracy:{average_train_accu}")
    print(f'max_test_accuracy:{max_test_accu2}')
    print(f"aver_test_accuracy:{aver_test_accu2}")
    print(f'max_test_f1:{max_test_f1}')
    print(f'aver_test_f1:{aver_test_f1}')
    print(f'max_test_acc5:{max_test_acc5}')
    print(f'aver_test_acc5:{aver_test_acc5}')
    print(f'max_test_acc7:{max_test_acc7}')
    print(f"aver_test_acc7:{aver_test_acc7}")
    print(f'min_test_mae:{min_test_mae}')
    print(f'max_test_corr:{max_test_corr}')
    with open('accuracy_results.txt', 'a') as file:
        file.write("..........................................................\n")
        file.write(f'dataset: {args.datasetName}\n')
        # file.write(f'max_train_accuracy: {max_train_accu}\n')
        # file.write(f'average_train_accuracy: {average_train_accu}\n')
        file.write(f'max_test_accuracy: {max_test_accu2}\n')
        file.write(f'aver_test_accuracy: {aver_test_accu2}\n')
        file.write(f'max_test_f1:{max_test_f1}\n')
        file.write(f'aver_test_f1:{aver_test_f1}\n')
        file.write(f'max_test_acc5:{max_test_acc5}\n')
        file.write(f'aver_test_acc5:{aver_test_acc5}\n')
        file.write(f'max_test_acc7:{max_test_acc7}\n')
        file.write(f"aver_test_acc7:{aver_test_acc7}\n")
        file.write(f'min_test_mae:{min_test_mae}\n')
        file.write(f'max_test_corr:{max_test_corr}\n')


if __name__ == '__main__':
    device = torch.device("cuda")
    main()
