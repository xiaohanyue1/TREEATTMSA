import pickle
import torch
from torch import nn
import argparse
import logging
from tqdm import tqdm
from tools.utils import AverageMeter, Contrastive_loss, totolloss, con_loss, KL_regular
from tools.functions import dict_to_str
from tools.schedule import get_scheduler



logger = logging.getLogger('MSA')
device = torch.device("cuda")


class L2_distance(nn.Module):
    def __init__(self):
        super(L2_distance, self).__init__()

    def forward(self, f1, f2):
        # 计算差值
        diff = torch.sub(input=f1, other=f2)
        squared_diff = diff ** 2
        # 沿最后两个维度求和（50, 256），得到每个样本的 L2 距离
        l2_dist = torch.sqrt(torch.sum(squared_diff, dim=(1, 2)))
        return torch.mean(l2_dist)

class GO():
    def __init__(self, args):
        assert args.train_mode == 'regression'
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.MSE = nn.MSELoss()
        self.l2_distance = L2_distance()

    def do_train(self, args, dataset, model, metrics, epoch):

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.ln_rate, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        scheduler = get_scheduler(optimizer, args)
        # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15, verbose=True)
        self.train(args, dataset['train'], model, optimizer, scheduler, metrics, epoch)
        vel_result = self.valid(args, dataset['valid'], model, optimizer, metrics, epoch)
        test_acc2, test_f1, test_acc5, test_acc7, test_mae, test_corr \
            = self.test(args, dataset['test'], model, optimizer, metrics, epoch)

        return test_acc2, test_f1, test_acc5, test_acc7, test_mae, test_corr

    def loss_part(self, model, input_ids, img, audio, input_mask, label):

        output = model(input_ids, img, audio, input_mask)
        loss = self.MSE(output, label)
        return output, loss
    
    def train(self, args, dataset, model, optimizer, scheduler, metrics, epoch):
        train_pbar = tqdm(enumerate(dataset))
        losses = AverageMeter()
        y_pred, y_true = [], []
        # techer_model.eval()
        model.train()
        for cur_iter, data in train_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)

            if args.use_bert:
                input_ids, input_mask, segment_ids = text[:, 0, :].long(), \
                                                     text[:, 1, :].float(), \
                                                     text[:, 2, :].long()
            else:
                input_ids, input_mask = text[:, 0, :].long(), text[:, 1, :].float()

            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batch_size = img.shape[0]
            # print(text_context.size())
            output, loss = self.loss_part(model, input_ids, img, audio, input_mask, label)

            losses.update(loss.item(), batch_size)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            train_pbar.set_description('train')
            train_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                    'all_loss': '{:.5f}'.format(losses.value_avg),
                                    # 'loss_distance' : '{:.5f}'.format((loss_distance * 0.1)),
                                    'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        train_results = metrics(pred, true)
        logger.info('%s: >> ' % dict_to_str(train_results))
        print('train: ', train_results)

        return train_results["acc_2"]

    def valid(self, args, dataset, model, optimizer, metrics, epoch):
        valid_pbar = tqdm(enumerate(dataset))

        losses = AverageMeter()
        y_pred, y_true = [], []

        model.eval()
        with torch.no_grad():
            for cur_iter, data in valid_pbar:
                img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)

                if args.use_bert:
                    input_ids, input_mask, segment_ids = text[:, 0, :].long(), \
                                                         text[:, 1, :].float(), \
                                                         text[:, 2, :].long()
                else:
                    input_ids, input_mask = text[:, 0, :].long(), text[:, 1, :].float()

                label = data['labels']['M'].to(device)
                label = label.view(-1, 1)
                batch_size = img.shape[0]
                # print(text_context.size())
                output, loss = self.loss_part(model, input_ids, img, audio, input_mask, label)

                losses.update(loss.item(), batch_size)

                y_pred.append(output.cpu())
                y_true.append(label.cpu())

                losses.update(loss.item(), batch_size)

                valid_pbar.set_description('eval')
                valid_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                        'loss': '{:.5f}'.format(losses.value_avg),
                                        'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            valid_results = metrics(pred, true)
            logger.info('%s: >> ' % dict_to_str(valid_results))
            print('valid: ', valid_results)

            return valid_results["acc_2"]

    def test(self, args, dataset, model, optimizer, metrics, epoch):
        test_pbar = tqdm(enumerate(dataset))

        losses = AverageMeter()
        y_pred, y_true = [], []

        model.eval()
        with torch.no_grad():
            for cur_iter, data in test_pbar:
                img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)

                if args.use_bert:
                    input_ids, input_mask, segment_ids = text[:, 0, :].long(), \
                                                         text[:, 1, :].float(), \
                                                         text[:, 2, :].long()
                else:
                    input_ids, input_mask = text[:, 0, :].long(), text[:, 1, :].float()

                label = data['labels']['M'].to(device)
                label = label.view(-1, 1)
                batch_size = img.shape[0]
                # print(text_context.size())
                output, loss = self.loss_part(model, input_ids, img, audio, input_mask, label)

                losses.update(loss.item(), batch_size)

                y_pred.append(output.cpu())
                y_true.append(label.cpu())

                losses.update(loss.item(), batch_size)

                test_pbar.set_description('test')
                test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                       'loss': '{:.5f}'.format(losses.value_avg),
                                       'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            test_results = metrics(pred, true)
            logger.info('%s: >> ' % dict_to_str(test_results))
            print('test: ', test_results)

            return test_results["acc_2"], test_results['F1_score'], \
                   test_results['Mult_acc_5'], test_results['Mult_acc_7'], \
                   test_results['MAE'], test_results['Corr']
