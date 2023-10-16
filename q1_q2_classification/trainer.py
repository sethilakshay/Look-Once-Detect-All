from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset
from tqdm import tqdm


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(run_name, epoch, model):
    filename = './Model_Checkpoints/{}-Checkpoint-epoch{}.pth'.format(run_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    writer = SummaryWriter(log_dir="./Summary/" + args.run_name)
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0
    writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            # Zeroing the gradients
            optimizer.zero_grad()

            # Predicitng the logits from the network
            output = model(data)

            # # Normalizing the model predictions (logits to be between 0 and 1)
            sigmoid = torch.nn.Sigmoid()

            # # Computing Binary Cross Entropy loss torch.clamp to prevent log values from becoming -infinity
            loss = -wgt*(target * torch.log(sigmoid(output) + 1e-12) + (1-target) * torch.log(1 - sigmoid(output) + 1e-12))

            # Computing the mean over loss
            loss = torch.mean(loss)

            # Backpropogating loss
            loss.backward()
            
            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("mAP: ", map)
                writer.add_scalar("mAP", map, cnt)
                model.train()
            
            cnt += 1

        # Adjusing the learning rate using scheduler
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(args.run_name, epoch, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
