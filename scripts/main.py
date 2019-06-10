import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader

import data
import models
import train
import utils
import copy


#raspic
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.cuda.reset_max_memory_allocated() 

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/data/kaggle-freesound-2019')
    parser.add_argument('--outpath', default='/data/runs/')
    parser.add_argument('--epochs', default=30)
    parser.add_argument('--batch_size', default=64) #32
    return parser.parse_args()


def main(args): 
    np.random.seed(19)
    torch.random.manual_seed(19)
    try:
        os.makedirs(args.outpath)
    except OSError:
        pass
    experiment_path = utils.get_new_model_path(args.outpath)

    train_writer = SummaryWriter(os.path.join(experiment_path, 'train_logs'))
    val_writer = SummaryWriter(os.path.join(experiment_path, 'val_logs'))
    trainer = train.Trainer(train_writer, val_writer)

    # todo: add config
    train_transform = data.build_preprocessing()
    eval_transform = data.build_preprocessing()

    trainds, evalds = data.build_dataset(args.datadir, None)
    trainds.transform = train_transform
    evalds.transform = eval_transform

    model = models.resnet34()
    opt = torch.optim.Adam(model.parameters())
    
    #raspic
    eta_min = 1e-9
    t_max = 25
    sch = CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)

    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    evalloader = DataLoader(evalds, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    max_lwlrap = 0
    for epoch in range(args.epochs):
        print("Epoch:" , epoch)
        trainer.train_epoch(model, opt, sch, trainloader, 3e-4) 
        metrics = trainer.eval_epoch(model, evalloader)

        state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(),
            loss=metrics['loss'],
            lwlrap=metrics['lwlrap'],
            global_step=trainer.global_step,
        )
        export_path = os.path.join(experiment_path, 'last.pth')
        torch.save(state, export_path)
        sch.step()      
        
        
        if metrics['lwlrap'] > max_lwlrap:
            max_lwlrap = metrics['lwlrap']
            best_export_path = os.path.join(experiment_path, 'best.pth')
            torch.save(copy.deepcopy(state), best_export_path)


if __name__ == "__main__":
    
    args = _parse_args()
    main(args)
