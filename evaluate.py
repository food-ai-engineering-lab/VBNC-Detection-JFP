import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
from architecture import MicrocolonyNet
from datatools import McolonyTestData

# Input image dimensions (match this with your dataset)
max_px = 640
min_px = 401

def main(args):    
    # Arguments from command line
    root_train = args.root_train
    root = args.root
    workers = args.workers
    batch = args.batch
    ckpt = args.ckpt
    
    # Set multiprocessing strategy to 'file_system' for pytorch
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Get class names from the training dataset directory
    ds_classes = sorted(os.listdir(root_train))
    
    # Get experimental group name from test dataset directory
    ds_group = root.split('/')[-2] # this reads the first two of each folder's name
    
    # Initialize DataModule   
    ds = McolonyTestData(root=root)
    test_dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)

    # Load model from checkpoint
    model = MicrocolonyNet()
    model = model.load_from_checkpoint(ckpt)
    model.cuda()    # Move model to GPU
    model.eval()    # Set model to evaluation mode
    print('Model loaded')

    itt = iter(test_dl)
    batches = int(np.ceil(len(ds)/batch))

    name_list = []
    pred_list = []
    gt_list = []

    # Loop over data batches
    for i in tqdm(range(batches)):
        d = next(itt)
        input = d[0]['image'].to('cuda')    # move input data to GPU
        fname = d[1]

        # Perform inference
        with torch.no_grad():
            pred = model(input)
            pred = torch.argmax(pred, dim=1)    # get class with highest probability
            pred = list(pred.cpu().numpy())     # move predictions back to CPU and convert to list

            pred_list.append(pred) #list, e.g. [[5], [0]]
            name_list.append(fname)

        # Clear GPU memory every 25 batches
        if i % 25 == 0:
            torch.cuda.empty_cache()

    # Post-processing on the prediction and ground truth lists
    pred_list = [float(x) for x in sum(pred_list,[])] #list, e.g. [5.0, 0.0]
    name_list = sum([list(x) for x in name_list],[])
    gt_list = [name_list[i][:2] for i in range(len(name_list))]
    pred_list = [ds_classes[int(x)] for x in pred_list]
    pred_list = [pred_list[i][:2] for i in range(len(pred_list))]
    
    # Save model inference results as a csv file
    test_df = pd.DataFrame({'filename': name_list, 'gt strain': gt_list, 'pred strain': pred_list})
    test_df.to_csv('microcolony_'+ds_group+'.csv', index=False, header=True)

    # Compute the accuracy
    accuracy = (test_df['gt strain'] == test_df['pred strain']).mean()
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Generate and save confusion matrix
    df = pd.read_csv('microcolony_'+ds_group+'.csv', sep=",")
    confusion_matrix = pd.crosstab(df['gt strain'], df['pred strain'], rownames=['Ground Truth'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix/confusion_matrix.sum(axis=1), annot=True, fmt='.2f', cmap='Blues', annot_kws = {"size": 20})
    plt.savefig('confusion_matrix_'+ds_group+'.png', dpi=400)

if __name__ == '__main__':
    # Define and parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-rt', '--root_train', type=str, help='Root folder of the training dataset', required=True)
    parser.add_argument('-r', '--root', type=str, help='Root folder of the test dataset', required=True)
    parser.add_argument('-c', '--ckpt', type=str, help='Path to checkpoint file', required=True)
    parser.add_argument('-w', '--workers', type=int, help='Number of dataloader workers per GPU', default=0)
    parser.add_argument('-b', '--batch', type=int, help='Batch size per GPU', default=1)
    args = parser.parse_args()
    main(args)
