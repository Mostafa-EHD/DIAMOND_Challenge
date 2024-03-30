import argparse
import os
import pandas as pd
import torch.nn as nn
import numpy as np
import torch
from model import DiamondModel
from torch.utils.data import DataLoader
from data.DiamondDataset import DiamondDataset
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import prepare_data, generate_val_csv
from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC, BinaryF1Score



def main(args):
    """
    Main function to train, validate, and test the DiamondModel for predicting the apparition of central diabetic edema.
    
    The function performs the following steps:
    1. Initializes TensorBoard for logging. Important: we need Tensorboard logs to send you the report of your algorithm. 
    2. Loads and splits the dataset into training, validation, and test sets.
    3. Initializes the DiamondModel, loss criterion, and optimizer.
    4. Trains the model for a specified number of epochs, saving the model with the best validation loss.
    5. Tests the best model on the test set and saves the results. Important: keep this part without changes. 
    
    Parameters:
    - args (argparse.Namespace): Command-line arguments specifying dataset paths, training parameters, and model configuration.
    """

    # Determine if CUDA (GPU) is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Initialize TensorBoard writer
    # Important : You should keep Tensoroard to get the Yaml file of your running
    log_dir = f'./logs/{args.submission_name}/{args.backbone}'
    # Check if the directory exists
    if not os.path.exists(log_dir):
        # If it doesn't exist, create it
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)  # Specify your log directory

    # Initialize metrics 
    AUC = BinaryAUROC(thresholds=None).to(device)
    F1 = BinaryF1Score().to(device)
    ECE = BinaryCalibrationError(n_bins=10, norm='l1').to(device)
    
    # Initialize Data 
    # Load dataframes
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    
    train_dataset = DiamondDataset(df_train, mode = 'train', args=args)
    val_dataset = DiamondDataset(df_val, mode = 'val', args=args)

    
    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    
        
    # Model Initialization
    model = DiamondModel(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay= float(args.weight_decay))
    model.to(device)
    
    best_loss = 999

    # Checkpoint directory
    ckpt_dir = f'./checkpoints/{args.submission_name}/{args.backbone}'

    # Check if the directory exists
    if not os.path.exists(ckpt_dir):
        # If it doesn't exist, create it
        os.makedirs(ckpt_dir)
    
    for epoch in range(args.epochs):

        AUC.reset()
        F1.reset()
        ECE.reset()

        avg_loss_list = []
        
        # Training step
        model.train()
        with torch.enable_grad():
            for batch_idx, data in enumerate(tqdm(train_dataloader)):
                img, labels = prepare_data(data, device)
                logits = model(img)
                loss = criterion(logits, labels)

                # Update metrics
                probs = torch.softmax(logits, dim=1)
                AUC.update(probs[:, 1], labels)
                F1.update(probs[:, 1], labels)
                ECE.update(probs[:, 1], labels)
                
                loss.backward()
                optimizer.step()
                
                avg_loss_list.append(loss.item())
                
            avg_loss_train = np.array(avg_loss_list).mean()
            
            # Log training loss and metrics
            writer.add_scalar('Loss/Train', avg_loss_train, epoch)
            writer.add_scalar('Metric/AUC_Train', AUC.compute(), epoch)
            writer.add_scalar('Metric/F1_Train', F1.compute(), epoch)
            writer.add_scalar('Metric/ECE_Train', ECE.compute(), epoch)
            
            avg_loss_list = []
            
            # Add other metrics
            
            print("[TRAIN] epoch={}/{}, Loss = {}".format(epoch, args.epochs, avg_loss_train))
            
        
        # Validation step
        model.eval()

        AUC.reset()
        F1.reset()
        ECE.reset()

        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                img, labels = prepare_data(data, device)
                logits = model(img)
                avg_loss_list.append(loss.item())

                # Update metrics for validation
                probs = torch.softmax(logits, dim=1)
                AUC.update(probs[:, 1], labels)
                F1.update(probs[:, 1], labels)
                ECE.update(probs[:, 1], labels)
            
            avg_loss_val = np.array(avg_loss_list).mean()
            # Log validation loss and metrics
            writer.add_scalar('Loss/Validation', avg_loss_val, epoch)
            writer.add_scalar('Metric/AUC_Val', AUC.compute(), epoch)
            writer.add_scalar('Metric/F1_Val', F1.compute(), epoch)
            writer.add_scalar('Metric/ECE_Val', ECE.compute(), epoch)
            
            # Add other metrics
            
            print("[VAL] epoch={}/{}, Loss = {}".format(epoch, args.epochs, avg_loss_val))
            
        
        # Save best model
        if avg_loss_val < best_loss:
            print('Saving best model based on Loss, in epoch = ',epoch)
            print('best Val_Loss  =',avg_loss_val)
            best_loss = avg_loss_val
            ep_loss = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth')) 
            
        print(f"[Best chekpoints values] Best_loss = {best_loss}")
        print(f"[Best chekpoints epochs] epoch = {ep_loss}")
    
    # Close the writer
    writer.close()
    
    # Preliminary validation step
    print('Generating the validation CSV using the Best Loss Model !')

    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model.pth')))
    
    # generate the validation csv 
    # DO NOT CHANGE THIS PART OF THE CODE
    generate_val_csv(model, val_dataloader, device)

    
 
    
        
    
if __name__ == "__main__":
    """
    Parses command-line arguments and runs the main training, validation, and testing pipeline for the DiamondModel.
    
    Command-line arguments include learning rate, weight decay, batch size, number of epochs, paths to dataset files, etc.
    """

    parser = argparse.ArgumentParser(description='Training script for predicting the apparition of central diabetic edema.')
    parser.add_argument('--submission_name', type=str, default='TL_Resnet50', help='Used submission name')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Used Backnone name')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--root_dir', type=str, default='./diamond_data', help='Path to the dataset') #./path/to/DeepDRiD
    parser.add_argument('--train_csv', type=str, default='training_set.csv', help='Path to the training CSV')
    parser.add_argument('--val_csv', type=str, default='test_set.csv', help='Path to the validation CSV')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    
    
    

    args = parser.parse_args()
    main(args)