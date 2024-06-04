import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from monai.data import PersistentDataset, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    Rotate90d,
    Resized,
    RandRotated,
    RandFlipd,
    RandZoomd,
    CropForegroundd,
    RandScaleCropd
)

def parse_tuple(s):
    try:
        return tuple(map(float, s.split(',')))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {s}")

class Trainer:
    def __init__(self, train_data, val_data, test_data, crop_size, batch_size, num_workers, saving_directory, device,
                 lr, max_lr, num_epochs, val_interval):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.saving_directory = saving_directory
        self.device = device
        self.lr = lr
        self.max_lr = max_lr
        self.num_epochs = num_epochs
        self.val_interval = val_interval
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_ds = None
        self.model = None
        self.loss_function = None
        self.loss_function_val = None
        self.optimizer = None
        self.scheduler = None
        self.auc_metric = None
        self.weight = None
        self.number_of_no_fractures = None
        self.number_of_fractures = None
        self.get_transforms()
        self.initialize_loaders()
        self.initialize_model()

    def threshold_for_cropforeground(self, x): 
        '''
        Helper function for Monai's CropForegroundd function. Specifies that rows or columns of pixels,
        which do not include pixels with values between 0.3 and 0.7 can be removed.    
        '''
        mask = (x >= 0.3) & (x <= 0.7)
        return mask
    
    def get_transforms(self):
        train_transforms = Compose([
            LoadImaged("image"),
            EnsureChannelFirstd("image"),
            ScaleIntensityd("image"),
            CropForegroundd(keys = "image", source_key = "image", allow_smaller = False, select_fn=self.threshold_for_cropforeground, margin=0),
            Rotate90d("image", k=3),
            RandScaleCropd("image", 0.9, random_center = True),
            RandRotated(keys="image", range_x= (-np.pi/12, np.pi/12), prob=0.8, keep_size=True),
            RandFlipd(keys="image", spatial_axis=1, prob=0.5),
            RandZoomd(keys="image", min_zoom=1, max_zoom=1.3, prob=1),
            Resized("image", spatial_size=self.crop_size, mode="area")
            ])
        
        val_test_transforms = Compose([
            LoadImaged("image"),
            EnsureChannelFirstd("image"),
            ScaleIntensityd("image"),
            CropForegroundd(keys = "image", source_key = "image", allow_smaller = False, select_fn=self.threshold_for_cropforeground, margin=0),
            Rotate90d("image", k=3),
            Resized("image", spatial_size=self.crop_size, mode="area")
        ])    
        return train_transforms, val_test_transforms

    def initialize_loaders(self):
        train_transforms, val_test_transforms = self.get_transforms()

        self.train_ds = PersistentDataset(data=self.train_data, transform=train_transforms,
                                      cache_dir=os.path.join(self.saving_directory, "dataset"))
        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

        val_ds = PersistentDataset(data=self.val_data, transform=val_test_transforms,
                                    cache_dir=os.path.join(self.saving_directory, "dataset"))
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

        test_ds = PersistentDataset(data=self.test_data, transform=val_test_transforms,
                                     cache_dir=os.path.join(self.saving_directory, "dataset"))
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def initialize_model(self):
        self.calculate_loss_weight()
        self.model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=1).to(self.device)
        self.loss_function = torch.nn.BCEWithLogitsLoss(self.pos_weight.to(self.device))
        self.loss_function_val = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.max_lr, epochs=self.num_epochs,
                                    steps_per_epoch=self.calculate_num_batches())
        self.auc_metric = ROCAUCMetric()

    def calculate_loss_weight(self):
        ''' calculate pos_weight for the loss function to balance the unequal number of fracture and non-
        fracture images'''
        train_labels = [item["label"] for item in self.train_data]
        self.number_of_fractures = sum(train_labels)
        self.number_of_no_fractures = len(train_labels) - sum(train_labels)
        self.weight = self.number_of_no_fractures / self.number_of_fractures
        self.pos_weight = torch.FloatTensor([self.weight])
        
    def calculate_num_batches(self):
        '''The number of batches for the scheduler'''
        num_batches = np.ceil(len(self.train_ds) / self.train_loader.batch_size).astype(int).item()
        return num_batches

    def training(self):
        best_metric = -1                  #highest validation AUC so far
        best_metric_epoch = -1            #index of best epoch
        acc_train_l=[]                    #list that holds the training accuracy per epoch
        acc_val_l=[]                      #list that holds the validation accuracy per epoch
        auc_train_l=[]                    #list that holds the training AUC per epoch
        auc_val_l=[]                      #list that holds the validation AUC per epoch
        loss_train_l=[]                   #list that holds the average training loss per epoch
        loss_val_l=[]                     #list that holds the average validation loss per epoch
        val_mcc_l = []                    #list that holds the validation MCC values
        
        for epoch in range(self.num_epochs):
            #set model to train mode
            self.model.train()
            train_loss = 0
            step = 0
        
            #define y_pred and y_true as empty tensors to hold the values of the predictions and ground truth
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y_true = torch.tensor([], dtype=torch.long, device=device)
            
            for train_batch in self.train_loader:
                step += 1
                #get the images and labels
                image = train_batch["image"].to(self.device)
                labels = train_batch["label"].to(self.device)
            
                #forward pass
                y_logits = self.model(image)
                y_logits = y_logits.squeeze(1)
                y_pred = torch.cat([y_pred, torch.sigmoid(y_logits)], dim=0)
                y_true = torch.cat([y_true, labels], dim=0)
        
                #zero_grad
                self.optimizer.zero_grad()
        
                #format labels and output to fit the loss function        
                labels = labels.to(torch.float)
          
                #calculate the loss
                loss = self.loss_function(y_logits, labels)
               
                #backward propagation
                loss.backward()
        
                #optimization step
                self.optimizer.step()
        
                #learning rate scheduler step
                self.scheduler.step()
        
                #add the loss of the current batch to the total training loss
                train_loss += loss.item()
                
                
            #devide the total training loss by a weighted sum of the number of non fractures and the number of fractures.
            #since the loss function included a weight, it is used again to calculate the average loss, to receive
            #an average loss that is comparable to the loss in the validation data set, where no weights were used.
            train_loss /= (self.number_of_no_fractures + self.number_of_fractures * self.weight)
        
            loss_train_l.append(train_loss)
        
            #AUC calculation
            self.auc_metric(y_pred, y_true)
            train_AUC = self.auc_metric.aggregate()
            self.auc_metric.reset()
            auc_train_l.append(train_AUC)
        
            #accuracy calculation
            train_acc = torch.eq(torch.round(y_pred), y_true)  #check for correct and incorrect prediction
            train_acc = train_acc.sum().item() / len(train_acc)    #calculate percentage of correct prediction
            acc_train_l.append(train_acc)
          
        
            if (epoch + 1) % self.val_interval == 0:
                #set model to evaluation mode with inference_mode 
                self.model.eval()
                with torch.inference_mode():
                    
                    #resets y_pred and y_true
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y_true = torch.tensor([], dtype=torch.long, device=device)
                    all_logits = torch.tensor([], dtype=torch.float32, device = device)
                    
                    #resets val_loss
                    val_loss = 0
                    val_step = 0
                    for val_batch in self.val_loader:
        
                        #get image and label
                        image = val_batch["image"].to(self.device)
                        val_labels = val_batch["label"].to(self.device)
                     
                        #forward pass
                        y_logits =  self.model(image)
                        y_logits = y_logits.squeeze(1)
                        all_logits = torch.cat([all_logits, y_logits], dim = 0)
                        y_pred = torch.cat([y_pred, torch.sigmoid(y_logits)], dim=0) 
                        y_true = torch.cat([y_true, val_labels], dim=0)
        
                        val_loss = self.loss_function_val(y_logits, val_labels.to(torch.float))
                  
        
                    #get the average loss by dividing the total loss by the number of images
                    val_loss /= y_logits.shape[0]
        
                    #add the loss to a list of epoch losses
                    loss_val_l.append(val_loss)
        
                    #calculate AUC
                    self.auc_metric(y_pred, y_true)
                    val_AUC = self.auc_metric.aggregate()
                    self.auc_metric.reset()
                    auc_val_l.append(val_AUC)
        
                    #calculate val Accuracy
                    acc_value = torch.eq(torch.round(y_pred), y_true)
                    val_acc = acc_value.sum().item() / len(acc_value)
                    acc_val_l.append(val_acc)
        
                    #calculate matthews correlation coefficient(MCC)
                    val_mcc = matthews_corrcoef(y_true.cpu().numpy(), torch.round(y_pred).cpu().numpy())
                    val_mcc_l.append(val_mcc)
                
                    #check if current epoch has the highest MCC so far
                    if val_mcc > best_metric:
                        best_metric = val_mcc   #set new highest mcc
                        best_metric_epoch = epoch + 1
                        
                        #save the current model 
                        torch.save(self.model.state_dict(), os.path.join(
                            self.saving_directory, "best_metric_model.pth"))

        #when done training, save the metrics.
        torch.save([loss_train_l,loss_val_l,acc_train_l,acc_val_l,auc_train_l,auc_val_l, val_mcc_l], os.path.join(self.saving_directory, "saved_metrics.pth"))

    def testing(self):
        #load best model
        self.model.load_state_dict(torch.load(os.path.join(self.saving_directory, "best_metric_model.pth"), map_location=torch.device(self.device)))
        
        #testing the model on the test dataset
        y_true = torch.tensor([], dtype=torch.float32, device = self.device)
        y_pred = []
        all_logits = torch.tensor([], dtype=torch.float32, device = self.device)
        unique_case_ids = []
        
        #set model to eval
        self.model.eval()
        with torch.inference_mode():
            for test_batch in self.test_loader:
                image = test_batch["image"].to(device)
                labels = test_batch["label"].to(device)
                
                #forward pass
                y_logits = torch.sigmoid(model(image))
                y_logits = y_logits.squeeze(1)
                all_logits = torch.cat([all_logits, y_logits], dim = 0)
                y_true = torch.cat([y_true, labels], dim=0)
        
        
        #calculate AUC with the logits
        self.auc_metric(all_logits, y_true)
        test_AUC = self.auc_metric.aggregate()
        self.auc_metric.reset()
        
        #get the prediction form the logit
        y_pred = torch.round(all_logits)
        
        #get classification report
        test_classification_report = classification_report(y_true.cpu(), y_pred.cpu(), target_names=["No Fracture", "Fracture"], digits=4)
        
        #get confusion matrix
        test_conf_matrix = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
        
        #calculate the test accuracy
        acc_value = torch.eq(y_pred, y_true)
        test_acc = acc_value.sum().item() / len(acc_value)
        
        # Calculate the MCC for the test data set
        test_mcc = matthews_corrcoef(y_true.cpu().numpy(), y_pred.cpu().numpy())
        
        torch.save([test_classification_report, test_conf_matrix, test_acc, test_mcc], "test_metrics.pth")        

parser = argparse.ArgumentParser(description="Train a CNN for ankle fracture classification")
parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--max_lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--max_num_workers', type=float, default=4, help='up to the provided maximum, all workers will be used')
parser.add_argument('--random_seed', type=float, default=0, help='set a random seed')
parser.add_argument('--crop_size', type=parse_tuple, default=(640, 640), help='set crop size as a tuple (width, height)')
parser.add_argument('--val_interval', type=float, default=1, help='number of training epochs until ')
parser.add_argument('--saving_directory', type=str, default='/sc-projects/sc-proj-cc06-ag-ki-radiologie/OSG/CNN_model', help='output_directory')
args = parser.parse_args()

num_workers = min(args.max_num_workers, os.cpu_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(args.random_seed)
set_determinism(seed=args.random_seed)
train_dataframe = pd.read_csv(os.path.join(args.saving_directory, "train_dataframe.csv"))
val_dataframe = pd.read_csv(os.path.join(args.saving_directory, "val_dataframe.csv"))
test_dataframe = pd.read_csv(os.path.join(args.saving_directory, "test_dataframe.csv"))
train_data = train_dataframe.to_dict(orient="records")
val_data = val_dataframe.to_dict(orient="records")
test_data = test_dataframe.to_dict(orient="records")

if __name__ == "__main__":
    trainer = Trainer(train_data = train_data, 
                      val_data = val_data,
                      test_data = test_data, 
                      crop_size = args.crop_size, 
                      batch_size = args.batch_size, 
                      num_workers = num_workers, 
                      saving_directory = args.saving_directory, 
                      device = device,
                      lr = args.lr, 
                      max_lr = args.max_lr, 
                      num_epochs = args.num_epochs,
                      val_interval = args.val_interval)
    trainer.training()
    trainer.testing()

