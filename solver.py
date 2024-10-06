import torch
import torch.nn as nn
import os
from model import RPAResUNet
from torch.optim import Adam
from metrics import iou_metric, iou
from unet import UNet


class Solver(object):

    def __init__(self, train_loader, valid_loader, test_loader, config):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.epochs = config.epochs
        self.alpha = config.alpha

    def train(self):

        # Define model and initialize training configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Using device: ",
            device,
            f"({torch.cuda.get_device_name()})" if torch.cuda.is_available() else "",
        )

        model = RPAResUNet(num_classes=1).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=1e-3)

        best_iou_score = 0.0
        train_sum_iou = 0.0

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            train_iou = 0.0

            for image, mask in self.train_loader:
                image = image.to(device)
                mask = mask.to(device)
                optimizer.zero_grad()
                outputs = model(image.float())

                loss = criterion(outputs.float(), mask.float())
                train_loss += loss.item()

                train_iou += iou_metric(outputs, mask)
                train_sum_iou += train_iou
                rev_iou = 16 - iou_metric(outputs, mask)
                loss += self.alpha * rev_iou

                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_loss = 0
                valid_iou = 0

                for image_val, mask_val in self.valid_loader:
                    image_val = image_val.to(device)
                    mask_val = mask_val.to(device)
                    output_val = model(image_val.float())
                    valid_loss += criterion(output_val.float(), mask_val.float())
                    valid_iou += iou_metric(output_val, mask_val)

            print(
                "Epoch ",
                epoch + 1,
                "Training Loss: ",
                train_loss / len(self.train_loader),
                "Validation Loss: ",
                valid_loss / len(self.valid_loader),
            )
            print(
                "Training IoU: ",
                train_iou / len(self.train_loader),
                "Validation IoU: ",
                valid_iou / len(self.valid_loader),
            )
            if best_iou_score < valid_iou:
                best_iou_score = valid_iou
                torch.save(model, 'best_model.pth')
                print("Model saved.")
            
            train_sum_iou += train_iou / len(self.train_loader)
        print("Mean IoU: ", train_sum_iou / self.epochs)

        PATH = "rdouplenet.pt"
        torch.save(model.state_dict(), PATH)