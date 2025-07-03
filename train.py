from dataset_create import NucleiData
from construct_model import CellNet
import torch
from torch_geometric.data import DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np
import logging


class Trainer:

    def __init__(self, label0_ds_pt, label1_ds_pt,
                 node_feat_dim, batch_size, 
                 lr, momentum, weight_decay,
                 epochs, res_dir, device):
        self.device = device
        self.label0_ds_pt = label0_ds_pt
        self.label1_ds_pt = label1_ds_pt
        self.batch_size = batch_size
        self.node_feat_dim = node_feat_dim
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.res_dir = res_dir

        self.loss_hist = []
        self.acc_hist = []
        self.loss_val_hist = []
        self.acc_val_hist = []

        # set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Prepare the datasets
        self.train_loader, self.val_loader, self.test_loader = self._prepare_data()
        self.logger.info("Data preparation complete.")

        # prepare the model
        self.model = CellNet(
            in_channels=self.node_feat_dim,
            out_channels=2,
            batch=True
        )
        self.model.to(self.device)
        self.logger.info("Model initialized.")

    def _prepare_data(self):
        # Load the datasets
        self.logger.info("Loading datasets...")
        label0_ds = torch.load(self.label0_ds_pt)
        label1_ds = torch.load(self.label1_ds_pt)
        self.logger.info(f"Label 0 dataset size: {len(label0_ds)}")
        self.logger.info(f"Label 1 dataset size: {len(label1_ds)}")
        # label0_ds train/val/test split
        self.logger.info("Splitting label 0 dataset...")
        label0_train_ds, label0_val_ds, label0_test_ds = self._split_dataset(label0_ds)
        # label1_ds train/val/test split
        self.logger.info("Splitting label 1 dataset...")
        label1_train_ds, label1_val_ds, label1_test_ds = self._split_dataset(label1_ds)

        train_list = []
        train_list.extend(label0_train_ds)
        train_list.extend(label1_train_ds)
        train_loader = DataLoader(train_list, batch_size=self.batch_size, shuffle=True)

        val_list = []
        val_list.extend(label0_val_ds)
        val_list.extend(label1_val_ds)
        val_loader = DataLoader(val_list, batch_size=self.batch_size, shuffle=False)

        test_list = []
        test_list.extend(label0_test_ds)
        test_list.extend(label1_test_ds)
        test_loader = DataLoader(test_list, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def _split_dataset(self, dataset, train_ratio=0.7, val_ratio=0.1):
        pids = np.array(
            list(set([graph.pid.numpy()[0, 0] for graph in dataset]))
        )
        n_pids = len(pids)
        n_train = int(n_pids * train_ratio)
        n_val = int(n_pids * val_ratio)
        n_test = n_pids - n_train - n_val
        self.logger.info(f"Splitting dataset: {n_train} train slides, {n_val} val slides, {n_test} test slides.")
        np.random.seed(2195719)
        pids_train = np.random.choice(pids, n_train, replace=False)
        pids_rest = np.array([i for i in pids if i not in pids_train])
        np.random.seed(2195719)
        pids_val = np.random.choice(pids_rest, n_val, replace=False)
        pids_test = np.array([i for i in pids_rest if i not in pids_val])

        train_ds = [graph for graph in dataset if graph.pid.numpy()[0, 0] in pids_train]
        val_ds = [graph for graph in dataset if graph.pid.numpy()[0, 0] in pids_val]
        test_ds = [graph for graph in dataset if graph.pid.numpy()[0, 0] in pids_test]
        self.logger.info(
            f"Sample size: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )
        return train_ds, val_ds, test_ds
    
    def train_process(self):
        self.logger.info("Starting training process...")
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            for i, batch in enumerate(self.train_loader):
                batch.to(self.device)
                outputs = self.model(batch)
                loss = criterion(outputs, batch.y.squeeze())

                # accuracy
                pred_labels = F.softmax(outputs.squeeze()).cpu().detach().numpy()
                pred_labels = np.argmax(pred_labels, axis=1).astype("int")
                true_labels = batch.y.cpu().numpy().squeeze()
                acc = np.sum(pred_labels == true_labels) / len(true_labels)

                # print and record
                self.logger.info(
                    f"Batch {i + 1}/{len(self.train_loader)}: "
                    f"Loss = {loss.item():.4f}, Accuracy = {acc:.4f}"
                )
                self.loss_hist.append(loss.item())
                self.acc_hist.append(acc)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.save(
                self.model.state_dict(),
                f"{self.res_dir}/model_epoch_{epoch + 1}.pth"
            )

            # validation
            self.model.eval()
            with torch.no_grad():
                loss = []
                acc = []
                for i, batch in enumerate(self.val_loader):
                    batch.to(self.device)
                    outputs = self.model(batch)
                    criterion = torch.nn.CrossEntropyLoss()
                    loss.append(criterion(outputs, batch.y.squeeze()).item())
                    pred_labels = F.softmax(outputs.squeeze()).cpu().detach().numpy()
                    pred_labels = np.argmax(pred_labels, axis=1).astype("int")
                    true_labels = batch.y.cpu().numpy().squeeze()
                    acc.append(np.sum(pred_labels == true_labels) / len(true_labels))

                # print and record
                loss = np.average(loss)
                acc = np.average(acc)
                self.logger.info(
                    f"Epoch {epoch + 1} Validation: "
                    f"Loss = {loss:.4f}, Accuracy = {acc:.4f}"
                )
                self.loss_val_hist.append(loss)
                self.acc_val_hist.append(acc)

            self.model.train()
        self.logger.info("Training process completed.")