from abc import ABC, abstractmethod
import torch
import shutil
from torch.utils.tensorboard import SummaryWriter
from utils import make_clean_dir


class Regression(ABC, object):
    def __init__(self, model, optimizer, save_dir="./data/regression/"):
        write_graph = True
        self._model = model
        self._optimizer = optimizer
        self._training_loss_object = self._get_training_loss_object()
        self._testing_loss_object = self._get_testing_loss_object()
        self.writer = SummaryWriter(save_dir)
        make_clean_dir(save_dir, confirm_deletion=True)
    #         if write_graph:
    #             self._write_graph()

    def _write_graph(self):
        self.writer.add_graph(self._model)
        self.writer.close()

    @abstractmethod
    def _get_training_loss_object(self):
        return

    @abstractmethod
    def _get_testing_loss_object(self):
        return

    def _train_step(self, xs, ys):
        self._model.train()
        y_predictions = self._model(xs)
        loss = self._training_loss_object(ys.float(), y_predictions.squeeze())
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def train_epoch(self, trainloader, epoch_idx, testloader=None):
        n_batch = len(trainloader)
        running_loss = 0.
        for xs, ys in trainloader:
            ys = ys.float() / 10.
            running_loss += self._train_step(xs, ys)
        mean_loss = running_loss / n_batch
        self.writer.add_scalar('training loss',
                               mean_loss,
                               epoch_idx)
        print(mean_loss, )
        self.writer.close()
        if testloader is not None:
            self.test_epoch(testloader, epoch_idx)

    def _test_step(self, xs, ys):
        self._model.eval()
        y_predictions = self._model(xs)
        loss = self._testing_loss_object(ys.float(), y_predictions.squeeze())
        return loss.item()

    def test_epoch(self, testloader, epoch_idx):
        n_batch = len(testloader)
        running_loss = 0.
        for xs, ys in testloader:
            ys = ys.float() / 10.
            running_loss += self._test_step(xs, ys)

        mean_loss = running_loss / n_batch
        self.writer.add_scalar('testing loss',
                               mean_loss,
                               epoch_idx)
        self.writer.close()
        return mean_loss


class LinearRegression(Regression):
    def __init__(self, model, optimizer, save_dir="./data/linear_regression/"):
        super().__init__(model, optimizer, save_dir)

    def _get_training_loss_object(self):
        return torch.nn.MSELoss()

    def _get_testing_loss_object(self):
        return torch.nn.MSELoss()
