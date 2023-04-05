# -*-coding:utf-8-*-



class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, optim, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, print_step_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()

        for step, batch_data in enumerate(self.data_loader):
            self.optim.zero_grad()
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)


            # Forward propagation
            outputs = self.model(inputs)


            # Loss computation
            loss = self.criterion(outputs, labels)
            loss = loss.requires_grad_(True)
            # Backpropagation

            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if print_step_loss:
                print("train: [Step: %d] Iteration loss: %.6f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
