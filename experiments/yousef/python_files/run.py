import torch
import time
from experiments.yousef.python_files.utils import _move # TODO move this to pytorch utils



if torch.cuda.is_available():
    DEVICE = 'cuda'
    print('CUDA device detected. Using GPU...')
else:
    DEVICE = 'cpu'
    print('CUDA device NOT detected. Using CPU...')

def train_longformer(model, optimizer, epochs, train_loader, val_loader=None, verbose=2):
    """
    Function to train longformer
    :return:
    """
    model.to(DEVICE)

    for epoch in range(epochs):

        # set model to train mode
        model.train()
        start_epoch_message = f'EPOCH {epoch + 1} STARTED'
        print(start_epoch_message)
        print(f'{"-" * len(start_epoch_message)}')

        start_epoch = time.time()
        # TODO model does not currently support saving
        start_load = time.time()

        training_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            start_train = time.time()
            inputs = _move(inputs)
            targets = _move(targets)
            optimizer.zero_grad()
            loss, outputs = model(labels=targets, **inputs, return_dict=False)
            print(loss)
            print(outputs)
            training_loss += loss.item()

            # TODO evaluate code here

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_train = time.time()

            if verbose > 1:
                print(
                    f'Batch {i + 1} complete. Time taken: load({start_train - start_load:.3g}), '
                    f'train({end_train - start_train:.3g}), total({end_train - start_load:.3g}). '
                )
            start_load = time.time()

