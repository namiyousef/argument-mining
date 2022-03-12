import torch
import time
from python_files.utils import _move # TODO move this to pytorch utils
from colabtools.utils import get_gpu_utilization



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
        print(f'GPU Utilisation at epoch {epoch}: {get_gpu_utilization()}')
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
            print(f'GPU Utilisation at batch {i+1} after data loading: {get_gpu_utilization()}')

            optimizer.zero_grad()
            loss, outputs = model(labels=targets, **inputs, return_dict=False)
            print(f'GPU Utilisation at batch {i+1} after training: {get_gpu_utilization()}')

            print(loss)
            print(outputs, outputs.shape)
            training_loss += loss.item()

            """def active_logits(raw_logits, word_ids):
                word_ids = word_ids.view(-1)
                active_mask = word_ids.unsqueeze(1).expand(word_ids.shape[0], HyperParameters.num_labels)
                active_mask = active_mask != NON_LABEL
                active_logits = raw_logits.view(-1, HyperParameters.num_labels)
                active_logits = torch.masked_select(active_logits, active_mask)  # return 1dTensor
                active_logits = active_logits.view(-1, HyperParameters.num_labels)
                return active_logits"""

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
            raise Exception()

