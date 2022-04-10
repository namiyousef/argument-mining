# -- public imports

#    utils
import gc
import plac
import time
import pandas as pd
from ast import literal_eval

#    pytorch
import torch
from torch.utils.data import DataLoader

#    huggingface
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig


# -- private imports
from colabtools.utils import get_gpu_utilization

# -- dev imports
from argminer.data import BigBirdDataset
from argminer.utils import encode_model_name, _move


if torch.cuda.is_available():
    DEVICE = 'cuda'
    print('CUDA device detected. Using GPU...')
else:
    DEVICE = 'cpu'
    print('CUDA device NOT detected. Using CPU...')


@plac.annotations(
    model_name=("name of HuggingFace model to use", "positional", None, str),
    max_length=("max number of tokens", "positional", None, int),
    batch_size=("Batch size for training", "option", "b", int),
    epochs=("Number of epochs to train for", "option", "e", int),
    save_freq=("How frequently to save model, in epochs", "option", None, int),
    verbose=("Set model verbosity", "option", None, int)
)
def main(model_name, max_length, epochs=5, batch_size=16, save_freq=1, verbose=2):
    """
    Trains a HuggingFace model
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    config_model = AutoConfig.from_pretrained(model_name)
    config_model.num_labels = 15

    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config_model)
    optimizer = torch.optim.Adam(params=model.parameters())

    df_texts = pd.read_csv('data/train_NER.csv')
    df_texts.entities = df_texts.entities.apply(lambda x: literal_eval(x))
    df_texts = df_texts.drop('id', axis=1)
    train_set = BigBirdDataset(df_texts, tokenizer, max_length, False)

    train_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True
    }

    train_loader = DataLoader(train_set, **train_params)
    model.to(DEVICE)
    print(f'Model pushed to device: {DEVICE}')
    for epoch in range(epochs):
        model.train()
        start_epoch_message = f'EPOCH {epoch + 1} STARTED'
        print(start_epoch_message)
        print(f'{"-" * len(start_epoch_message)}')
        start_epoch = time.time()

        start_load = time.time()
        training_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            start_train = time.time()
            inputs = _move(inputs, DEVICE)
            targets = _move(targets, DEVICE)
            if DEVICE != 'cpu':
                print(f'GPU Utilisation at batch {i+1} after data loading: {get_gpu_utilization()}')

            optimizer.zero_grad()

            loss, outputs = model(
                labels=targets,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=False
            )
            if DEVICE != 'cpu':
                print(f'GPU Utilisation at batch {i+1} after training: {get_gpu_utilization()}')


            training_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del targets, inputs, loss, outputs
            gc.collect()
            torch.cuda.empty_cache()

            end_train = time.time()

            if verbose > 1:
                print(
                    f'Batch {i + 1} complete. Time taken: load({start_train - start_load:.3g}), '
                    f'train({end_train - start_train:.3g}), total({end_train - start_load:.3g}). '
                )
            start_load = time.time()

        print_message = f'Epoch {epoch + 1}/{epochs} complete. ' \
                        f'Time taken: {start_load - start_epoch:.3g}. ' \
                        f'Loss: {training_loss/(i+1): .3g}'

        if verbose:
            print(f'{"-" * len(print_message)}')
            print(print_message)
            print(f'{"-" * len(print_message)}')

        if epoch % save_freq == 0:
            encoded_model_name = encode_model_name(model_name, epoch+1)
            # TODO
            # - at the beginning of loop create models dir if not eixst
            # - fix saving using .save_pretrained()
            # - make the save path global_var?
            save_path = f'models/{encoded_model_name}.pt'
            torch.save(model.state_dict(), save_path)
            print(f'Model saved at epoch {epoch+1} at: {save_path}')

    encoded_model_name = encode_model_name(model_name, 'final')
    save_path = f'models/{encoded_model_name}.pt'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved at epoch {epoch + 1} at: {save_path}')

if __name__ == '__main__':
    plac.call(main)
