import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader.datautils import DataUtils
from model.FeedforwardNN import FeedForwadNN
from model.modelutils import train_loop, test_loop
from pathlib import Path
import json
from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':
    print("running train.py")

    root_folder = root_folder=Path('data')/'face2comics_v2.0.0_by_Sxela'/'face2comics_v2.0.0_by_Sxela'
    
    hyperparams = {
        'epochs': [1, 3, 5],
        'learning_rate': [0.01, 0.001]
    }
    hp_grid = list(ParameterGrid(hyperparams))

    best_hp = None
    best_loss = 1e6

    tuning_list = []
    for hp in hp_grid:
        print(f"-"*10)

        print("loading data")
        data_utils = DataUtils(root_folder=root_folder, train_len=5000, val_len=2000)

        training_dataset = data_utils.get_training_dataset(all=False)
        validation_dataset = data_utils.get_validation_dataset()

        training_loader = DataLoader(training_dataset, shuffle=True, batch_size=64)
        validation_loader = DataLoader(validation_dataset, shuffle=False, batch_size=64)

        print("configuring model and optimizer")
        print(f"epochs: {hp['epochs']}, learning rate: {hp['learning_rate']}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"current device: {device}")

        model = FeedForwadNN().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

        epochs = hp['epochs']
        for t in range(epochs):
            print(f"\n\t Epoch {t+1}\n")
            train_loop(training_loader, model, loss_fn, optimizer, device)
            test_loss, test_accuracy = test_loop(validation_loader, model, loss_fn, device)

            if test_loss < best_loss:
                best_hp = hp
        
        hp['test_loss'] = test_loss
        hp['test_accuracy'] = test_accuracy
        tuning_list.append(hp)
        print("Done!")

    print("saving tuning results")
    with open(Path('tuning_results.json'), 'w') as fp:
        json.dump(tuning_list, fp)

    print("training model over using best hp")
    print(f"best hp: {best_hp}")

    training_dataset = data_utils.get_training_dataset(all=True)

    training_loader = DataLoader(training_dataset, shuffle=True, batch_size=64)

    print("configuring model and optimizer")
    print(f"epochs: {best_hp['epochs']}, learning rate: {best_hp['learning_rate']}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"current device: {device}")

    model = FeedForwadNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

    epochs = hp['epochs']
    for t in range(epochs):
        print(f"\n\t Epoch {t+1}\n")
        train_loop(training_loader, model, loss_fn, optimizer, device)

    print("saving best model")
    torch.save(model.state_dict(), Path('best_model.pth'))
    

