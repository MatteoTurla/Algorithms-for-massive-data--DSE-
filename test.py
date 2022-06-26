import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader.datautils import DataUtils
from model.FeedforwardNN import FeedForwadNN
from model.modelutils import test_loop
from pathlib import Path

if __name__ == '__main__':
    print("running test.py")

    root_folder = root_folder=Path('data')/'face2comics_v2.0.0_by_Sxela'/'face2comics_v2.0.0_by_Sxela'
    

    print(f"-"*10)

    print("loading data")
    data_utils = DataUtils(root_folder=root_folder, train_len=5000, val_len=2000)

    testing_dataset = data_utils.get_testing_dataset()
    testing_loader = DataLoader(testing_dataset, shuffle=False, batch_size=256)

    print("loading model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"current device: {device}")
    model = FeedForwadNN()
    model.load_state_dict(torch.load(Path('best_model.pth')))
    model = model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    print("computing accuracy over test dataset")
    test_loop(testing_loader, model, loss_fn, device)





