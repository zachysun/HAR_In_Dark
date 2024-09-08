import sys
sys.path.append('')

import os
import torch
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchinfo import summary
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from trainer.trainer_slowfast import Trainer
from dataset import DarkVideoDataset
from img_enhance import LowLightEnhancer
from models.slowfastnet import resnet50

if __name__ == '__main__':
    # ***Set parameters***
    seed = 37
    batch_size = 8
    num_frames = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 6
    lr = 1e-3
    weight_decay = 5e-1
    epochs = 100
    train_enhance_pipeline = ['bright_contrast_adjust']
    eval_enhance_pipeline = ['bright_contrast_adjust']
    normalization_params = ([0.267, 0.267, 0.267], [0.170, 0.170, 0.170])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    lle_train = LowLightEnhancer(None, None, train_enhance_pipeline)
    lle_eval = LowLightEnhancer(None, None, eval_enhance_pipeline)

    # ***Load dataset***
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.4),
        transforms.RandomVerticalFlip(0.4),
        transforms.RandomRotation(40),
        transforms.RandomAffine(40),
        transforms.ToTensor(),
        transforms.Normalize(*normalization_params),
    ])

    transform_eval = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*normalization_params),
    ])

    train_dataset = DarkVideoDataset(source_dir='./data', mode='train_val', transform=transform_train,
                                     sampling_type='uniform', num_frames=num_frames, split_ratio=0.8, is_val=False,
                                     seed=seed, LowLightEnhancer=lle_train)
    val_dataset = DarkVideoDataset(source_dir='./data', mode='train_val', transform=transform_eval,
                                   sampling_type='uniform', num_frames=num_frames, split_ratio=0.8, is_val=True,
                                   seed=seed, LowLightEnhancer=lle_eval)
    test_dataset = DarkVideoDataset(source_dir='./data', mode='validate', transform=transform_eval,
                                    sampling_type='uniform', num_frames=num_frames, 
                                    LowLightEnhancer=lle_eval)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ***Load model***
    # Training slowfast model from scratch
    model_scratch = resnet50(class_num=num_classes, dropout=.5)
    # print('The architecture of slowfast from scratch:\n', model_scratch)
    # print('State keys of slowfast from scratch:\n', model_scratch.state_dict().keys())

    # Setting model
    model_scratch.to(device)
    model = model_scratch
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    engine = Trainer(model, optimizer, criterion, device)
    # Training and validating data
    train_loss_acc = []
    val_loss_acc = []
    no_improve_epochs = 0
    best_val_acc = 0
    no_improve_epochs_max = 10


    def print_log(*values, log=None, end="\n"):
        print(*values, end=end)
        if log:
            if isinstance(log, str):
                log = open(log, "a", encoding='utf-8')
            print(*values, file=log, end=end)
            log.flush()


    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = "./logs/slowfast/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_filename = os.path.join(log_path, f"training_log_{now}.txt")

    with open(log_filename, "w") as log:
        log.truncate()

    print_log("----------Training ---------", log=log_filename)
    print_log(summary(model, input_size=(batch_size, 3, num_frames, 224, 224), verbose=0), log=log_filename)
    print_log(f'learning rate: {lr}, weight decay: {weight_decay}', log=log_filename)
    print_log(f'Epochs: {epochs}, Batch size: {batch_size}, Number of frames: {num_frames}', log=log_filename)
    print_log(f'Method(s) to enhance train data: {train_enhance_pipeline}\n'
              f'Method(s) to enhance val and test data: {eval_enhance_pipeline}\n', log=log_filename)
    print_log(f'Normalization parameters: {normalization_params}', log=log_filename)
    print_log(f'Max number of epochs that no improvement: {no_improve_epochs_max}', log=log_filename)

    print_log("\n-----Loss and Accuracy-----", log=log_filename)
    
    saved_model_dir = "./saved_models/slowfast/"
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    saved_model_path = os.path.join(saved_model_dir, f"slowfast_{now}.pth")

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = engine.train_one_epoch(train_loader)
        train_loss_acc.append((train_loss, train_acc))
        print_log(f'\nepoch:{epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.4f}',
                  log=log_filename)

        val_loss, val_acc = engine.eval_one_epoch(val_loader)
        val_loss_acc.append((val_loss, val_acc))
        print_log(f'val loss: {val_loss:.4f}, val accuracy: {val_acc:.4f}', log=log_filename)

        if val_acc > best_val_acc:
            # save best model
            torch.save(engine.model.state_dict(), saved_model_path)
            best_val_acc = val_acc
            no_improve_epochs = 0
            print_log(f'New best validation accuracy: {best_val_acc:.4f}', log=log_filename)
        else:
            no_improve_epochs += 1
            print_log(f'No improvement in validation accuracy for {no_improve_epochs} epoch(s)',
                      log=log_filename)

        if no_improve_epochs >= no_improve_epochs_max:
            print_log(f'\nEarly Stopping. Best validation accuracy: {best_val_acc:.4f}', log=log_filename)
            # load best model
            engine.model.load_state_dict(torch.load(saved_model_path))
            test_loss, test_acc = engine.test(test_loader)
            print_log(f'\nTest dataset accuracy: {test_acc:.4f}', log=log_filename)
            break

    plot_path = "./plot/slowfast/"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot([x[0] for x in train_loss_acc], label='Train Loss', color=color)
    ax1.plot([x[0] for x in val_loss_acc], label='Validation Loss', color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot([x[1] for x in train_loss_acc], label='Train Accuracy', color=color)
    ax2.plot([x[1] for x in val_loss_acc], label='Validation Accuracy', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Training and Validation Loss and Accuracy')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(plot_path, f"metrics_{now}.png"))
    plt.show()
