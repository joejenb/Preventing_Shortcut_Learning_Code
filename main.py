from audioop import avg
from datetime import datetime
from filecmp import cmp
from inspect import Parameter
import os
import sys
#from sklearn import datasets
import yaml
import argparse

import lightly
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision

from lightly.data.dataset import LightlyDataset

from utils.data.water_birds import WaterbirdsDataset, eval
from utils.intrinsic_dim import intrinsic_dim as ID
from utils.similarity import SimilarityMeasure
from utils.data.cifar_10 import Cifar_10_C, Cifar10_2_Dataset
from utils.data.mnist import ColouredMNIST10

from models.BYOL import BYOL
from models.ResNet import ResNet18


logs_root_dir = os.path.join(os.getcwd(), 'validation_logs')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29511"

torch.cuda.ipc_collect()
torch.cuda.empty_cache()
#print(torch.cuda.memory_summary())

# use a GPU if available
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

def log_state(logger, epoch, intrinsic_dim, cka_similarity, r2cca_similarity, avg_identity, scheduler):
    for block_num, id in enumerate(intrinsic_dim.values()):
        logger.add_scalar(f'Intrinsic Dimensionality/Block {block_num}', id, epoch)
    
    logger.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch)

    #Add get image
    logger.add_figure('Heatmap/CKA', heatmap(cka_similarity), epoch)
    logger.add_figure('Heatmap/R2CCA', heatmap(r2cca_similarity), epoch)

    for block_num in avg_identity.keys():
        logger.add_scalar(f'Identity Measure/Block {block_num}', avg_identity[block_num], epoch)
    logger.flush()

def log_loss(logger, epoch, avg_train_loss, avg_measure_losses, avg_scaled_losses):
    logger.add_scalar('Loss/Train', avg_train_loss, epoch)
    logger.add_scalar('Loss/Measure', sum(avg_measure_losses.values())/len(avg_measure_losses), epoch)
    logger.add_scalar('Loss/Scaled Measure', sum(avg_scaled_losses.values())/len(avg_scaled_losses), epoch)

    for block_num in avg_measure_losses.keys():
        logger.add_scalar(f'Loss/Measure/Block {block_num}', avg_measure_losses[block_num], epoch)
        logger.add_scalar(f'Loss/Scaled Measure/Block {block_num}', avg_scaled_losses[block_num], epoch)

    logger.flush()

def log_accuracy(logger, epoch, train_top_1, val_top_1, test_top_1, train_top_5, val_top_5, test_top_5, ood_top_1, ood_top_5):
    for dataset_name, sub_dataset in ood_top_1.items():
        for sub_dataset_name, sub_dataset_top_1 in sub_dataset.items():
            logger.add_scalar(f'Accuracy-Top-1/{dataset_name}/{sub_dataset_name}', sub_dataset_top_1, epoch)
            sub_dataset_top_5 = ood_top_5[dataset_name][sub_dataset_name]
            logger.add_scalar(f'Accuracy-Top-5/{dataset_name}/{sub_dataset_name}', sub_dataset_top_5, epoch)

    logger.add_scalar('Accuracy-Top-1/Train', train_top_1, epoch)
    logger.add_scalar('Accuracy-Top-1/Validation', val_top_1, epoch)
    logger.add_scalar('Accuracy-Top-1/Test', test_top_1, epoch)
    logger.add_scalar('Accuracy-Top-5/Train', train_top_5, epoch)
    logger.add_scalar('Accuracy-Top-5/Validation', val_top_5, epoch)
    logger.add_scalar('Accuracy-Top-5/Test', test_top_5, epoch)
    logger.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument('--scale', type=float)

    args = parser.parse_args()

    config_name = args.conf
    scale = args.scale
    with open(os.path.join("configs", config_name)) as file:
        config = yaml.safe_load(file)

    config["epochs"] = 21

    return config, scale

def save_params(log_dir, epoch, model, optimiser, scheduler, batch_size, params_file):
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimiser': optimiser.state_dict(),
                'scheduler': scheduler.state_dict(), 'batch_size': batch_size, 'log_dir': log_dir
                }, params_file)

def load_params(parser, logger, model, optimiser, scheduler):
    start_epoch = 0
    if parser["params_file"]:
        params_file = parser["params_file"]
        checkpoint = torch.load(params_file)
        log_dir = checkpoint['log_dir']
        logger.add_text("hparams:", log_dir)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return model, optimiser, scheduler, start_epoch

def heatmap(activation_map):
    '''fig = plt.figure()
    fig.add_subplot(111)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = torch.tensor(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
    return data
    '''

    return  sns.heatmap(activation_map.to("cpu"),
                vmin=0.0, vmax=1.0, cmap="magma", xticklabels=2, yticklabels=2,
                cbar=True, annot=True, cbar_kws={'shrink': 0.9}, square=True).get_figure()

    
def get_data_sets(parser):
    # No additional augmentations for the test set
    train_transforms_list = [torchvision.transforms.Resize(size=(parser["image_size"], parser["image_size"]))]
    train_transforms_list.append(torchvision.transforms.Grayscale(3))

    #if parser["model_name"] == "ResNet18":# or parser["dataset_name"] == "coloured_mnist":
    train_transforms_list.append(torchvision.transforms.ToTensor())

    train_transforms = torchvision.transforms.Compose(train_transforms_list)

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(parser["image_size"], parser["image_size"])),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(
        #    mean=lightly.data.collate.imagenet_normalize['mean'],
        #    std=lightly.data.collate.imagenet_normalize['std'],
        #)
    ])

    if parser["dataset_name"] == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root=parser["data_path"], train=True, download=True)
        val_set = torchvision.datasets.CIFAR10(root=parser["data_path"], train=True, download=True)
        #train_set, _ = random_split(full_set_1, [45000, 5000])
        #_, val_set = random_split(full_set_2, [45000, 5000])
        test_set = torchvision.datasets.CIFAR10(root=parser["data_path"], train=False, download=True)
        num_classes = 10
    elif parser["dataset_name"] == "waterbirds":
        '''May have non working version of wilds on local machine'''
        full_dset = WaterbirdsDataset(root_dir=parser["data_path"], download=True)
        test_set = full_dset.get_subset("test")
        train_set = full_dset.get_subset("train")
        print(train_set)
        val_set = full_dset.get_subset("val")
        num_classes = 2
    elif parser["dataset_name"] == "coloured_mnist":
        #Start with different test set for val
        #Too complex to 
        train_set = torchvision.datasets.MNIST(root=parser["data_path"], train=True, download=True)#, transform=torchvision.transforms.ToTensor())
        val_set = torchvision.datasets.MNIST(root=parser["data_path"], train=False, download=True)#, transform=torchvision.transforms.ToTensor())
        test_set = torchvision.datasets.MNIST(root=parser["data_path"], train=False, download=True)#, transform=torchvision.transforms.ToTensor())

        train_set = LightlyDataset.from_torch_dataset(dataset=train_set, transform=train_transforms)
        val_set = LightlyDataset.from_torch_dataset(dataset=val_set, transform=test_transforms)
        test_set = LightlyDataset.from_torch_dataset(dataset=test_set, transform=test_transforms)

        #train_set = ColouredMNIST10(train_set, classes=10, colors=[1, 1], std=0.00, train=True)
        #val_set = ColouredMNIST10(val_set, classes=10, colors=train_set.colors, std=0.00)
        #test_set = ColouredMNIST10(test_set, classes=10, colors=[1, 1], std=0)
        train_set = ColouredMNIST10(train_set, classes=10, colors=[0, 1], std=0.05, train=True)
        val_set = ColouredMNIST10(val_set, classes=10, colors=train_set.colors, std=0.05)
        test_set = ColouredMNIST10(test_set, classes=10, colors=[1, 1], std=0)
        num_classes = 10

    #train_set = LightlyDataset.from_torch_dataset(dataset=train_set, transform=train_transforms)
    #val_set = LightlyDataset.from_torch_dataset(dataset=val_set, transform=test_transforms)
    #test_set = LightlyDataset.from_torch_dataset(dataset=test_set, transform=test_transforms)
    return train_set, val_set, test_set, num_classes

def get_data_loaders(parser, num_gpus_to_use, rank):
    """Helper method to create dataloaders for ssl, kNN train and kNN test
    Args:
        batch_size: Desired batch size for all dataloaders
    """
    # Use SimCLR augmentations, additionally, disable blur for cifar10
    col_fn = None
    if parser["model_name"] == "BYOL":

        #cj_prob=0.,
        col_fn = lightly.data.SimCLRCollateFunction(
            input_size=parser["image_size"],
            gaussian_blur=0.,
        )

    #batch_size = batch_size // num_gpus_to_use

    train_set, val_set, test_set, num_classes = get_data_sets(parser)
    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_set, num_replicas=num_gpus_to_use, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_set, num_replicas=num_gpus_to_use, rank=rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=test_set, num_replicas=num_gpus_to_use, rank=rank)
    else:
        train_sampler = val_sampler = test_sampler = None

    dataloader_train_ssl = torch.utils.data.DataLoader(
        train_set,
        batch_size=parser["batch_size"],
        shuffle=False,
        collate_fn=col_fn,
        drop_last=True,
        pin_memory=True,
        num_workers=parser["num_workers"],
        sampler=train_sampler
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        val_set,
        batch_size=parser["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=parser["num_workers"],
        sampler=val_sampler
    )

    dataloader_test = torch.utils.data.DataLoader(
        test_set,
        batch_size=parser["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=parser["num_workers"],
        sampler=test_sampler
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test, num_classes


def clean_up(model, num_gpus_to_use):
    # delete model and trainer + free up cuda memory
    del model
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    if num_gpus_to_use > 1:
        torch.distributed.destroy_process_group()


def validate(dataloader_test, model, similarity_measure, collect_state=False):
    torch.cuda.empty_cache()
    total_top_1, total_top_5, total_num = 0, 0, 0
    layers = [2, 3, 4, 5] if collect_state else []

    total_identity_measures = {block_num: torch.zeros(1, device=model.device) for block_num in layers}
    total_intrinsic_dim = {block_num: torch.zeros(1, device=model.device) for block_num in layers}

    total_cka_measures = torch.zeros(9, 9).to(model.device)
    total_r2cca_measures = torch.zeros(9, 9).to(model.device)

    with torch.no_grad():
        for val_batch in dataloader_test:
            num, activations, top_1, top_5 = model.validation_step(val_batch, collect_state)
            
            total_top_1 += top_1
            total_top_5 += top_5
            total_num += torch.tensor(num, device=model.device)

            if collect_state:
                activations_subset = {block_num: activations[block_num][1] for block_num in layers}
                activations_list = []
                for activation in activations.values():
                    activations_list += activation

                #identity_measure = similarity_measure.calculate_identity_similarity(activations_subset)
                cka_measure, r2cca_measure = similarity_measure.calculate_within_layer_similarity(activations_list)

                for block_num in total_identity_measures.keys():
                    intrinsic_dim, _, _ = ID(activations_subset[block_num], 256)
                    total_intrinsic_dim[block_num] += torch.Tensor([float(intrinsic_dim)]).to(model.device)
                    #total_identity_measures[block_num] += identity_measure[block_num]
                
                total_cka_measures += cka_measure
                total_r2cca_measures += r2cca_measure

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(total_num)
            dist.all_reduce(total_top_1)
            dist.all_reduce(total_top_5)
            #dist.all_reduce(total_cka_measures)
            #dist.all_reduce(total_r2cca_measures)

        avg_top_1 = float(total_top_1.item() / total_num.item())
        avg_top_5 = float(total_top_5.item() / total_num.item())
        avg_cka_measure = total_cka_measures / len(dataloader_test)
        avg_r2cca_measure = total_r2cca_measures / len(dataloader_test)

    avg_identity_measure, avg_intrinsic_dim = dict(), dict()
    for block_num in total_identity_measures.keys():
        avg_identity_measure[block_num] = float(total_identity_measures[block_num].item() / len(dataloader_test))
        avg_intrinsic_dim[block_num] = float(total_intrinsic_dim[block_num].item() / len(dataloader_test))

    return avg_top_1, avg_top_5, avg_identity_measure, avg_intrinsic_dim, avg_cka_measure, avg_r2cca_measure


def validate_ood_datasets(ood_dataset_names, batch_size, model, similarity_measure, num_workers, num_gpus_to_use, rank):
    test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(32, 32)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(
                mean=lightly.data.collate.imagenet_normalize['mean'],
                std=lightly.data.collate.imagenet_normalize['std'],
            )
        ])

    all_avg_top_1 = dict()
    all_avg_top_5 = dict()
    if "cifar-10-2" in ood_dataset_names:
        sub_dataset_unprocessed = Cifar10_2_Dataset()
        sub_dataset = LightlyDataset.from_torch_dataset(dataset=sub_dataset_unprocessed, transform=test_transforms)
        sub_dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset=sub_dataset, num_replicas=num_gpus_to_use, rank=rank)
        sub_dataset_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=num_workers, sampler=sub_dataset_sampler)

        avg_top_1, avg_top_5, _, _ = validate(sub_dataset_loader, model, similarity_measure)
        all_avg_top_1["cifar-10-2"] = {"cifar-10-2": avg_top_1}
        all_avg_top_5["cifar-10-2"] = {"cifar-10-2": avg_top_5}

    if "cifar-10-c" in ood_dataset_names:
        dataset_obj = Cifar_10_C()
        all_avg_top_1["cifar-10-c"] = dict()
        all_avg_top_5["cifar-10-c"] = dict()
        sub_dataset_names = dataset_obj.get_dataset_names()
        test_data = torchvision.datasets.CIFAR10("datasets/", train=False)
        for sub_dataset_name in sub_dataset_names:
            sub_dataset_unprocessed = dataset_obj.get_sub_dataset(test_data, sub_dataset_name)
            sub_dataset = LightlyDataset.from_torch_dataset(dataset=sub_dataset_unprocessed, transform=test_transforms)
            sub_dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset=sub_dataset, num_replicas=num_gpus_to_use, rank=rank)
            sub_dataset_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=num_workers, sampler=sub_dataset_sampler)

            avg_top_1, avg_top_5, _, _ = validate(sub_dataset_loader, model, similarity_measure)
            all_avg_top_1["cifar-10-c"][sub_dataset_name] = avg_top_1
            all_avg_top_5["cifar-10-c"][sub_dataset_name] = avg_top_5
    return all_avg_top_1, all_avg_top_5


def train(dataloader_train, model, model_wrapped, similarity_measure, optimiser, scheduler, sim_scale, use_measure_loss=False):
    # paired augmented samples, labels, file name
    total_train_loss, total_num = 0, torch.tensor(0, device=model.device)
    #layer_coifficients = {block_num: 0. for block_num in range(similarity_measure.num_blocks)}

    total_measure_losses = {block_num: 0 for block_num in range(similarity_measure.num_blocks)}
    total_scaled_losses = {block_num: 0 for block_num in range(similarity_measure.num_blocks)}

    for train_batch in dataloader_train:
        
        num, activations, loss = model_wrapped(train_batch)

        '''Want to eventually have so that train linear map 5 times -> backprop measure loss each time but not scaled loss'''

        measure_losses, scaled_losses = similarity_measure.similarity(activations)
        #print("measure", measure_losses)
        #print("scaled", scaled_losses)

        if use_measure_loss:
            scaled_loss = sum(scaled_losses.values())
            loss += sim_scale * scaled_loss

        for block_num in total_measure_losses.keys():
            total_measure_losses[block_num] += measure_losses[block_num].detach()
            total_scaled_losses[block_num] += scaled_losses[block_num].detach()

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(loss)

        total_train_loss += loss.detach()

        total_num += torch.tensor(num, device=model.device)
        loss.backward()

        optimiser.step()
        optimiser.zero_grad()
        scheduler.step()
    
    model.training_epoch_end(None)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total_num)
        dist.all_reduce(total_train_loss)

    avg_train_loss = float(total_train_loss.item() / total_num.item())

    avg_measure_losses, avg_scaled_losses = dict(), dict()
    for block_num in measure_losses.keys():
        avg_measure_losses[block_num] = float(total_measure_losses[block_num].item() / len(dataloader_train))
        avg_scaled_losses[block_num] = float(total_scaled_losses[block_num].item() / len(dataloader_train))

    return avg_train_loss, avg_measure_losses, avg_scaled_losses#, activations


def main_worker(rank, num_gpus_to_use, model_class, log_dir, params_file, parser, sim_scale):
    logger = SummaryWriter(log_dir=log_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if num_gpus_to_use > 1:
        torch.distributed.init_process_group(backend='nccl', world_size=num_gpus_to_use, rank=rank)
        n_gpus_total = torch.distributed.get_world_size()
        print("Number of GPUs Being Used:    ", n_gpus_total)

    dataloader_train, dataloader_val, dataloader_test, num_classes = get_data_loaders(parser, num_gpus_to_use, rank)

    if num_gpus_to_use:
        torch.cuda.set_device(rank)
        device = torch.device('cuda:%d' % rank)

    model = model_class(dataloader_val, num_classes, parser["image_size"]).to(device)
    model_wrapped = model

    if num_gpus_to_use > 1:
        model_wrapped = DDP(model, device_ids=[rank], find_unused_parameters=True)
        model_wrapped = nn.SyncBatchNorm.convert_sync_batchnorm(model_wrapped)

    optimiser, scheduler = model.configure_optimizers(parser["batch_size"], parser["lr_scaler"], parser["epochs"], parser["weight_decay"])

    model, optimiser, scheduler, start_epoch = load_params(parser, logger, model, optimiser, scheduler)
    logger.add_hparams({'Batch Size': parser["batch_size"], 'Weight Decay Coifficient': parser["weight_decay"], 'Comparison Method': parser["comparison_method"], 'Number Of GPUs': num_gpus_to_use}, dict())
    logger.flush()

    similarity_measure = SimilarityMeasure(use_measure_loss=parser["measure_loss"], measure_type=parser["comparison_method"], kernel_type=parser["kernel_type"], device=device)

    for epoch in range(start_epoch, parser["epochs"]):
        train_loss, measure_losses, scaled_losses = train(dataloader_train, model, model_wrapped, similarity_measure, optimiser, scheduler, sim_scale, parser["measure_loss"])
        print(f"TRAIN: epoch: {epoch:>02}, loss: {train_loss:.5f}")

        if not epoch % parser["val_every"]:
            train_top_1, train_top_5, avg_identity, intrinsic_dim, cka_similarity, r2cca_similarity = validate(dataloader_train, model, similarity_measure, collect_state=True)
            val_top_1, val_top_5, _, _, _, _ = validate(dataloader_val, model, similarity_measure)
            test_top_1, test_top_5, _, _, _, _ = validate(dataloader_test, model, similarity_measure)
            ood_top_1, ood_top_5 = validate_ood_datasets(parser["ood_datasets"], parser["batch_size"], model, similarity_measure, parser["num_workers"], num_gpus_to_use, rank)

            print(f"Train: epoch: {epoch:>02}, Top-1: {train_top_1}, Top-5: {train_top_5}")
            print(f"Val: epoch: {epoch:>02}, Top-1: {val_top_1}, Top-5: {val_top_5}")
            print(f"Test: epoch: {epoch:>02}, Top-1: {test_top_1}, Top-5: {test_top_5}")
        
        if not epoch % parser["save_every"] and not rank:
            save_params(log_dir, epoch, model, optimiser, scheduler, parser["batch_size"], params_file)
            log_state(logger, epoch, intrinsic_dim, cka_similarity, r2cca_similarity, avg_identity, scheduler)
            log_accuracy(logger, epoch, train_top_1, val_top_1, test_top_1, train_top_5, val_top_5, test_top_5, ood_top_1, ood_top_5)
            log_loss(logger, epoch, train_loss, measure_losses, scaled_losses)

    print()
    clean_up(model_wrapped, num_gpus_to_use)
    

def main():
    # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
    # If multiple runs are specified a subdirectory for each run is created.
    parser, sim_scale = parse_args()
    model_class = getattr(sys.modules[__name__], parser["model_name"])
    current_time = str(datetime.now()).replace(' ', '-')

    log_dir = f'runs/{parser["model_name"]}-{parser["dataset_name"]}-{parser["comparison_method"]}-{current_time}-{parser["kernel_type"]}-{sim_scale}/logs'
    params_file = f'runs/{parser["model_name"]}-{parser["dataset_name"]}-{parser["comparison_method"]}-{current_time}-{parser["kernel_type"]}-{sim_scale}/params.ckpt'
    
    if parser["distributed"]:
        num_gpus_to_use = min(num_gpus, 2)
        torch.multiprocessing.spawn(main_worker, nprocs=num_gpus_to_use, args=(num_gpus_to_use, model_class, log_dir, params_file, parser, sim_scale))

    else:
        # limit to single gpu if not using distributed training
        num_gpus_to_use = min(num_gpus, 1)
        main_worker(0 if num_gpus_to_use else None, num_gpus_to_use, model_class, log_dir, params_file, parser, sim_scale)

      
if __name__ == "__main__":
    main()
