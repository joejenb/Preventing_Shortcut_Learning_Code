import lightly
from models.modified_benchmark import BenchmarkModule
from lightly.models.resnet import BasicBlock

import torch
import torch.nn as nn


class ResNet18(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes, image_size, device="cpu", layers_to_hook = [2, 3, 4, 5]):
        super().__init__(dataloader_kNN, num_classes)
        if image_size == 32:
            linear_size = 512
        if image_size == 48:
            linear_size = 512

        self.layers_to_hook = layers_to_hook
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        resnet_blocks = list(resnet.children())[:-1]
        resnet_blocks.append(nn.AdaptiveAvgPool2d(1))
        self.backbone = nn.ModuleList(resnet_blocks)


        self.prediction_head = nn.Sequential(
            nn.Linear(linear_size, num_classes),
        )
        
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
   
    def forward(self, batch):
        #x, target, _ = batch
        (x, _), target, _ = batch
        x = x.to(self.device)
        target = target.to(self.device)

        activations, activation_num = dict(), 0
        block_input = x
        for block_num, block in enumerate(self.backbone):
            block_output = block(block_input)
            if block_num in self.layers_to_hook:
                activations[activation_num] = block_output
                activation_num += 1
            block_input = block_output

        repres = block_output.flatten(start_dim=1)

        pred = self.prediction_head(repres)

        loss = self.criterion(pred, target)
        return x.size()[0], activations, loss

    def validation_step(self, batch, collect_state):
        # we can only do kNN predictions once we have a feature bank
        x, target, _ = batch
        x = x.to(self.device)
        target = target.to(self.device)
        
        activations, activation_num = dict(), 0
        block_input = x
        for block_num, block in enumerate(self.backbone):
            if sum(1 for _ in block.children()):
                sub_block_outputs = []
                for sub_block_num, sub_block in enumerate(block.children()):
                    block_output = sub_block(block_input)
                    sub_block_outputs.append(block_output)
                    block_input = block_output
                
                if isinstance(sub_block, (nn.Conv2d, BasicBlock)) and collect_state:
                    activations[block_num] = sub_block_outputs
                    activation_num += 1
            else:
                block_output = block(block_input)
                if isinstance(block, (nn.Conv2d, BasicBlock)) and collect_state:
                    activations[block_num] = [block_output]
                    activation_num += 1
                block_input = block_output

        repres = block_output.flatten(start_dim=1)
        pred = self.prediction_head(repres)

        _, pred_1 = torch.max(pred, 1)
        _, pred_5 = torch.topk(pred, min(5, self.num_classes), dim=1)

        top_1 = (pred_1.unsqueeze(dim=1)[:, 0] == target).float().sum()
        top_5 = 0
        for ind in range(min(5, self.num_classes)):
            top_5 += (pred_5[:, ind] == target).float().sum()

        return x.size()[0], activations, top_1, top_5

    def configure_optimizers(self, batch_size, lr_scaler, epochs, weight_decay):
        params = list(self.backbone.parameters())#[block.parameters() for block in self.backbone]#
        optim = torch.optim.SGD(
            params, 
            lr=6e-2 * batch_size / lr_scaler,
            momentum=0.9, 
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
        return optim, scheduler

