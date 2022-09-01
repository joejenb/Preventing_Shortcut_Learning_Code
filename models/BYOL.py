import lightly
from lightly.models.modules import heads
from lightly.models import utils
from lightly.utils import knn_predict
from lightly.models.resnet import BasicBlock
from models.modified_benchmark import BenchmarkModule

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class BYOL(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes, image_size, device="cpu", layers_to_hook = [2, 3, 4, 5]):
        super().__init__(dataloader_kNN, num_classes)

        if image_size == 32:
            linear_size = 512
        elif image_size == 48:
            linear_size = 512

        self.layers_to_hook = layers_to_hook
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        resnet_blocks = list(resnet.children())[:-1]
        resnet_blocks.append(nn.AdaptiveAvgPool2d(1))
        self.backbone = nn.ModuleList(resnet_blocks)

        # create a byol model based on ResNet
        self.projection_head = heads.BYOLProjectionHead(linear_size, 1024, 256)
        self.prediction_head = heads.BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.NegativeCosineSimilarity()
        #Only want to maximise covariance of those that are the equivalent representation 
        #For this case know are the same so want to maximise maximum correlation for each dimension/representation
        #Could just maximise diagonal as should have exactly equally properties i.e whole representation should be ordered the same
   
    def project(self, x):
        activations, activation_num = dict(), 0
        block_input = x
        for block_num, block in enumerate(self.backbone):
            block_output = block(block_input)
            if block_num in self.layers_to_hook:
                activations[activation_num] = block_output
                activation_num += 1
            block_input = block_output

        y = block_output.flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p, activations

    def forward_momentum(self, x):
        block_input = x
        for block_num, block in enumerate(self.backbone):
            block_output = block(block_input)
            block_input = block_output

        y = block_output.flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def forward(self, batch):
        (x0, x1), target, _ = batch
        x0, x1 = x0.to(self.device), x1.to(self.device)
        target = target.to(self.device)

        utils.update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)

        p0, activations_p0 = self.project(x0)
        z0 = self.forward_momentum(x0)
        p1, _ = self.project(x1)
        z1 = self.forward_momentum(x1)

        #Prediction head does not exist for identifying equivalence -> rather to map back to old state
        #Equivalence should then exist between old projection and predicted old projection
        #Covariance is effectively normalised dot product
        #Loss given by BYOL is effectively correlation between units -> in this case want this to be high 
        #As whole representation should be the same -> Should be equivalent to svca though if components are representations
        #Only other thing is to try with non linear kernel
        #Might be that same representations are present but with varying/inaccurate quantities so require vmat

        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        return x0.size()[0], activations_p0, loss

    def validation_step(self, batch, collect_state):
        # we can only do kNN predictions once we have a feature bank
        x0, target, _ = batch
        if isinstance(x0, list):
            x0 = x0[0]

        x0 = x0.to(self.device)
        target = target.to(self.device)
        
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            activations, activation_num = dict(), 0
            block_input = x0
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

            feature = block_output.squeeze()
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(
                feature,
                self.feature_bank,
                self.targets_bank,
                self.num_classes,
                self.knn_k,
                self.knn_t
            )

            top_1 = (pred_labels[:, 0] == target).float().sum()
            top_5 = 0
            for ind in range(min(5, self.num_classes)):
                top_5 += (pred_labels[:, ind] == target).float().sum()

            return x0.size()[0], activations, top_1, top_5

    def configure_optimizers(self, batch_size, lr_scaler, epochs, weight_decay):
        params = list(self.backbone.parameters()) \
            + list(self.projection_head.parameters()) \
            + list(self.prediction_head.parameters())
        optim = torch.optim.SGD(
            params, 
            lr=6e-2 * batch_size / lr_scaler,
            momentum=0.9, 
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
        return optim, scheduler

