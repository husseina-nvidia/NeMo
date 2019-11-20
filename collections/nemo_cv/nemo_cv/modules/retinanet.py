# Copyright (C) husseina, NVIDIA Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Hussein Al-barazanchi"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torchvision.models import resnet as vrn

import os.path
import io
import math

from . import backbones as backbones_mod
from ._C import Engine
from .box import generate_anchors, snap_to_anchors, decode, nms

from nemo.backends.pytorch.nm import TrainableNM

from nemo.core import NeuralType, AxisType, DeviceType,\
    BatchTag, ChannelTag, HeightTag, WidthTag, ListTag, BoundingBoxTag, \
    LogProbabilityTag

"""
Alot of the code below is heavily borrowed from 
https://github.com/NVIDIA/retinanet-examples
"""

class FocalLoss(nn.Module):
    'Focal Loss - https://arxiv.org/abs/1708.02002'

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target):
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1,  pred, 1 - pred)
        return alpha * (1. - pt) ** self.gamma * ce


class SmoothL1Loss(nn.Module):
    'Smooth L1 Loss'

    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class ResNet(vrn.ResNet):
    'Deep Residual Network - https://arxiv.org/abs/1512.03385'

    def __init__(self, layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, outputs=[5], url=None):
        self.stride = 128        
        self.bottleneck = bottleneck
        self.outputs = outputs
        self.url = url
        super().__init__(bottleneck, layers)

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            level = i + 2
            if level > max(self.outputs):
                break
            x = layer(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs


class FPN(nn.Module):
    'Feature Pyramid Network - https://arxiv.org/abs/1612.03144'

    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features

        is_light = features.bottleneck == vrn.BasicBlock
        channels = [128, 256, 512] if is_light else [512, 1024, 2048]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.apply(init_layer)

        self.features.initialize()

    def forward(self, x):
        c3, c4, c5 = self.features(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, scale_factor=2) + p3

        p6 = self.pyramid6(c5)
        p7 = self.pyramid7(F.relu(p6))

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return [p3, p4, p5, p6, p7]


class Model(nn.Module):
    'RetinaNet - https://arxiv.org/abs/1708.02002'

    def __init__(self, backbones='ResNet18FPN', classes=80, config={}):
        super().__init__()

        if not isinstance(backbones, list):
            backbones = [backbones]

        #self.backbones = nn.ModuleDict({b: getattr(backbones_mod, b)() for b in backbones})
        self.backbones = FPN(ResNet(layers=[2, 2, 2, 2], bottleneck=vrn.BasicBlock, outputs=[3, 4, 5], url=vrn.model_urls['resnet18']))
        self.name = 'RetinaNet'
        self.exporting = False

        self.ratios = [1.0, 2.0, 0.5]
        self.scales = [4 * 2**(i/3) for i in range(3)]
        self.anchors = {}
        self.classes = classes

        self.threshold  = config.get('threshold', 0.05)
        self.top_n      = config.get('top_n', 1000)
        self.nms        = config.get('nms', 0.5)
        self.detections = config.get('detections', 100)

        self.stride = max([b.stride for _, b in self.backbones.items()])

        # classification and box regression heads
        def make_head(out_size):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)

        anchors = len(self.ratios) * len(self.scales)
        self.cls_head = make_head(classes * anchors)
        self.box_head = make_head(4 * anchors)

        self.cls_criterion = FocalLoss()
        self.box_criterion = SmoothL1Loss(beta=0.11)

    def __repr__(self):
        return '\n'.join([
            '     model: {}'.format(self.name),
            '  backbone: {}'.format(', '.join([k for k, _ in self.backbones.items()])),
            '   classes: {}, anchors: {}'.format(self.classes, len(self.ratios) * len(self.scales)),
        ])

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbone(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)

        # Initialize class head prior
        def initialize_prior(layer):
            pi = 0.01
            b = - math.log((1 - pi) / pi)
            nn.init.constant_(layer.bias, b)
            nn.init.normal_(layer.weight, std=0.01)
        self.cls_head[-1].apply(initialize_prior)

    def forward(self, x):
        if self.training: x, targets = x

        # Backbones forward pass
        features = []
        for _, backbone in self.backbones.items():
            features.extend(backbone(x))

        # Heads forward pass
        cls_heads = [self.cls_head(t) for t in features]
        box_heads = [self.box_head(t) for t in features]

        if self.training:
            return self._compute_loss(x, cls_heads, box_heads, targets.float())

        cls_heads = [cls_head.sigmoid() for cls_head in cls_heads]

        if self.exporting:
            self.strides = [x.shape[-1] // cls_head.shape[-1] for cls_head in cls_heads]
            return cls_heads, box_heads

        # Inference post-processing
        decoded = []
        for cls_head, box_head in zip(cls_heads, box_heads):
            # Generate level's anchors
            stride = x.shape[-1] // cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)

            # Decode and filter boxes
            decoded.append(decode(cls_head, box_head, stride,
                self.threshold, self.top_n, self.anchors[stride]))

        # Perform non-maximum suppression
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        return nms(*decoded, self.nms, self.detections)

    def _extract_targets(self, targets, stride, size):
        cls_target, box_target, depth = [], [], []
        for target in targets:
            target = target[target[:, -1] > -1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)
            snapped = snap_to_anchors(
                target, [s * stride for s in size[::-1]], stride,
                self.anchors[stride].to(targets.device), self.classes, targets.device)
            for l, s in zip((cls_target, box_target, depth), snapped): l.append(s)
        return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)

    def _compute_loss(self, x, cls_heads, box_heads, targets):
        cls_losses, box_losses, fg_targets = [], [], []
        for cls_head, box_head in zip(cls_heads, box_heads):
            size = cls_head.shape[-2:]
            stride = x.shape[-1] / cls_head.shape[-1]

            cls_target, box_target, depth = self._extract_targets(targets, stride, size)
            fg_targets.append((depth > 0).sum().float().clamp(min=1))

            cls_head = cls_head.view_as(cls_target).float()
            cls_mask = (depth >= 0).expand_as(cls_target).float()
            cls_loss = self.cls_criterion(cls_head, cls_target)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())

            box_head = box_head.view_as(box_target).float()
            box_mask = (depth > 0).expand_as(box_target).float()
            box_loss = self.box_criterion(box_head, box_target)
            box_loss = box_mask * box_loss
            box_losses.append(box_loss.sum())

        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        box_loss = torch.stack(box_losses).sum() / fg_targets
        return cls_loss, box_loss

    def save(self, state):
        checkpoint = {
            'backbone': [k for k, _ in self.backbones.items()],
            'classes': self.classes,
            'state_dict': self.state_dict()
        }

        for key in ('iteration', 'optimizer', 'scheduler'):
            if key in state:
                checkpoint[key] = state[key]

        torch.save(checkpoint, state['path'])

    @classmethod
    def load(cls, filename):
        if not os.path.isfile(filename):
            raise ValueError('No checkpoint {}'.format(filename))

        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        # Recreate model from checkpoint instead of from individual backbones
        model = cls(backbones=checkpoint['backbone'], classes=checkpoint['classes'])
        model.load_state_dict(checkpoint['state_dict'])

        state = {}
        for key in ('iteration', 'optimizer', 'scheduler'):
            if key in checkpoint:
                state[key] = checkpoint[key]

        del checkpoint
        torch.cuda.empty_cache()

        return model, state

    def export(self, size, batch, precision, calibration_files, calibration_table, verbose, onnx_only=False, opset=None):
        import torch.onnx.symbolic

        if opset is not None and opset < 9:
            # Override Upsample's ONNX export from old opset if required (not needed for TRT 5.1+)
            @torch.onnx.symbolic.parse_args('v', 'is')
            def upsample_nearest2d(g, input, output_size):
                height_scale = float(output_size[-2]) / input.type().sizes()[-2]
                width_scale = float(output_size[-1]) / input.type().sizes()[-1]
                return g.op("Upsample", input,
                    scales_f=(1, 1, height_scale, width_scale),
                    mode_s="nearest")
            torch.onnx.symbolic.upsample_nearest2d = upsample_nearest2d

        # Export to ONNX
        print('Exporting to ONNX...')
        self.exporting = True
        onnx_bytes = io.BytesIO()
        zero_input = torch.zeros([1, 3, *size]).cuda()
        extra_args = { 'opset_version': opset } if opset else {}
        torch.onnx.export(self.cuda(), zero_input, onnx_bytes, *extra_args)
        self.exporting = False

        if onnx_only:
            return onnx_bytes.getvalue()

        # Build TensorRT engine
        model_name = '_'.join([k for k, _ in self.backbones.items()])
        anchors = [generate_anchors(stride, self.ratios, self.scales).view(-1).tolist() 
            for stride in self.strides]
        return Engine(onnx_bytes.getvalue(), len(onnx_bytes.getvalue()), batch, precision,
            self.threshold, self.top_n, anchors, self.nms, self.detections, calibration_files, model_name, calibration_table, verbose)


class RetinaNet(TrainableNM):
    """
        Wrapper class around the RetinaNet model.
    """

    @staticmethod
    def create_ports():
        input_ports = {
            # Batch of images.
            "images": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(ChannelTag, 3),
                                  2: AxisType(HeightTag),
                                  3: AxisType(WidthTag)}),
            # Batch of bounding boxes.
            "bounding_boxes": NeuralType({0: AxisType(BatchTag),
                                          1: AxisType(ListTag),
                                          2: AxisType(BoundingBoxTag)}),
            # Batch of targets.
            "targets": NeuralType({0: AxisType(BatchTag)})
        }
        output_ports = {
            "predictions": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(LogProbabilityTag)
                                       })

        }
        return input_ports, output_ports

    def __init__(self, num_classes, pretrained=False):
        """
        Creates the Faster R-CNN model.

        Args:
            num_classes: Number of output classes of the model.
            pretrained: use weights of model pretrained on COCO train2017.
        """

        super().__init__()

        # Create
        self.model = FPN(ResNet(layers=[2, 2, 2, 2], bottleneck=vrn.BasicBlock, outputs=[3, 4, 5], url=vrn.model_urls['resnet18']))

        # Get number of input features for the classifier.
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self.to(self._device)

    def forward(self, images, bounding_boxes, targets):
        """
        Performs the forward step of the model.

        Args:
            images: Batch of images to be classified.
        """

        # We need to put this in a tuple again, as OD "framework" assumes it :]
        targets_tuple = [{"boxes": b, "labels": t} for b, t
                         in zip(bounding_boxes, targets)]

        predictions = self.model(images, targets_tuple)
        return predictions
