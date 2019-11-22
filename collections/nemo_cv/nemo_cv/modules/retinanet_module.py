# Copyright (C) , NVIDIA INC. All Rights Reserved.
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

from retinanet.model import Model
from retinanet.main import parse, load_model, worker

from nemo.backends.pytorch.nm import TrainableNM


class RetinaNet(TrainableNM):
    """
        Wrapper class around the RetinaNet model.
    """

    @staticmethod
    def create_ports():
        
        return None, None

    def __init__(self, args):
        """
        Creates the RetinaNet model.

        Args:
            num_classes: Number of output classes of the model.
            pretrained: use weights of model pretrained on COCO train2017.
        """

        super().__init__()

        # Create
        self.model, self.state = load_model(args, verbose=True)
        if self.model: 
            self.model.share_memory()

    def forward(self, images, bounding_boxes, targets):
        pass

    def execute(self, args):

        world = torch.cuda.device_count()
        if args.command == 'export' or world <= 1:
            worker(0, args, 1, self.model, self.state)
        else:
            torch.multiprocessing.spawn(worker, args=(args, world, self.model, self.state), nprocs=world)
