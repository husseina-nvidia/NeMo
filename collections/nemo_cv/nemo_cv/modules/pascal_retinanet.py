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


import sys
from retinanet_module import *


def main(args=None):
    'Entry point for the retinanet command'

    args = parse(args or sys.argv[1:])

    detector = RetinaNet(args)
    detector.execute(args)

if __name__ == '__main__':
    main()
