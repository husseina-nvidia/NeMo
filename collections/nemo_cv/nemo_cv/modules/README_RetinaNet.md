
# The docker image that is tested working with RetinaNet is nvcr.io/nvidia/pytorch:19.09-py3

# To run RetinaNet Nemo Wrapper follow the below steps:

docker run -it --rm --ipc=host --gpus all -v {path to the project}:/workspace nvcr.io/nvidia/pytorch:19.09-py3

pip install --no-cache-dir git+https://github.com/nvidia/retinanet-examples

git clone https://github.com/NVIDIA/NeMo.git 

cd NeMo/nemo
# Change the version of Pytorch in Nemo/nemo/setup.py to 'torch==1.2.0'

python setup install

cd ..
cd collections/nemo_cv/nemo_cv/modules/

python pascal_retinanet.py train model_mydataset.pth --backbone ResNet18FPN --classes 20 --iters 10000 --val-iters 1000 --lr 0.0005 --resize 512 --jitter 480 640 --images /workspace/PASCAL_VOC/JPEGImages/ --annotations /workspace/PASCAL_VOC/pascal_train2012.json --val-annotations /workspace/PASCAL_VOC/pascal_val2012.json
