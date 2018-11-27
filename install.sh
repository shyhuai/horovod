#HOROVOD_NCCL_HOME=/home/comp/csshshi/local/nccl2.2.12 HOROVOD_GPU_ALLREDUCE=NCCL python setup.py install 
#HOROVOD_NCCL_HOME=/home/comp/csshshi/local/nccl2.2.12 HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
cp horovod/torch/__init__.py /home/comp/csshshi/anaconda2/lib/python2.7/site-packages/horovod-0.15.1-py2.7-linux-x86_64.egg/horovod/torch/
