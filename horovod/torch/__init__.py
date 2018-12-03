# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.common import check_extension

try:
    check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                    __file__, 'mpi_lib_v2')
except:
    check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                    __file__, 'mpi_lib', '_mpi_lib')

from horovod.torch.compression import Compression
from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
from horovod.torch.mpi_ops import allgather, allgather_async
from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
from horovod.torch.mpi_ops import poll, synchronize
from horovod.torch.mpi_ops import init, shutdown
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import mpi_threads_supported
from mpi4py import MPI

import torch
import numpy as np

import collections

comm = MPI.COMM_WORLD

def sparse_allreduce(sparse_tensor, storage):
    nnzs_1d = storage['nnzs_1d']
    values_1d = storage['values_1d']
    indexes_1d = storage['indexes_1d']
    displ_1d = storage['displ_1d']
    tensor = sparse_tensor
    index = np.flatnonzero(tensor).astype(np.int32)
    if len(index) == 0:
        index = np.array([0], dtype=np.int32)
    
    nnz = len(index)

    # Tell others how many nnzs
    comm.Allgather(np.array(nnz, dtype=np.int32), nnzs_1d)
    comm.Barrier()
    #print('[rank:%d] nnzs_1d: %s' %(rank(), nnzs_1d))

    for i, nnz in enumerate(nnzs_1d[0:-1]):
        displ_1d[i+1] = displ_1d[i] + nnz
    #print('[rank:%d] displ_1d: %s' %(rank(), displ_1d))
    #print('[rank:%d] tensor: %s' %(rank(), tensor[index]))
    #print('[rank:%d] index: %s' %(rank(), index))
    comm.Allgatherv(tensor[index], [values_1d, nnzs_1d, displ_1d, MPI.FLOAT])
    #print('[rank:%d] allgather values: %s' %(rank(), values_1d))
    comm.Allgatherv(index, [indexes_1d, nnzs_1d, displ_1d, MPI.INT])
    #print('[rank:%d] allgather index: %s' %(rank(), indexes_1d))
    comm.Barrier()

    result = storage['result']
    result.fill(0)
    #print('[rank:%d] len of reuslt: %d' %(rank(), len(result)))
    for i in range(size()):
        nnz = nnzs_1d[i]
        displ = displ_1d[i]
        index = indexes_1d[displ:displ+nnz]
        result[index] += values_1d[displ:displ+nnz]

    return result


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression, is_sparse=False):
        super(self.__class__, self).__init__(params)
        self._compression = compression
        self._sparse = is_sparse

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        if len(named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                     in sorted(named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'allreduce.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}

        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self.local = False
        if size() > 1:
            self._register_hooks()


    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

        if self._sparse:
            self._init_sparse_space()

    def _allreduce_grad_async(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad.data
        tensor_compressed, ctx = self._compression.compress(tensor, name)
        handle = allreduce_async_(tensor_compressed, average=True, name=name)
        return handle, ctx

    def _sparse_allreduce(self, p):
        name = self._parameter_names.get(p)
        #print('[rank:%d] sparse allreduce: %s' %(rank(), name))
        tensor = p.grad.data
        tensor_compressed, ctx = self._compression.compress(tensor, name)
        if tensor_compressed.is_cuda:
            ny = tensor_compressed.cpu().numpy()
        else:
            ny = tensor_compressed.numpy()
        shape = ny.shape
        result = sparse_allreduce(ny.flatten(), self._sparse_storage[p])
        result = result.reshape(shape)
        r = torch.from_numpy(result)
        if tensor_compressed.is_cuda:
            r = r.cuda(tensor.device, non_blocking=False)
        #if rank() == 0:
            #print('[rank:%d] == name:%s, norm: %s' %(rank(), name, float(r.norm())))
        return r 


    def _make_hook(self, p):
        def hook(*ignore):
            assert p not in self._handles
            assert not p.grad.requires_grad
            if not self.local:
                if self._sparse:
                    self._handles[p] = self._sparse_allreduce(p)
                else:
                    handle, ctx = self._allreduce_grad_async(p)
                    self._handles[p] = (handle, ctx)
        return hook

    def _init_sparse_space(self):
        def __new_storage(numel, num_workers):
            num_elements_of_tensor = numel
            storage = {}
            storage['nnzs_1d']      = np.zeros(num_workers, dtype=np.int32)
            storage['values_1d']    = np.zeros(num_elements_of_tensor * num_workers, dtype=np.float32)
            storage['indexes_1d']   = np.zeros(num_elements_of_tensor * num_workers, dtype=np.int32)
            storage['displ_1d']     = np.zeros(num_workers, dtype=np.int32)
            storage['result']       = np.zeros(num_elements_of_tensor, dtype=np.float32)
            return storage

        self._sparse_storage = {}
        num_workers = size()
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    name = self._parameter_names.get(p)
                    numel = p.grad.numel()
                    #print('name: %s, type: %s, numel: %d', name, p.grad.dtype, numel)
                    self._sparse_storage[p] = __new_storage(numel, num_workers)
                    #print('[rank:%d] [name:%s] init len of reuslt: %d' %(rank(), name, numel))

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            if self._sparse:
                self._handles[p] = self._sparse_allreduce(p)
            else:
                handle, ctx = self._allreduce_grad_async(p)
                self._handles[p] = (handle, ctx)

        for p, value in self._handles.items():
            name = self._parameter_names.get(p)
            if self._sparse:
                output = value
                #if rank() == 0:
                    #print('[rank:%d] -- name:%s, norm: %s' %(rank(), name, float(output.norm())))
                p.grad.data.set_(self._compression.decompress(output, None, name=name))
            else:
                handle, ctx = value
                output = synchronize(handle)
                p.grad.data.set_(self._compression.decompress(output, ctx, name=name))

        self._handles.clear()

    def step(self, closure=None):
        if not self.local:
            self.synchronize()
        return super(self.__class__, self).step(closure)



def DistributedOptimizer(optimizer, named_parameters=None, compression=Compression.none, is_sparse=False):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.

    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls(optimizer.param_groups, named_parameters, compression, is_sparse)


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        handle = broadcast_async_(p, root_rank, name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()
