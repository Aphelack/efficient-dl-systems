import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Any, NamedTuple
import gc

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.profiler import record_function
from torch.distributed.device_mesh import DeviceMesh, _get_device_handle
from torch.distributed.tensor import Shard, DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.distributed.tensor import Replicate, Partial
from torch.distributed import ReduceOp


logger = logging.getLogger(__name__)

torch.use_deterministic_algorithms(True)
cls_to_fsdp_cls: dict[type, type] = {}


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


class TrainingState(Enum):
    FORWARD = auto()
    PRE_BACKWARD = auto()
    POST_BACKWARD = auto()
    IDLE = auto()


class ShardedState(Enum):
    SHARDED = auto()
    UNSHARDED = auto()


class FSDPParam:
    orig_dtype: torch.dtype
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    orig_size: torch.Size
    sharded_size: torch.Size
    sharded_param: nn.Parameter
    unsharded_param: nn.Parameter
    _sharding_spec: DTensorSpec

    def __init__(
        self,
        param: nn.Parameter,
        module: nn.Module,
        param_name: str,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy,
        param_fqn: str,
    ):
        self._module = module
        self._param_name = param_name
        self.mesh = mesh
        self._param_fqn = param_fqn
        self._init_sharded_param(param)
        self._init_dtype_attrs(mp_policy)
        self._init_unsharded_param()

    @torch.no_grad()
    def _init_sharded_param(self, param: nn.Parameter):
        fsdp_placement = Shard(0)
        shard_dim = fsdp_placement.dim
        self._sharding_spec = DTensorSpec(
            self.mesh,
            (fsdp_placement,),
            tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
        )
        self.orig_size = param.size()
        shard_rank = self.mesh.get_local_rank()
        shard_world_size = self.mesh.size()

        assert param.size(shard_dim) % shard_world_size == 0
        # TODO(task1): shard the full `param` into `sharded_param`
        sharded_param = param.data.chunk(shard_world_size, dim=shard_dim)[shard_rank].clone()
        self.sharded_size = sharded_param.size()
        self.sharded_param = nn.Parameter(
            self.to_sharded_dtensor(sharded_param),
            requires_grad=param.requires_grad,
        )
        self._setattr_on_module(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def _init_unsharded_param(self):
        self.unsharded_param = nn.Parameter(
            torch.empty(
                self.orig_size,
                dtype=self.param_dtype or self.orig_dtype,
                device=self.sharded_param.device,
            ),
            requires_grad=self.sharded_param.requires_grad,
        )
        self.free_unsharded_param()

    def _init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = mp_policy.param_dtype, mp_policy.reduce_dtype
        self.orig_dtype = self.sharded_param.dtype
        # Clamp `reduce_dtype` to `None` if no casting is required: since
        # gradients are computed in `param_dtype`, if `reduce_dtype` matches,
        # then we do not need extra casting
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        # Clamp `param_dtype` to `None` if no casting is required
        if param_dtype == self.orig_dtype:
            param_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        # None indicates that the mixed precision is not enabled

    def to_sharded(self) -> None:
        self._setattr_on_module(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def to_unsharded(self) -> None:
        # Assume that the data has been allocated and all-gathered
        self._setattr_on_module(self.unsharded_param)
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_module(self, param: nn.Parameter) -> None:
        unsafe_setattr_param(self._module, self._param_name, param)

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        if tensor.shape != self.sharded_size:
            raise AssertionError(
                f"Expects size {self.sharded_size} but got {tensor.shape}."
            )
        return DTensor.from_local(
            tensor,
            self._sharding_spec.mesh,
            self._sharding_spec.placements,
            shape=self._sharding_spec.shape,
            stride=self._sharding_spec.stride,
        )

    def alloc_unsharded_param(self) -> None:
        alloc_storage(self.unsharded_param)

    def free_unsharded_param(self) -> None:
        free_storage(self.unsharded_param)

    def __repr__(self):
        return f"FSDPParam(fqn={self._param_fqn}, orig_size={self.orig_size})"


def alloc_storage(tensor: torch.Tensor) -> None:
    size = tensor.numel() * tensor.itemsize
    if (storage := tensor.untyped_storage()).size() != size:
        storage.resize_(size)


def free_storage(tensor: torch.Tensor) -> None:
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)


# NOTE: These bypass `nn.Module.__setattr__` checks, which incur non-trivial
# CPU overhead, if the module did not override it. For FSDP, we know we do not
# need those checks when transitioning between sharded/unsharded parameters.
def unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = param
    else:  # slow path
        setattr(module, param_name, param)


class FSDPCommContext:
    def __init__(self, device_type: str):
        self.device_handle = _get_device_handle(device_type)
        high_priority = -1
        self.all_gather_stream = self.device_handle.Stream(priority=high_priority)
        self.reduce_scatter_stream = self.device_handle.Stream(priority=high_priority)
        # Post-forward order for explicit backward prefetching
        self.post_forward_order: list[FSDPModule] = []


class AllGatherResult(NamedTuple):
    param_all_gather_outputs: list[torch.Tensor]
    # or all_gather_output: torch.Tensor if you choose
    # to use a single all-gather per FSDPModule
    all_gather_event: torch.Event | None = None


def fully_shard(
    module: nn.Module,
    *,
    module_fqn: str,
    comm_ctx: FSDPCommContext,
    mesh: DeviceMesh,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
):
    if mesh.ndim != 1:
        raise ValueError(f"fully_shard expects a 1D DeviceMesh but got {mesh}.")

    device_handle = _get_device_handle(mesh.device_type)
    device = torch.device(mesh.device_type, device_handle.current_device())

    module.register_forward_pre_hook(pre_forward, prepend=True, with_kwargs=True)
    module.register_forward_hook(post_forward, prepend=False)

    module.to(device)

    module.fsdp_params = [
        FSDPParam(
            param,
            submodule,
            param_name,
            mesh,
            mp_policy,
            f"{f'{submodule_fqn}.' if submodule_fqn else ''}{param_name}",
        )
        for submodule_fqn, submodule in module.named_modules()
        for param_name, param in submodule.named_parameters(recurse=False)
    ]
    module._training_state = TrainingState.IDLE
    module._sharded_state = ShardedState.SHARDED
    module._module_fqn = module_fqn
    module.comm_ctx = comm_ctx
    module._post_forward_indices = []
    module._reshard_after_forward = reshard_after_forward
    module.reshard_after_backward = True
    module.reduce_grads = True
    module._all_gather_result = None
    module._post_reduce_event = None

    # Place FSDP leftmost for highest priority in the method resolution order
    cls = module.__class__
    new_cls = cls_to_fsdp_cls.get(cls, None)
    if not new_cls:
        new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), {})
        cls_to_fsdp_cls[cls] = new_cls
    module.__class__ = new_cls
    return module


class FSDPModule:
    fsdp_params: list[FSDPParam]
    _training_state: TrainingState
    _sharded_state: ShardedState
    _module_fqn: str
    comm_ctx: FSDPCommContext
    _post_forward_indices: list[int]
    _reshard_after_forward: bool
    reshard_after_backward: bool
    reduce_grads: bool
    _all_gather_result: AllGatherResult | None
    _post_reduce_event: torch.Event | None

    def __new__(cls, *args, **kwargs):
        # Use index 2 since 0 is the dynamically constructed `FSDP<...>` class
        # and index 1 is the `FSDPModule` class itself
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def unshard(self):
        if self._all_gather_result is not None:  # already called, pending wait
            return
        if self._sharded_state == ShardedState.UNSHARDED:
            return  # no-op
        with record_function(self.with_fqn("FSDP::all_gather")):
            # TODO(task1): gather the parameters shards (cast to `param_dtype`) for each parameter
            param_all_gather_outputs = []
            with torch.cuda.stream(self.comm_ctx.all_gather_stream):
                with torch.no_grad():
                    for fsdp_param in self.fsdp_params:
                        sharded_tensor = fsdp_param.sharded_param
                        if fsdp_param.param_dtype is not None:
                            sharded_tensor = sharded_tensor.to(fsdp_param.param_dtype)
                            
                        unsharded_dtensor = DTensor.redistribute(sharded_tensor, placements=[Replicate()])
                        local = unsharded_dtensor.to_local()
                        param_all_gather_outputs.append(local)
                        
                        # Clean up intermediate DTensor references
                        del sharded_tensor
                        del unsharded_dtensor
                    
            
            all_gather_event = torch.cuda.Event()
            all_gather_event.record(self.comm_ctx.all_gather_stream)
            # TODO(task2): create an event which marks the end of all-gather
            # and save it in `AllGatherResult`
            # self._all_gather_result.all_gather_event = torch.cuda.Event()
            self._all_gather_result = AllGatherResult(
                param_all_gather_outputs=param_all_gather_outputs,
                all_gather_event=all_gather_event
            )


    def wait_for_unshard(self):
        # TODO(task2): wait for the end of the all-gather launched by `unshard`
        # self._all_gather_result.all_gather_event.synchronize()
        # TODO(task1): for each parameter:
        #   - allocate its unsharded paramter
        #   - copy the all-gather output into it
        #   - assign the unsharded parameter into the module (call `.to_unsharded()`)
        # then free the `all_gather_result`
        # NOTE: copy to the `.data` attribute
        outputs = self._all_gather_result.param_all_gather_outputs

        if self._all_gather_result.all_gather_event is not None:
            self._all_gather_result.all_gather_event.wait()
        
        for fsdp_param in self.fsdp_params:
            all_gather_output = outputs.pop(0)

            fsdp_param.alloc_unsharded_param()
            
            with torch.no_grad():
                fsdp_param.unsharded_param.data.copy_(all_gather_output)
                
            fsdp_param.to_unsharded()
            
            del all_gather_output

        self._sharded_state = ShardedState.UNSHARDED
        self._all_gather_result = None

        copy_event = torch.cuda.Event()
        copy_event.record()
        self.comm_ctx.all_gather_stream.wait_event(copy_event)
        # TODO(task2): block all-gather stream until copy is complete,
        copy_event = torch.cuda.Event()
        copy_event.record()
        self.comm_ctx.all_gather_stream.wait_event(copy_event)
        # so it doesn't interfere with the next unshard

    def reshard(self):
        if self._training_state == TrainingState.FORWARD and not self._reshard_after_forward:
            return
            
        for fsdp_param in self.fsdp_params:
            fsdp_param.free_unsharded_param()
            fsdp_param.to_sharded()
            
        self._sharded_state = ShardedState.SHARDED


    def record_post_forward(self) -> None:
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)

    def register_post_backward_final_callback(self):
        Variable._execution_engine.queue_callback(self._post_backward_final_callback)

    def _post_backward_final_callback(self) -> None:
        with torch.profiler.record_function("FSDP::root_post_backward_callback"):
            if self._training_state != TrainingState.POST_BACKWARD:
                # Run post-backward in case forward inputs did not require
                # gradient so the autograd backward did not run
                post_backward(self)
            self._training_state = TrainingState.IDLE
            if self._post_reduce_event is not None:
                self._post_reduce_event.wait()
            self._post_forward_indices.clear()
            self.comm_ctx.post_forward_order.clear()

    def _backward_prefetch(self) -> None:
        # TODO(task3): using `self._post_forward_indices` and `self.comm_ctx.post_forward_order`
        # find the right FSDPModule to prefetch
        if not self._post_forward_indices:
            return
            
        current_idx = self._post_forward_indices.pop()

        if current_idx > 0:
            target_fsdp_module = self.comm_ctx.post_forward_order[current_idx - 1]
            self._prefetch_unshard(target_fsdp_module)

    @staticmethod
    def _prefetch_unshard(target_fsdp_module: "FSDPModule") -> None:
        with (
            record_function(
                f"FSDP::backward_prefetch for {target_fsdp_module._module_fqn}"
            ),
            target_fsdp_module.use_training_state(TrainingState.PRE_BACKWARD),
        ):
            target_fsdp_module.unshard()

    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState):
        old_training_state = self._training_state
        self._training_state = training_state
        try:
            yield
        finally:
            self._training_state = old_training_state

    def with_fqn(self, label: str) -> str:
        if self._module_fqn:
            return f"{label} ({self._module_fqn})"
        return label


def pre_forward(
    module: FSDPModule, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    logger.debug("%s", module.with_fqn("FSDP::pre_forward"))
    with record_function(module.with_fqn("FSDP::pre_forward")):
        module._training_state = TrainingState.FORWARD
        torch.cuda.synchronize()
        module.unshard()
        module.wait_for_unshard()
        args, kwargs = register_post_backward_hook(module, args, kwargs)
        return args, kwargs


def post_forward(module: FSDPModule, input: Any, output: Any):
    logger.debug("%s", module.with_fqn("FSDP::post_forward"))
    with record_function(module.with_fqn("FSDP::post_forward")):
        module.reshard()
        module.record_post_forward()
        module._training_state = TrainingState.IDLE
        output = register_pre_backward_hook(partial(pre_backward, module), output)
        return output


def pre_backward(module: FSDPModule, grad: torch.Tensor):
    module.register_post_backward_final_callback()
    logger.debug("%s", module.with_fqn("FSDP::pre_backward"))
    with record_function(module.with_fqn("FSDP::pre_backward")):
        module._training_state = TrainingState.PRE_BACKWARD
        module.unshard()  # no-op if prefetched
        module.wait_for_unshard()
        module._backward_prefetch()
        # TODO(task3): uncomment the next line
    return grad


def post_backward(module: FSDPModule):
    logger.debug("%s", module.with_fqn("FSDP::post_backward"))
    module._training_state = TrainingState.POST_BACKWARD
    
    with record_function(module.with_fqn("FSDP::post_backward_reshard")):
        # TODO(task1): reshard the module
        module.reshard()
        
    with record_function(module.with_fqn("FSDP::post_backward_reduce")):
        current_stream = torch.cuda.current_stream()
        
        # NOTE: wait for the current stream to finish its backward pass
        module.comm_ctx.reduce_scatter_stream.wait_stream(current_stream)
        
        with torch.no_grad():
            grad_copies = []
            
            # TODO(task3): allocate the inputs for the reduce-scatter in the reduce-scatter stream
            # copy the grads into some memory allocated in the reduce-scatter stream
            with torch.cuda.stream(module.comm_ctx.reduce_scatter_stream):
                for fsdp_param in module.fsdp_params:
                    if fsdp_param.unsharded_param.grad is not None:
                        grad_copy = fsdp_param.unsharded_param.grad.clone()
                        grad_copies.append((fsdp_param, grad_copy))
                    else:
                        grad_copies.append((fsdp_param, None))

            # TODO(task3): now block current stream until reduce-scatter stream finishes the copy

            copy_event = torch.cuda.Event()
            copy_event.record(module.comm_ctx.reduce_scatter_stream)
            current_stream.wait_event(copy_event)
            
            for fsdp_param in module.fsdp_params:
                fsdp_param.unsharded_param.grad = None

            with torch.cuda.stream(module.comm_ctx.reduce_scatter_stream):
                for fsdp_param, grad_copy in grad_copies:
                    if grad_copy is None:
                        continue
                        
                    if fsdp_param.reduce_dtype is not None:
                        grad_copy = grad_copy.to(fsdp_param.reduce_dtype)
                        
                    # TODO(task1): reduce-scatter the gradients
                    partial_grad = DTensor.from_local(
                        grad_copy,
                        device_mesh=fsdp_param.mesh,
                        placements=[Partial('avg')],
                        shape=fsdp_param.orig_size,
                        stride=grad_copy.stride(),
                    )
                    
                    sharded_grad_dt = DTensor.redistribute(partial_grad, placements=[Shard(0)])
                    
                    del partial_grad
                    del grad_copy
                    
                    if sharded_grad_dt.dtype != fsdp_param.orig_dtype:
                        sharded_grad_dt = sharded_grad_dt.to(fsdp_param.orig_dtype)

                    if fsdp_param.sharded_param.grad is not None:
                        fsdp_param.sharded_param.grad.add_(sharded_grad_dt)
                    else:
                        fsdp_param.sharded_param.grad = sharded_grad_dt

                # TODO(task3): create an event which marks the end of the reduce-scatter
                # and save it to `_post_reduce_event` to wait for it when the whole backward finishes
                reduce_event = torch.cuda.Event()
                reduce_event.record(module.comm_ctx.reduce_scatter_stream)
                module._post_reduce_event = reduce_event



def register_pre_backward_hook(hook: Callable, output: Any) -> Any:
    if not torch.is_grad_enabled():
        return output
    flat_outputs, _ = tree_flatten(output)
    for t in flat_outputs:
        if torch.is_tensor(t) and t.requires_grad:
            t.register_hook(hook)
    return output


def register_post_backward_hook(
    module: FSDPModule, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if not torch.is_grad_enabled():
        return args, kwargs
    args_list, args_spec = tree_flatten(args)
    kwargs_list, kwargs_spec = tree_flatten(kwargs)
    args_kwargs_list = list(args_list) + list(kwargs_list)
    inp_tensor_indices: list[int] = []
    inp_tensors: list[torch.Tensor] = []
    for i, obj in enumerate(args_kwargs_list):
        if torch.is_tensor(obj) and obj.requires_grad:
            inp_tensor_indices.append(i)
            inp_tensors.append(obj)
    if len(inp_tensors) == 0:
        return args, kwargs  # no tensors that require gradients
    inp_tensors = RegisterPostBackwardFunction.apply(module, *inp_tensors)
    for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
        args_kwargs_list[inp_tensor_idx] = inp_tensor
    args_list = args_kwargs_list[: len(args_list)]
    kwargs_list = args_kwargs_list[len(args_list) :]
    args = tree_unflatten(args_list, args_spec)
    kwargs = tree_unflatten(kwargs_list, kwargs_spec)
    return args, kwargs


class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, module: FSDPModule, *inputs: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        # All tensors in `inputs` should require gradient
        ctx.module = module
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> tuple[None | torch.Tensor, ...]:
        post_backward(ctx.module)
        return (None, *grads)
