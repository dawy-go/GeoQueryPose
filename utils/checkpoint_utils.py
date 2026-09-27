import gorilla
import os
import torch
from omegaconf import OmegaConf


def _checkpoint_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _checkpoint_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_checkpoint_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_checkpoint_to_cpu(value) for value in obj)
    return obj


def _atomic_torch_save(obj, path):
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    basename = os.path.basename(path)
    tmp_path = os.path.join(directory, f".{basename}.tmp.{os.getpid()}")
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def save_checkpoint(model, path, epoch, iter, optimizer=None, scheduler=None, config=None):
    meta = {'iter': iter, "epoch": epoch}
    checkpoint = {
        'meta'     : meta,
        'model'    : _checkpoint_to_cpu(model.state_dict())
    }

    if scheduler is not None:
        if hasattr(scheduler, "state_dict"):
            scheduler = scheduler.state_dict()
        checkpoint['scheduler'] = _checkpoint_to_cpu(scheduler)
    if optimizer is not None:
        if hasattr(optimizer, "state_dict"):
            optimizer = optimizer.state_dict()
        checkpoint['optimizer'] = _checkpoint_to_cpu(optimizer)
    if config is not None:
        checkpoint['config'] = dict(config)
    _atomic_torch_save(checkpoint, path)


def load_checkpoint_old(model, path, optimizer=None, scheduler=None, device='cuda'):
    """
    Simple checkpoint loading function

    Args:
        model: model instance
        path: checkpoint file path
        optimizer: optimizer instance (optional)
        scheduler: scheduler instance (optional)
        device: target device ('cuda' or 'cpu')
    """
    # Load checkpoint to target device
    checkpoint = torch.load(path, map_location=device)

    # Return metadata
    meta = checkpoint.get('meta', {})

    if meta.get('epoch', 0)<=30:
        model.octree_model.load_state_dict(checkpoint['model'])
    else:
        # Load model
        model.load_state_dict(checkpoint['model'],strict=False)

    # Load optimizer (if provided)
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if device == 'cuda' and torch.cuda.is_available():
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.cuda()

    # Load scheduler (if provided)
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return meta.get('epoch', 0), meta.get('iter', 0)


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, config=None, device='cuda'):
    """
    Load complete checkpoint (including optimizer, scheduler, etc.),
    automatically free memory and transfer to GPU

    Args:
        model: PyTorch model
        checkpoint_path: path to checkpoint file
        optimizer: optimizer instance (optional)
        scheduler: learning rate scheduler (optional)
        device: target device
        config: configuration (optional)

    Returns:
        tuple: (epoch, iteration)
    """
    import gc

    print(f">>>>> Loading checkpoint: {checkpoint_path}")

    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Record initial GPU memory
    if device == 'cuda' and torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024 ** 3
        print(f">>>>> Initial GPU memory: {initial_memory:.2f}GB")

    # Load checkpoint to CPU
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract metadata
    meta = checkpoint.get('meta', {})
    epoch = meta.get('epoch', 0)
    iteration = meta.get('iter', 0)

    # Temporarily move model to CPU for loading
    model = model.cpu()

    # Load model weights
    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
        print(">>>>> Model weights loaded")
    elif 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
        print(">>>>> Model weights loaded (using state_dict)")
    else:
        raise RuntimeError(">>>>> Cannot find model weights in checkpoint")

    exclude_prefixes = []
    if config is not None:
        configured = OmegaConf.select(
            config,
            "checkpoint_loading.exclude_prefixes",
            default=[],
        )
        if configured:
            exclude_prefixes = [str(prefix).rstrip(".") for prefix in configured]

    def normalized_key(key):
        return key[7:] if key.startswith("module.") else key

    def excluded(key):
        key = normalized_key(key)
        return any(
            key == prefix or key.startswith(prefix + ".")
            for prefix in exclude_prefixes
        )

    model_dict = model.state_dict()
    loaded_dict = {}
    excluded_keys = []
    mismatched_keys = []
    for checkpoint_key, value in pretrained_dict.items():
        model_key = checkpoint_key
        if model_key not in model_dict:
            candidate = normalized_key(model_key)
            if candidate in model_dict:
                model_key = candidate
        if excluded(model_key):
            excluded_keys.append(checkpoint_key)
            continue
        if model_key not in model_dict:
            mismatched_keys.append((checkpoint_key, "key not found"))
            continue
        if value.shape != model_dict[model_key].shape:
            mismatched_keys.append((checkpoint_key, "unmatched shape"))
            continue
        loaded_dict[model_key] = value

    for key, reason in mismatched_keys:
        print(f">>> Mis-matched key: {key}, (reason: {reason})")
    if exclude_prefixes:
        print(
            ">>>>> Selective checkpoint load excluded {} tensors under: {}".format(
                len(excluded_keys),
                ", ".join(exclude_prefixes),
            )
        )
    print(
        ">>>>> Checkpoint tensor summary: loaded={} excluded={} mismatched={}".format(
            len(loaded_dict),
            len(excluded_keys),
            len(mismatched_keys),
        )
    )
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict, strict=False)

    # Move model to target device
    print(f"\n>>>>> Moving model to: {device}...")
    model = model.to(device)

    if optimizer is not None and 'optimizer' in checkpoint:
        print(">>>>> Loading optimizer state...")
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Move optimizer states to target device
        if device == 'cuda' and torch.cuda.is_available():
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.cuda()

        print(f">>>>> Optimizer state moved to {device}")

    # Load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

        # Move scheduler tensor states if needed
        if device == 'cuda' and torch.cuda.is_available():
            for attr_name in dir(scheduler):
                if not attr_name.startswith('_'):  # Skip private attributes
                    attr = getattr(scheduler, attr_name)
                    if torch.is_tensor(attr):
                        setattr(scheduler, attr_name, attr.cuda())

        print(">>>>> Learning rate scheduler states loaded")

    # Load config
    checkpoint_mode = (
        str(OmegaConf.select(config, "checkpoint_loading.mode", default="standard"))
        .lower()
        if config is not None
        else "standard"
    )
    if config is not None and 'config' in checkpoint and checkpoint_mode != "selective":
        config.update(gorilla.Config(checkpoint['config']))
        print(">>>>> Configs loaded")
    elif config is not None and 'config' in checkpoint and checkpoint_mode == "selective":
        print(">>>>> Embedded checkpoint config ignored for selective warm start")

    # Free checkpoint memory
    del checkpoint, model_dict

    # Final memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        # Show final GPU memory usage
        final_memory = torch.cuda.memory_allocated() / 1024 ** 3
        memory_increase = final_memory - initial_memory if 'initial_memory' in locals() else 0
        print(f">>>>> Final GPU memory: {final_memory:.2f}GB (increased: {memory_increase:.2f}GB)")

    print(f">>>>> Checkpoint loaded successfully: {checkpoint_path}")

    return epoch, iteration
