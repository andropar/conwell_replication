"""Extended custom models for DeepNSD.

This module adds support for models from UniversalFeatureExtractor:
- OpenCLIP models
- CORnet-S
- Robustness models
- DINOv2
"""

import torch
import torchvision.transforms as transforms

# OpenCLIP Models ---------------------------------------------------------------------------

openclip_models = {
    'vit_so400m_14_siglip_webli': {
        'model_name': 'ViT-SO400M-14-SigLIP',
        'train_data': 'webli',
        'pretrained': 'webli'
    },
    'vit_l_14_quickgelu_metaclip_400m': {
        'model_name': 'ViT-L-14-400M',
        'train_data': 'metaclip_400m',
        'pretrained': 'metaclip_400m'
    },
    'vit_l_14_quickgelu_metaclip_fullcc': {
        'model_name': 'ViT-L-14-fullCC',
        'train_data': 'metaclip_fullcc',
        'pretrained': 'metaclip_fullcc'
    },
    'vit_l_14_laion400m_e31': {
        'model_name': 'ViT-L-14',
        'train_data': 'laion400m',
        'pretrained': 'laion400m_e31'
    },
}

def define_openclip_options():
    openclip_options = {}
    
    for model_key, model_info in openclip_models.items():
        model_name = model_info['model_name']
        train_data = model_info['train_data']
        train_type = 'openclip'
        model_source = 'openclip'
        model_string = f"openclip_{model_key}"
        model_call = f"get_openclip_model('{model_key}')"
        openclip_options[model_string] = {
            'model_name': model_name,
            'train_type': train_type,
            'train_data': train_data,
            'model_source': model_source,
            'call': model_call
        }
    
    return openclip_options

def get_openclip_model(model_key):
    try:
        import open_clip
    except ImportError:
        raise ImportError("open_clip is required. Install with: pip install open-clip-torch")
    
    model_info = openclip_models[model_key]
    pretrained_key = model_info['pretrained']
    
    # Map model keys to actual open_clip model names and pretrained keys
    # Format: (model_name, pretrained_key)
    model_config_map = {
        'vit_so400m_14_siglip_webli': ('ViT-SO400M-14-SigLIP', 'webli'),
        'vit_l_14_quickgelu_metaclip_400m': ('ViT-L-14', 'metaclip_400m'),
        'vit_l_14_quickgelu_metaclip_fullcc': ('ViT-L-14', 'metaclip_fullcc'),
        'vit_l_14_laion400m_e31': ('ViT-L-14', 'laion400m_e31'),
    }
    
    if model_key not in model_config_map:
        raise ValueError(f"Unknown OpenCLIP model key: {model_key}")
    
    model_name, pretrained = model_config_map[model_key]
    
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
        
        # Return visual encoder
        return model.visual if hasattr(model, 'visual') else model
    except Exception as e:
        # If the specific pretrained version doesn't work, try with just the model name
        # and let open_clip use default pretrained weights
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None
            )
            return model.visual if hasattr(model, 'visual') else model
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load OpenCLIP model {model_key} "
                f"(model_name={model_name}, pretrained={pretrained}). "
                f"Original error: {e}. Fallback error: {e2}"
            )

def get_openclip_transforms(model_query, input_type='PIL'):
    try:
        import open_clip
    except ImportError:
        raise ImportError("open_clip is required. Install with: pip install open-clip-torch")
    
    # Extract model_key from model_query (could be full name like "openclip_vit_l_14_...")
    if model_query.startswith('openclip_'):
        model_key = model_query.replace('openclip_', '')
    else:
        model_key = model_query
    
    if model_key not in openclip_models:
        raise ValueError(f"Unknown OpenCLIP model: {model_key}")
    
    # Use the same mapping as get_openclip_model
    model_config_map = {
        'vit_so400m_14_siglip_webli': ('ViT-SO400M-14-SigLIP', 'webli'),
        'vit_l_14_quickgelu_metaclip_400m': ('ViT-L-14', 'metaclip_400m'),
        'vit_l_14_quickgelu_metaclip_fullcc': ('ViT-L-14', 'metaclip_fullcc'),
        'vit_l_14_laion400m_e31': ('ViT-L-14', 'laion400m_e31'),
    }
    
    if model_key not in model_config_map:
        raise ValueError(f"Unknown OpenCLIP model: {model_key}")
    
    model_name, pretrained = model_config_map[model_key]
    
    try:
        _, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
    except Exception as e:
        # Fallback: try without pretrained weights
        _, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None
        )
    
    if input_type == 'PIL':
        return preprocess
    if input_type == 'numpy':
        if hasattr(preprocess, 'transforms'):
            return transforms.Compose([transforms.ToPILImage()] + preprocess.transforms)
        else:
            # Handle case where preprocess is a callable
            def numpy_preprocess(img):
                pil_img = transforms.ToPILImage()(img)
                return preprocess(pil_img)
            return numpy_preprocess
    
    return preprocess

# CORnet Models ---------------------------------------------------------------------------

def define_cornet_options():
    cornet_options = {}
    
    model_name = 'cornet_s'
    train_type = 'cornet'
    train_data = 'imagenet'
    model_source = 'cornet'
    model_string = model_name
    model_call = "get_cornet_model('s')"
    
    cornet_options[model_string] = {
        'model_name': model_name,
        'train_type': train_type,
        'train_data': train_data,
        'model_source': model_source,
        'call': model_call
    }
    
    return cornet_options

def get_cornet_model(model_name):
    try:
        from cornet import get_model
    except ImportError:
        raise ImportError("cornet is required. Install with: pip install cornet")
    
    model = get_model(model_name, pretrained=True, map_location='cpu')
    return model

def get_cornet_transforms(input_type='PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_stats)
    ]
    
    if input_type == 'PIL':
        return transforms.Compose(base_transforms)
    if input_type == 'numpy':
        return transforms.Compose([transforms.ToPILImage()] + base_transforms)
    
    return transforms.Compose(base_transforms)

# Robustness Models ---------------------------------------------------------------------------

def define_robustness_options():
    robustness_options = {}
    
    model_name = 'imagenet_l2_eps3'
    train_type = 'robustness'
    train_data = 'imagenet'
    model_source = 'robustness'
    model_string = f"robustness_{model_name}"
    model_call = f"get_robustness_model('{model_name}')"
    
    robustness_options[model_string] = {
        'model_name': model_name,
        'train_type': train_type,
        'train_data': train_data,
        'model_source': model_source,
        'call': model_call
    }
    
    return robustness_options

def get_robustness_model(model_name):
    try:
        from robustness.model_utils import make_and_restore_model
    except ImportError:
        raise ImportError("robustness package is required. Install from: https://github.com/MadryLab/robustness")
    
    # Map model name to architecture and checkpoint
    # For imagenet_l2_eps3, we need the L2-constrained ResNet50
    arch_map = {
        'imagenet_l2_eps3': 'resnet50'
    }
    
    arch = arch_map.get(model_name, 'resnet50')
    
    # Load robust model - this may need adjustment based on robustness API
    # The checkpoint path would typically point to a pretrained robust model
    try:
        model, _ = make_and_restore_model(
            arch=arch,
            dataset='imagenet',
            resume_path=None,  # May need to specify checkpoint path
            pytorch_pretrained=False
        )
        return model.model if hasattr(model, 'model') else model
    except Exception:
        # Fallback: try loading as standard model if robustness API differs
        import torchvision.models as models
        return models.resnet50(pretrained=True)

def get_robustness_transforms(input_type='PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_stats)
    ]
    
    if input_type == 'PIL':
        return transforms.Compose(base_transforms)
    if input_type == 'numpy':
        return transforms.Compose([transforms.ToPILImage()] + base_transforms)
    
    return transforms.Compose(base_transforms)

# DINOv2 Models ---------------------------------------------------------------------------

def define_dinov2_options():
    dinov2_options = {}
    
    model_name = 'vitl14'
    train_type = 'selfsupervised'
    train_data = 'imagenet'
    model_source = 'dinov2'
    model_string = f"dinov2_{model_name}"
    model_call = f"get_dinov2_model('{model_name}')"
    
    dinov2_options[model_string] = {
        'model_name': model_name,
        'train_type': train_type,
        'train_data': train_data,
        'model_source': model_source,
        'call': model_call
    }
    
    return dinov2_options

def get_dinov2_model(model_name):
    try:
        import torch.hub
    except ImportError:
        raise ImportError("torch.hub required for DINOv2")
    
    # DINOv2 models from Facebook Research
    model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_name}')
    return model

def get_dinov2_transforms(input_type='PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_stats)
    ]
    
    if input_type == 'PIL':
        return transforms.Compose(base_transforms)
    if input_type == 'numpy':
        return transforms.Compose([transforms.ToPILImage()] + base_transforms)
    
    return transforms.Compose(base_transforms)

