"""Mapping from UniversalFeatureExtractor model names to DeepNSD model names.

This module provides a mapping dictionary that translates model names used in
UniversalFeatureExtractor to the corresponding model names used in DeepNSD.
"""

# Mapping from UniversalFeatureExtractor names to DeepNSD names
UNIVERSAL_TO_DEEPNSD_MAPPING = {
    # Torchvision models
    'torchvision_vgg16_imagenet1k_v1': 'vgg16_classification',
    'torchvision_resnet50_imagenet1k_v1': 'resnet50_classification',
    'torchvision_alexnet_imagenet1k_v1': 'alexnet_classification',
    'torchvision_convnext_base_imagenet1k_v1': 'convnext_base_classification',
    'torchvision_vit_l_16_imagenet1k_v1': 'vit_large_patch16_224_classification',
    
    # VISSL models
    'vissl_resnet50_supervised': 'resnet50_classification',  # VISSL doesn't have supervised, use standard
    'vissl_resnet50_barlowtwins': 'ResNet50-BarlowTwins-BS2048_selfsupervised',
    'vissl_resnet50_mocov2': 'ResNet50-MoCoV2-BS256_selfsupervised',
    
    # SLIP models
    'slip_vit_l_slip': 'ViT-L-SLIP_slip',
    'slip_vit_l_simclr': 'ViT-L-SimCLR_slip',
    
    # VICREG models
    'vicreg_resnet50': 'resnet50_vicreg_selfsupervised',
    
    # OpenCLIP models (now in custom_models)
    'openclip_vit_so400m_14_siglip_webli': 'openclip_vit_so400m_14_siglip_webli',
    'openclip_vit_l_14_quickgelu_metaclip_400m': 'openclip_vit_l_14_quickgelu_metaclip_400m',
    'openclip_vit_l_14_quickgelu_metaclip_fullcc': 'openclip_vit_l_14_quickgelu_metaclip_fullcc',
    'openclip_vit_l_14_laion400m_e31': 'openclip_vit_l_14_laion400m_e31',
    
    # CORnet models (now in custom_models)
    'cornet_s': 'cornet_s',
    
    # Robustness models (now in custom_models)
    'robustness_imagenet_l2_eps3': 'robustness_imagenet_l2_eps3',
    
    # DINOv2 models (now in custom_models)
    'dinov2_vitl14': 'dinov2_vitl14',
    
    # TIMM models - these may need exact name verification
    'timm_vit_large_patch14_clip_224_laion2b': 'timm_vit_large_patch14_clip_224_laion2b',  # Check exact name
    'timm_vit_large_patch14_clip_224_dfn2b': 'timm_vit_large_patch14_clip_224_dfn2b',  # Check exact name
    'timm_vit_large_patch14_clip_quickgelu_224_openai': 'timm_vit_large_patch14_clip_quickgelu_224_openai',  # Check exact name
}

def get_deepnsd_model_name(universal_name):
    """Get DeepNSD model name from UniversalFeatureExtractor name.
    
    Parameters
    ----------
    universal_name : str
        Model name as used in UniversalFeatureExtractor
        
    Returns
    -------
    str
        Corresponding DeepNSD model name, or the original name if no mapping exists
    """
    return UNIVERSAL_TO_DEEPNSD_MAPPING.get(universal_name, universal_name)

def get_universal_model_name(deepnsd_name):
    """Get UniversalFeatureExtractor name from DeepNSD model name.
    
    Parameters
    ----------
    deepnsd_name : str
        Model name as used in DeepNSD
        
    Returns
    -------
    str or None
        Corresponding UniversalFeatureExtractor model name, or None if no reverse mapping exists
    """
    reverse_mapping = {v: k for k, v in UNIVERSAL_TO_DEEPNSD_MAPPING.items()}
    return reverse_mapping.get(deepnsd_name)

def list_mapped_models():
    """List all models that have mappings.
    
    Returns
    -------
    dict
        Dictionary with 'universal' and 'deepnsd' keys containing lists of model names
    """
    return {
        'universal': list(UNIVERSAL_TO_DEEPNSD_MAPPING.keys()),
        'deepnsd': list(UNIVERSAL_TO_DEEPNSD_MAPPING.values())
    }

