from .api import API


def get_api_class_from_name(name):
    # Lazy import to improve loading speed and reduce libary dependency.
    if name == 'HFGPT':
        from .hf_api import HFAPI
        return HFAPI
    if name == 'AzureGPT':
        from .azure_api import AzureAPI
        return AzureAPI
    else:
        raise ValueError(f'Unknown API name {name}')


__all__ = ['get_api_class_from_name', 'API','openai_completions']
