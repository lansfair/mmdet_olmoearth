waves_list= {
    "COASTAL_AEROSOL":      0.44,
    "BLUE":                 0.49,
    "GREEN":                0.56,
    "RED":                 0.665,
    "RED_EDGE_1":          0.705,
    "RED_EDGE_2":           0.74, 
    "RED_EDGE_3":          0.783,
    "NIR_BROAD":           0.832,
    "NIR_NARROW":          0.864,
    "WATER_VAPOR":         0.945,
    "CIRRUS":              1.373,
    "SWIR_1":               1.61,
    "SWIR_2":               2.20,
    "THEMRAL_INFRARED_1":  10.90,
    "THEMRAL_INFRARED_12": 12.00, 
    "VV":                  5.405,
    "VH":                  5.405,
    "ASC_VV":              5.405,
    "ASC_VH":              5.405,
    "DSC_VV":              5.405,
    "DSC_VH":              5.405,
    "VV-VH":               5.405
}


arch_settings = {
    "base": dict(embed_dim=768, depth=12, num_heads=12, default_out_indices=(4, 6, 10, 11)),
    "large": dict(embed_dim=1024, depth=24, num_heads=16, default_out_indices=(5, 9, 15, 21)),
}


def get_wavelenghts(model_bands: list[str]) -> list[float]:
    """Extract wavelength values for given spectral bands.
    
    Args:
        model_bands: List of band names (e.g., ['RED', 'NIR', 'SWIR_1'])
    
    Returns:
        List of corresponding wavelength values in micrometers
    """
    wavelengths = [waves_list[x.split('.')[-1]] for x in model_bands]
    return wavelengths


def get_arch_setting(arch: str = 'large'):
    if arch not in arch_settings:
        raise KeyError(f"Unsupported DOFA v2 arch: {arch}")
    return arch_settings.get(arch)
