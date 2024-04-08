"""
TK-PLANES dataparsers configuration file.
"""
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

# from pynerf.data.dataparsers.mipnerf_dataparser import MipNerf360DataParserConfig
from tkplanes.data.dataparsers.okutama_dataparser import OkutamaDataParserConfig

# mipnerf360_dataparser = DataParserSpecification(config=MipNerf360DataParserConfig())
okutama_dataparser = DataParserSpecification(config=OkutamaDataParserConfig())