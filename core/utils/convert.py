from ..models.pinn import PINN

from ..layers.monarch import MonarchLinear, MonarchTensor
from ..layers.steam import STEAMLinear, STEAMTensor

from torch import nn, Tensor
import copy

def linear_to_monarch(model: PINN):
    layers: list[nn.Module] = []
    for layer in model.layers:
        
        if isinstance(layer, nn.Linear) and layer.out_features == layer.in_features:

            layers.append(
                MonarchLinear.from_tensors(MonarchTensor.from_dense(layer.weight), layer.bias)
            )

        else:            
            layers.append(layer)
    
    model.layers = nn.ModuleList(layers)
    return model


def linear_to_steam(model: PINN, T: int=100, alpha: float=0.001):
    layers: list[nn.Module] = []
    for layer in model.layers:
        
        if isinstance(layer, nn.Linear) and layer.out_features == layer.in_features:

            layers.append(
                STEAMLinear.from_tensors(STEAMTensor.from_dense(layer.weight, T, alpha), layer.bias)
            )
        else:
            layers.append(layer)

    model.layers = nn.ModuleList(layers)
    return model


def monarch_to_linear(model: PINN):
    layers: list[nn.Module] = []
    for layer in model.layers:
        if isinstance(layer, MonarchLinear):
            linear_layer = nn.Linear(layer.in_features, layer.out_features, bias=True)

            linear_layer.weight.data = layer.tensor().dense.clone().contiguous()
            linear_layer.bias.data = layer.bias.clone().contiguous()
            layers.append(linear_layer)
        else:
            layers.append(layer)

    model.layers = nn.ModuleList(layers)
    return model


def steam_to_linear(model: PINN):
    layers: list[nn.Module] = []
    for layer in model.layers:
        if isinstance(layer, STEAMLinear):
            linear_layer = nn.Linear(layer.in_features, layer.out_features, bias=True)
            
            linear_layer.weight.data = layer.dense.clone().contiguous()
            linear_layer.bias.data = layer.bias.clone().contiguous()

            layers.append(linear_layer)
        else:
            layers.append(layer)
    
    model.layers = nn.ModuleList(layers)
    return model

def convert(pinn: PINN, input_factor: str, output_factor: str):
    
    
    model_to_convert = copy.deepcopy(pinn)

    if input_factor == output_factor:
        return model_to_convert
    
    if input_factor == "monarch" and output_factor == "linear":
        return monarch_to_linear(model_to_convert)
    
    if input_factor == "linear" and output_factor == "monarch":
        return linear_to_monarch(model_to_convert)
    
    if input_factor == "steam" and output_factor == "linear":
        return steam_to_linear(model_to_convert)
    
    if input_factor == "linear" and output_factor == "steam":
        return linear_to_steam(model_to_convert)
    
    if input_factor == "monarch" and output_factor == "steam":
        return linear_to_steam(linear_to_monarch(model_to_convert))
    
    if input_factor == "steam" and output_factor == "monarch":
        return monarch_to_linear(linear_to_steam(model_to_convert))
    
    raise ValueError(f"Invalid input factor: {input_factor} and output factor: {output_factor}")