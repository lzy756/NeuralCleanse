import torch

class MaskedModelWrapper:
    """神经元屏蔽包装器"""
    def __init__(self, model, layer='classifier.1'):
        self.model = model
        self.masked_neurons = set()
        
        # 注册前向传播钩子
        for name, module in self.model.named_modules():
            if name == layer:
                module.register_forward_hook(self._apply_mask_hook)
    
    def _apply_mask_hook(self, module, input_, output):
        """应用神经元屏蔽的钩子函数"""
        mask = torch.ones_like(output)
        for idx in self.masked_neurons:
            mask[:, idx] = 0  # 将屏蔽的神经元输出置零
        return output * mask
    
    def add_masked_neuron(self, neuron_idx):
        """添加要屏蔽的神经元"""
        self.masked_neurons.add(neuron_idx)
        
    def reset(self):
        """重置所有屏蔽"""
        self.masked_neurons.clear()
