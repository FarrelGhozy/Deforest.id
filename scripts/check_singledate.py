import torch
state = torch.load('models/unet_deforest_v2/best.pth', map_location='cpu', weights_only=True)
print('Single-date model checkpoint:')
for k in list(state.keys())[:5]:
    print(f'  {k}: {state[k].shape}')
print(f'Total param count: {sum(p.numel() for p in state.values()):,}')
