import torch
from torch import nn


class DARConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1):
        super(DARConv2d, self).__init__()
        self.padding = padding
        self.position = nn.Conv2d(inc, 9, kernel_size=3, padding=1, stride=1)
        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1,padding=self.padding)
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self, x):
        position_nine = torch.sigmoid(self.position(x))
        deform_attention = None
        for i in range(9):
            if i==0:
                deform_attention = torch.mul(x.squeeze(dim=0),position_nine[:,0:1,:,:])
            else:
                deform_attention =  deform_attention + torch.mul(x,position_nine[:,i:i+1,:,:])
        
        deform_attention_mask = self.softmax(torch.div(deform_attention, 9))
        x_with_mask = torch.mul(x,deform_attention_mask)
        out = self.conv(x_with_mask)
        
        return out
