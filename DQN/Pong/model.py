import torch.nn as nn

class Q_net_Pong(nn.Module):
    def __init__(self, FRAME_SIZE, n_last_frames, n_actions):
        super(Q_net_Pong, self).__init__()
        self.frame_size = FRAME_SIZE
        self.n_last_frames = n_last_frames
        filters = [32, 64, 64]
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]        
        self.convs = nn.ModuleList()        
        self.n_convs = len(filters)
        
        for i in range(self.n_convs):
            if i==0:
                self.convs.append(nn.Conv2d(self.n_last_frames, filters[i], kernel_size=kernel_sizes[i], stride=strides[i]))
            else:
                self.convs.append(nn.Conv2d(filters[i-1], filters[i], kernel_size=kernel_sizes[i], stride=strides[i])) 
                
        def convs_output_size(frame_size):
            height, width = frame_size
            for i in range(self.n_convs):
                height = (height - (kernel_sizes[i] - 1) - 1) // strides[i] + 1
                width = (width - (kernel_sizes[i] - 1) - 1) // strides[i] + 1
            return height, width
        
        h, w = convs_output_size(self.frame_size)
        self.n_inputs = filters[-1] * h * w
        n_hidden = 512
        self.linear1 = nn.Linear(self.n_inputs, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_actions)
        self.relus = nn.ModuleList()
        for i in range(self.n_convs+1):
            self.relus.append(nn.ReLU())
        
    def forward(self, x):
        x = x.view(-1, self.n_last_frames, self.frame_size[0], self.frame_size[1])
        for i in range(self.n_convs):
            x = self.relus[i](self.convs[i](x))
        h3 = x.view(-1, self.n_inputs)
        h4 = self.relus[-1](self.linear1(h3))
        o = self.linear2(h4)
        return o