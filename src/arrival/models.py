"""Small arrival classifiers and a discrete-time hazard alternative."""
import torch
from torch import nn
from torch.nn import functional as F
from models.multi_modal_convlstm import ConvLSTMCell


class ConvGRUCell(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__();self.hidden_dim=hidden_dim
        self.gates=nn.Conv2d(input_dim+hidden_dim,2*hidden_dim,3,padding=1)
        self.candidate=nn.Conv2d(input_dim+hidden_dim,hidden_dim,3,padding=1)
    def forward(self,x,h):
        reset,update=torch.sigmoid(self.gates(torch.cat([x,h],1))).chunk(2,1)
        candidate=torch.tanh(self.candidate(torch.cat([x,reset*h],1)))
        return (1-update)*h+update*candidate


class ArrivalNet(nn.Module):
    def __init__(self,history=6,channels=7,kind='gru',output='categorical',dropout=.15,hidden_dim=64,pooling='global',recurrent_layers=1,encoder_extra_layers=0):
        super().__init__()
        if kind not in ('cnn','gru','lstm') or output not in ('categorical','hazard'):raise ValueError('Unknown model')
        if hidden_dim<1 or pooling not in ('global','spatial','spatial_change'):raise ValueError('Invalid hidden width or pooling')
        if pooling=='spatial_change' and kind=='cnn':raise ValueError('Temporal feature change requires recurrence')
        if recurrent_layers not in (1,2,3) or encoder_extra_layers not in (0,1,2):raise ValueError('Unsupported depth')
        if kind=='cnn' and recurrent_layers!=1:raise ValueError('CNN has no recurrent layers')
        self.hidden_dim=hidden_dim
        self.config=dict(history=history,channels=channels,kind=kind,output=output,dropout=dropout,hidden_dim=hidden_dim,pooling=pooling,recurrent_layers=recurrent_layers,encoder_extra_layers=encoder_extra_layers)
        self.kind,self.output,self.history=kind,output,history
        self.encoder=nn.Sequential(nn.Conv2d(channels*history if kind=='cnn' else channels,16,3,stride=2,padding=1),
            nn.ReLU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU())
        for _ in range(encoder_extra_layers):
            self.encoder.append(nn.Conv2d(32,32,3,padding=1));self.encoder.append(nn.ReLU())
        if kind=='gru':self.cell=ConvGRUCell(32,hidden_dim)
        elif kind=='lstm':self.cell=ConvLSTMCell(32,hidden_dim,(3,3),True)
        self.extra_cells=nn.ModuleList([
            ConvGRUCell(hidden_dim,hidden_dim) if kind=='gru' else ConvLSTMCell(hidden_dim,hidden_dim,(3,3),True)
            for _ in range(recurrent_layers-1)])
        features=32 if kind=='cnn' else hidden_dim
        # A 4x4 grid retains relative position around the centered target.
        grid=1 if pooling=='global' else 4
        features*=grid*grid
        if pooling=='spatial_change':features*=2
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(grid),nn.Flatten(),nn.Dropout(dropout),
                                nn.Linear(features,32),nn.ReLU(),nn.Linear(32,12 if output=='hazard' else 13))

    def forward(self,x):
        b,t,c,h,w=x.shape
        if t!=self.history:raise ValueError('Input history does not match model')
        if self.kind=='cnn':features=self.encoder(x.reshape(b,t*c,h,w))
        else:
            encoded=self.encoder(x.reshape(b*t,c,h,w));encoded=encoded.reshape(b,t,*encoded.shape[1:])
            states=[x.new_zeros(b,self.hidden_dim,*encoded.shape[-2:]) for _ in range(1+len(self.extra_cells))]
            memories=[torch.zeros_like(state) for state in states]
            for i in range(t):
                layer_input=encoded[:,i]
                for j,cell in enumerate([self.cell,*self.extra_cells]):
                    if self.kind=='gru':states[j]=cell(layer_input,states[j])
                    else:states[j],memories[j]=cell(layer_input,(states[j],memories[j]))
                    layer_input=states[j]
                if i==0:first_features=states[-1]
            features=states[-1]
            if self.config['pooling']=='spatial_change':
                features=torch.cat([features,features-first_features],dim=1)
        return self.head(features)

    def log_probs(self,x):
        logits=self(x)
        return F.log_softmax(logits,1) if self.output=='categorical' else hazard_log_probs(logits)


def hazard_log_probs(logits):
    log_survival=F.logsigmoid(-logits)
    before=torch.cat([torch.zeros_like(logits[:,:1]),log_survival.cumsum(1)[:,:-1]],1)
    return torch.cat([before+F.logsigmoid(logits),log_survival.sum(1,keepdim=True)],1)
