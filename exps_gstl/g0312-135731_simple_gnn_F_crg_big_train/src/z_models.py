import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv
from torch_geometric.utils import scatter
import utils
# Define the MLP for GINConv
class GINMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class GCN(torch.nn.Module):
    def __init__(self, input_dim, ego_state_dim, args):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.ego_state_dim = ego_state_dim
        self.output_dim = args.condition_dim  # gnn_feat_dim + ego_state_dim
        _conv_list = []
        all_dims = [self.input_dim] + args.hiddens + [self.output_dim]
        self.lin = []
        for layer_in_dim, layer_out_dim in zip(all_dims[:-1], all_dims[1:]):
            if args.gat:
                _conv_list.append(GATv2Conv(layer_in_dim, layer_out_dim))
            elif args.gin_conv:
                _conv_list.append(GINConv(GINMLP(layer_in_dim, layer_out_dim, layer_out_dim)))
            else:
                if args.gcn_no_self_loops:
                    _conv_list.append(GCNConv(layer_in_dim, layer_out_dim, add_self_loops=False))
                else:
                    _conv_list.append(GCNConv(layer_in_dim, layer_out_dim))
            if args.residual:
                if layer_in_dim==layer_out_dim:
                    self.lin.append(nn.Identity())
                else:
                    self.lin.append(nn.Linear(layer_in_dim, layer_out_dim))
        if args.residual:
            self.lin = torch.nn.ModuleList(self.lin)
        self.conv_list = torch.nn.ModuleList(_conv_list)
        self.mlp = utils.build_relu_nn(input_dim=self.ego_state_dim, output_dim=self.ego_state_dim, hiddens=args.mlp_hiddens)

        if args.with_predict_head:
            self.predict_head = nn.Linear(args.condition_dim, 4)
        
        if args.predict_score:
            n_dim = 64
            self.predict_score_head = utils.build_relu_nn(input_dim=args.horizon * 2, output_dim=args.condition_dim, hiddens=[n_dim])
        
    def predict(self, embedding):
        logits = self.predict_head(embedding)
        return logits
        
    def forward(self, ego_states, data):
        x, edge_index = data.x, data.edge_index
        for layer_i in range(len(self.conv_list)):
            if self.args.residual:
                identity = self.lin[layer_i](x)
            x = self.conv_list[layer_i](x, edge_index)
            if self.args.residual and self.args.post_residual==False:
                x = x + identity
            if layer_i != len(self.conv_list)-1:
                x = F.relu(x)
            if self.args.residual and self.args.post_residual==True:
                x = x + identity

        if self.args.aggr_type==0:
            x = scatter(x, data.batch, dim=0 ,reduce="mean")
        elif self.args.aggr_type==1:
            x = scatter(x, data.batch, dim=0 ,reduce="max")
        elif self.args.aggr_type==2:
            x0 = scatter(x, data.batch, dim=0 ,reduce="mean")
            x2 = scatter(x, data.batch, dim=0 ,reduce="max")
            x = (x0+x2)/2
        elif self.args.aggr_type==3:
            x0 = scatter(x, data.batch, dim=0 ,reduce="mean")
            x1 = scatter(x, data.batch, dim=0 ,reduce="min")
            x2 = scatter(x, data.batch, dim=0 ,reduce="max")
            x = (x0+x1+x2)/3
        elif self.args.aggr_type==4:
            root_indices = torch.where(data.batch[:-1]!=data.batch[1:])[0]+1
            root_indices = F.pad(root_indices, [1,0], mode='constant', value=0)
            x = x[root_indices]

        if ego_states is None:
            return x
        else:
            ego_feat = self.mlp(ego_states)
            x = torch.cat([x, ego_feat], dim=-1)
            return x

class ScorePredictor(torch.nn.Module):
    def __init__(self, encoder, feat_dim, nt, state_dim, args):
        super(ScorePredictor, self).__init__()
        self.encoder = encoder
        self.args = args
        self.state_dim = state_dim
        self.nt = nt
        self.traj_mlp = utils.build_relu_nn(input_dim=state_dim*nt, output_dim=feat_dim, hiddens=args.traj_hiddens)
        self.score_mlp = utils.build_relu_nn(input_dim=feat_dim, output_dim=1, hiddens=args.score_hiddens)
    
    def get_stl_embedding(self, stl_data):
        return self.encoder(None, stl_data)
    
    def get_traj_embedding(self, trajs):
        return self.traj_mlp(trajs)
    
    def pred_score(self, stl_feat, traj_feat):
        fuse_feat = stl_feat + traj_feat
        score = self.score_mlp(fuse_feat).squeeze(-1)
        return score
    
    def forward(self, ego_states, stl_data, trajs):
        stl_feat = self.encoder(None, stl_data)  # (BS, *)
        
        traj_feat = self.traj_mlp(trajs)  # (BS, *)
        fuse_feat = stl_feat + traj_feat
        score = self.score_mlp(fuse_feat).squeeze(-1)
        return score # (BS, *)
    
    def dual_forward(self, ego_states, stl_data, trajs, mini_batch, stl_feat=None):
        if stl_feat is None:
            stl_feat = self.encoder(None, stl_data)  # (BS, *)
        
        traj_feat = self.traj_mlp(trajs)  # (2*BS, *)
        BS = stl_feat.shape[0]
        doub_traj_feat = traj_feat.reshape(2,BS,traj_feat.shape[-1])[:,:mini_batch]
        doub_stl_feat = torch.stack([stl_feat[:mini_batch], stl_feat[:mini_batch]], dim=0)
        fuse_feat = doub_stl_feat + doub_traj_feat
        score = self.score_mlp(fuse_feat).squeeze(-1)
        return score