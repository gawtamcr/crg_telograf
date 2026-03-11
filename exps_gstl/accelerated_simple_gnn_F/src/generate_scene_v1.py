import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stl_d_lib import * 
# STL store format
# node_id, node_type, time_window, len(children), [children_id_list]
# node_type - semantics
# 0 - AND
# 1 - OR 
# 2 - NOT 
# 3 - Imply
# 4 - Next
# 5 - Eventually
# 6 - Always
# 7 - Until
# 8 - AP's / objects

def generate_trajectories(s, us, dt, v_max=None):
    new_ds = torch.cumsum(us, dim=-2) * dt
    trajs = s[..., None, :] + new_ds
    trajs = torch.cat([s[..., None, :], trajs], dim=-2)
    return trajs

def generate_trajectories_dubins(s, us, dt, v_max, unclip=False):
    trajs=[s]
    for ti in range(us.shape[-2]):
        prev_s = trajs[-1]
        new_x = prev_s[..., 0] + prev_s[..., 3] * torch.cos(prev_s[..., 2]) * dt
        new_y = prev_s[..., 1] + prev_s[..., 3] * torch.sin(prev_s[..., 2]) * dt
        new_th = (prev_s[..., 2] + us[..., ti, 0] * dt + np.pi) % (2 * np.pi) - np.pi
        if unclip:
            new_v = prev_s[..., 3] + us[..., ti, 1] * dt
        else:
            new_v = torch.clip(prev_s[..., 3] + us[..., ti, 1] * dt, -0.01, v_max)
        new_s = torch.stack([new_x, new_y, new_th, new_v], dim=-1)
        trajs.append(new_s)
    trajs = torch.stack(trajs, dim=-2)
    return trajs

def check_stl_type(node):
    if isinstance(node, SimpleAnd) or isinstance(node, SimpleListAnd):
        node_type=0
    elif isinstance(node, SimpleOr) or isinstance(node, SimpleListOr):
        node_type=1
    elif isinstance(node, SimpleNot):
        node_type=2
    elif isinstance(node, SimpleImply):
        node_type=3
    elif isinstance(node, SimpleNext):
        node_type=4
    elif isinstance(node, SimpleF):
        node_type=5
    elif isinstance(node, SimpleG):
        node_type=6
    elif isinstance(node, SimpleUntil):
        node_type=7
    elif isinstance(node, SimpleReach):
        node_type=8
    else:
        raise NotImplementedError
    return node_type

def convert_stl_to_string(stl, numpy=False):
    id=0
    last_id=0
    queue = [(stl, id)]
    lines=[]
    while len(queue)>0:
        node, id = queue[0]
        del queue[0]
        node_type = check_stl_type(node)
        if numpy:
            curr_s = [id, node_type, node.ts, node.te, len(node.children)]
            if node_type!=8:
                append_s = [last_id+new_i+1 for new_i in range(len(node.children))]
            else:
                if node.ap_type is not None:
                    append_s = [node.obj_id, node.obj_x, node.obj_y, node.obj_z, node.obj_r, node.ap_type]
                else:
                    append_s = [node.obj_id, node.obj_x, node.obj_y, node.obj_z, node.obj_r]
            newline = curr_s + append_s
        else:
            curr_s = "%d %d %d %d %d" % (id, node_type, node.ts, node.te, len(node.children))
            if node_type!=8:
                append_s = " ".join(["%d"%(last_id+new_i+1) for new_i in range(len(node.children))])
            else:
                if node.ap_type is not None:
                    append_s = "%d %.4f %.4f %.4f %.4f %d"%(node.obj_id, node.obj_x, node.obj_y, node.obj_z, node.obj_r, node.ap_type)
                else:
                    append_s = "%d %.4f %.4f %.4f %.4f"%(node.obj_id, node.obj_x, node.obj_y, node.obj_z, node.obj_r)
            newline = curr_s + " " + append_s
        lines.append(newline)
        if node_type!=8:
            for new_i, child in enumerate(node.children):
                queue.append((child, last_id+new_i+1))
            last_id = last_id + len(node.children)
    return lines

def find_ap_in_lines(id, stl_dict, objects_d, lines, numpy=False, real_stl=False, ap_mode="l2", until1=False):
    for line in lines:
        if numpy:
            res = line
        else:
            res = line.strip().split()
        node_id = int(res[0])
        if node_id==id:
            node_type = int(res[1])
            
            # non-object formulas
            if node_type!=8:
                node_ts = int(res[2])
                node_te = int(res[3])
                len_children = int(res[4])
                children_ids = [int(res_id_s) for res_id_s in res[5:]]
                children = []
                for child_id in children_ids:
                    if child_id not in stl_dict:
                        child = find_ap_in_lines(child_id, stl_dict, objects_d, lines, numpy=numpy, real_stl=real_stl, ap_mode=ap_mode, until1=until1)
                        if isinstance(child, SimpleReach): # TODO use node_type as NOT to detect this is an obstacle
                            objects_d[child.obj_id] = {"x":child.obj_x, "y":child.obj_y, "z":child.obj_z, "r":child.obj_r, "is_obstacle":int(node_type)==2}
                        stl_dict[child_id] = child
                    children.append(stl_dict[child_id])
            if real_stl:
                if node_type==0:
                    if len_children==2:
                        stl = And(lhs=children[0], rhs=children[1])
                    else:
                        stl = ListAnd(lists=children)
                elif node_type==1:
                    if len_children==2:
                        stl = Or(lhs=children[0], rhs=children[1])
                    else:
                        stl = ListOr(lists=children)
                elif node_type==2:
                    stl = Not(node=children[0])
                elif node_type==3:
                    stl = Imply(lhs=children[0], rhs=children[1])
                elif node_type==4:
                    stl = Eventually(1, 2, node=children[0])
                elif node_type==5:
                    stl = Eventually(ts=node_ts, te=node_te, node=children[0])
                elif node_type==6:
                    stl = Always(ts=node_ts, te=node_te, node=children[0])
                elif node_type==7:
                    if until1:
                        stl = Until1(ts=node_ts, te=node_te, lhs=children[0], rhs=children[1])
                    else:
                        stl = Until(ts=node_ts, te=node_te, lhs=children[0], rhs=children[1])
                elif node_type==8:
                    obj_id = int(res[5])
                    obj_x = float(res[6])
                    obj_y = float(res[7])
                    obj_z = float(res[8])
                    obj_r = float(res[9])
                    if len(res)==11:
                        ap_type = int(res[10])
                    if ap_mode=="l2":
                        stl = AP(lambda x:obj_r**2 - (x[...,0]-obj_x)**2 - (x[..., 1]-obj_y)**2, comment="R%d"%(obj_id))
                    elif ap_mode=="grid":
                        stl = AP(lambda x:obj_r * 0.5 - torch.maximum(torch.abs(x[...,0]-obj_x), torch.abs(x[..., 1]-obj_y)), comment="R'%d"%(obj_id))
                    elif ap_mode=="panda":
                        if ap_type==0:  # reach an object
                            stl = And(
                                AP(reach_obj_from_panda_decorator(res), comment="R%d"%(obj_id)),
                                AP(reach_obj_from_panda_vert_decorator(res), comment="V%d"%(obj_id))
                            )
                        elif ap_type==1:  # avoid an obstacle's reach
                            stl = AP(reach_obj_from_panda_big_decorator(res), comment="R%d"%(obj_id))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                if node_type==0:
                    if len_children==2:
                        stl = SimpleAnd(left=children[0], right=children[1])
                    else:
                        stl = SimpleListAnd(list_aps=children)
                elif node_type==1:
                    if len_children==2:
                        stl = SimpleOr(left=children[0], right=children[1])
                    else:
                        stl = SimpleListOr(list_aps=children)
                elif node_type==2:
                    stl = SimpleNot(ap=children[0])
                elif node_type==3:
                    stl = SimpleImply(left=children[0], right=children[1])
                elif node_type==4:
                    stl = SimpleNext(ap=children[0])
                elif node_type==5:
                    stl = SimpleF(ts=node_ts, te=node_te, ap=children[0])
                elif node_type==6:
                    stl = SimpleG(ts=node_ts, te=node_te, ap=children[0])
                elif node_type==7:
                    stl = SimpleUntil(ts=node_ts, te=node_te, left=children[0], right=children[1])
                elif node_type==8:
                    obj_id = int(res[5])
                    obj_x = float(res[6])
                    obj_y = float(res[7])
                    obj_z = float(res[8])
                    obj_r = float(res[9])
                    stl = SimpleReach(obj_id=obj_id, obj_x=obj_x, obj_y=obj_y, obj_z=obj_z, obj_r=obj_r)
                else:
                    raise NotImplementedError
            break
    stl_dict[id] = stl
    return stl

def reach_obj_from_panda_decorator(res):
    obj_id = int(res[5])
    obj_x = float(res[6])
    obj_y = float(res[7])
    obj_z = float(res[8])
    obj_r = float(res[9])
    
    def reach_obj_from_panda(x):
        # x should be a dict
        # x["ee"] end effector (B, T+1, 3)
        # x["points"] joint points (M, B, T+1, 3)
        # x["joints"] joint angles (B, T+1, dof)
        # lambda x:obj_r**2 - (x[...,0]-obj_x)**2 - (x[..., 1]-obj_y)**2
        z_offset = 0.1
        return ((obj_r**2) - (x["ee"][...,0]-obj_x)**2 - (x["ee"][..., 1]-obj_y)**2 - (x["ee"][..., 2]-obj_z-z_offset)**2)/((obj_r))
    return reach_obj_from_panda

def reach_obj_from_panda_big_decorator(res):
    obj_id = int(res[5])
    obj_x = float(res[6])
    obj_y = float(res[7])
    obj_z = float(res[8])
    obj_r = float(res[9])
    
    def reach_obj_from_panda_big(x):
        # x should be a dict
        # x["ee"] end effector (B, T+1, 3)
        # x["points"] joint points (M, B, T+1, 3)
        # x["joints"] joint angles (B, T+1, dof)
        # lambda x:obj_r**2 - (x[...,0]-obj_x)**2 - (x[..., 1]-obj_y)**2
        return (3*(obj_r**2) - (x["ee"][...,0]-obj_x)**2 - (x["ee"][..., 1]-obj_y)**2 - (x["ee"][..., 2]-obj_z)**2)/(np.sqrt(3)*(obj_r))
    return reach_obj_from_panda_big

def reach_obj_from_panda_vert_decorator(res):
    obj_id = int(res[5])
    obj_x = float(res[6])
    obj_y = float(res[7])
    obj_z = float(res[8])
    obj_r = float(res[9])
    
    def reach_obj_from_panda_vert(x):
        # x should be a dict
        # x["ee"] end effector (B, T+1, 3)
        # x["points"] joint points (M, B, T+1, 3)
        # x["joints"] joint angles (B, T+1, dof)
        # lambda x:obj_r**2 - (x[...,0]-obj_x)**2 - (x[..., 1]-obj_y)**2
        vector = x["points"][7, :] - x["points"][8, :]
        vector = vector / torch.clip(torch.norm(vector, dim=-1, keepdim=True), min=1e-4, max=1e4)
        # unit_vector = torch.tensor([[[0, 0, 1.]]]).to(vector.device)
        vertical_val = vector[..., 2] - 0.9
        return vertical_val
    return reach_obj_from_panda_vert

# only takes notes for storage
class SimpleSTL:
    def __init__(self):
        self.children = []
        self.ts = -1
        self.te = -1
        
    def print_out(self, mode="simple"):
        s = self.__class__.__name__
        s_else = ",".join([child.__class__.__name__ for child in self.children])
        print(s + " | " + s_else)
        for child in self.children:
            child.print_out(mode)
        
class SimpleAnd(SimpleSTL):
    def __init__(self, left, right):
        super().__init__()
        self.children.append(left)
        self.children.append(right)

class SimpleListAnd(SimpleSTL):
    def __init__(self, list_aps):
        super().__init__()
        for ap in list_aps:        
            self.children.append(ap)

class SimpleOr(SimpleSTL):
    def __init__(self, left, right):
        super().__init__()
        self.children.append(left)
        self.children.append(right)

class SimpleListOr(SimpleSTL):
    def __init__(self, list_aps):
        super().__init__()
        for ap in list_aps:        
            self.children.append(ap)

class SimpleNot(SimpleSTL):
    def __init__(self, ap):
        super().__init__()
        self.children.append(ap)

class SimpleImply(SimpleSTL):
    def __init__(self, left, right):
        super().__init__()
        self.children.append(left)
        self.children.append(right)

class SimpleNext(SimpleSTL):
    def __init__(self, ap):
        super().__init__()
        self.children.append(ap)

class SimpleF(SimpleSTL):
    def __init__(self, ts, te, ap):
        super().__init__()
        self.ts = ts
        self.te = te
        self.children.append(ap)
    
    def print_out(self, mode="simple"):
        s = self.__class__.__name__
        if len(self.children)==1 and isinstance(self.children[0], SimpleReach):
            if mode=="simple":
                print(s + "(" + "Reach" + str(self.children[0].obj_id) + ")")
            else:
                print("%s[%d,%d] Reach (%s)"%(s, self.ts, self.te, self.children[0].obj_id))
        else:
            s_else = ",".join([child.__class__.__name__ for child in self.children])
            if mode=="simple":
                print(s + " | " + s_else)
            else:
                print("%s[%d,%d] | %s"%(s, self.ts, self.te, s_else))
            for child in self.children:
                child.print_out(mode)

class SimpleG(SimpleSTL):
    def __init__(self, ts, te, ap):
        super().__init__()
        self.ts = ts
        self.te = te
        self.children.append(ap)
        
    def print_out(self, mode="simple"):
        s = self.__class__.__name__
        s_else = ",".join([child.__class__.__name__ for child in self.children])
        if mode=="simple":
            print(s + " | " + s_else)
        else:
            print("%s[%d,%d] | %s"%(s, self.ts, self.te, s_else))
        for child in self.children:
            child.print_out(mode)

class SimpleUntil(SimpleSTL):
    def __init__(self, ts, te, left, right):
        super().__init__()
        self.ts = ts
        self.te = te
        self.children.append(left)
        self.children.append(right)

class SimpleReach(SimpleSTL):
    def __init__(self, obj_id, obj_x=None, obj_y=None, obj_z=None, obj_r=None, object=None, mode="2d", ap_type=None):
        super().__init__()
        self.obj_id = obj_id
        if object is not None:
            self.obj_x = object[0]
            self.obj_y = object[1]
            if mode=="2d":
                self.obj_z = 0
                self.obj_r = object[2]
                self.ap_type = ap_type
            else:
                assert mode=="panda"
                self.ap_type = ap_type
                self.obj_z = object[2]
                self.obj_r = object[3]
        else:
            self.obj_x = obj_x
            self.obj_y = obj_y
            self.obj_z = obj_z
            self.obj_r = obj_r
    
    def print_out(self, mode="simple"):
        s = self.__class__.__name__
        print(s + " obj_id: " + str(self.obj_id))

def plot_tree(stl):
    color_list = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "brown", "gray", "pink", "royalblue", "lightgray", "lightgreen", "darkgreen", "salmon", "lightblue"]
    stl_color_list = ["orange", "yellow", "red", None, None, "cyan", "green", "purple", "gray"]
    stl_alpha = 1.0
    stl_r = 0.2
    stl_dx = 2.0
    stl_dy = 0.8
    init_width = 10.0
    stl_lw = 1.5
    stl_line_color = "darkblue"
    stl_line_alpha = 0.8
    stl_node_str_d={0:"&", 1:"|", 2:"¬", 3:"->", 4:"O", 5:"F", 6:"G", 7:"U", 8:"R"}
    
    ax = plt.gca()
    # plot the tree
    total_id = 0
    queue = [(0, stl, 0, init_width, 0, None)]
    coords = {0:[0,0]}
    while len(queue)!=0:
        stl_id, node, depth, width, order, father = queue[0]
        del queue[0]
        
        # plot self node
        # type_i = 0
        base_x = coords[stl_id][0]
        base_y = coords[stl_id][1]
        type_i = check_stl_type(node)
        circ = Circle([base_x, base_y], radius=stl_r, color=stl_color_list[type_i], alpha=stl_alpha, zorder=999)
        ax.add_patch(circ)
        
        # plot text
        plt.text(base_x-0.05, base_y-0.1, stl_node_str_d[type_i], fontsize=12, zorder=1005)
        
        n_child = len(node.children)
        for new_i, new_node in enumerate(node.children):
            # update child pos
            new_y = base_y - stl_dy
            if n_child==1:
                new_x = base_x
            else:
                M = 2 * n_child
                left_x = base_x - width / 2
                right_x = base_x + width / 2
                new_x = (M-new_i * 2 - 1) / M * left_x + (new_i * 2 + 1) / M * right_x
            
            # plot lines
            plt.plot([base_x, new_x], [base_y, new_y], linewidth=stl_lw, color=stl_line_color, alpha=stl_line_alpha, zorder=1)
            coords[total_id] = [new_x, new_y]
            
            # insert to queue
            queue.append([total_id, new_node, depth+1, width/n_child, new_i, node])
            total_id += 1
    
    plt.axis("scaled")
    plt.xlim(-init_width/2, init_width/2)
    plt.ylim(-(init_width-.5), 0.5)
    
    return