
        
        
from calendar import c
from pickletools import optimize
import numpy as np
from torch.utils.data import Dataset
import random
import gym
from gym import spaces

from copy import deepcopy

from math import ceil, floor
import matplotlib.pyplot as plt
import os
import imageio

def clamp(num, min_value, max_value):
    num[0] = max(min(num[0], max_value), min_value)
    num[1] = max(min(num[1], max_value), min_value)
    return num


def draw_rect(im, x, y, channel, size):
    # Draws a rectangle, interpolating to allow floating point coords
    im[channel, ceil(x):floor(x + size), ceil(y):floor(y + size)] = 1.0

    # X borders
    if floor(x) != ceil(x):
        im[channel, floor(x), ceil(y):floor(y + size)] = ceil(x) - x
        im[channel, ceil(x + size - 1), ceil(y):floor(y + size)] = x - floor(x)

    # Y borders
    if floor(y) != ceil(y):
        im[channel, ceil(x):floor(x + size), floor(y)] = ceil(y) - y
        im[channel, ceil(x):floor(x + size), ceil(y + size - 1)] = y - floor(y)

    # Corners
    if floor(y) != ceil(y) and floor(x) != ceil(x):
        im[channel, floor(x), floor(y)] = (ceil(x) - x) * (ceil(y) - y)
        im[channel, floor(x), ceil(y + size - 1)] = (ceil(x) - x) * (y - floor(y))

        im[channel, ceil(x + size - 1), floor(y)] = (x - floor(x)) * (ceil(y) - y)
        im[channel, ceil(x + size - 1), ceil(y + size - 1)] = (x - floor(x)) * (y - floor(y))

    return im


class Node(object):
    def __init__(self, flow = None, parent = None, cost = None, is_root = False):
        self.flow = flow
        self.parent = parent
        self.cost = cost  # only used in RRT*
        self.is_root = is_root
        self.children=[]


class RRT(Node):
    def __init__(self, env, opt = "puck"):

        self.min_pos = 0.
        self.max_pos = 60.
        self.max_iters = 100
        self.max_dist = 2
        self._goal_sample_prob = 0.2
        self.tree = []
        self.env = env
        self.opt=opt
        if self.opt == "puck":
            self.curr_pos = env.curr_pos
            self.goal_flow = env.puck_pos
        if self.opt == "goal":
            self.goal_flow = env.goal_pos
            self.curr_pos = env.puck_pos
        #self.curr_pos=env.curr_pos
        self.goal_threshold = 2.
        self.path = []
        self.actions=[]
        self.num_interp = 2
        self.env = env
        self.node_collect=[]
    def get_env(self):
        return self.env
    
    def init_gridMap(self):
        grid = np.zeros((64, 64))
        # idx=np.random.randint(50,51,2)
        # print("!!!!!!!!!!!!",idx,idx.shape)
        # grid[ idx[0]-20:idx[0]+20,idx[1]-20:idx[1]+20]=1
        #print(grid)
        return grid
    def convert_coord2Index(self, coord, grid_shape):
        range_ = (self.max_pos - self.min_pos)
        min_coord_Map = np.array((self.min_pos, self.min_pos))
        index = np.floor((coord - min_coord_Map) / 64. * grid_shape).astype(np.int64)
        return index

    def sample(self):
        coord = np.random.uniform(self.min_pos, self.max_pos, 2)
        if random.random() < self._goal_sample_prob:
            coord = self.env.goal_pos
        return coord

    def new_optimized_action_image(self,path):
        actions=[]
        image_grid=np.zeros((64,64))
        grid_shape=image_grid.shape
        for i in range(1,len(path)):
            actions.append(path[i]-path[i-1])
            list_=np.linspace(path[i], path[i-1], 5, endpoint=False)
            for l in list_:
                idx_grid=self.convert_coord2Index(l, grid_shape)
                image_grid[idx_grid[0],idx_grid[1]]=1
        return actions,image_grid
            

    def get_path(self, node):
        path = [node.flow]
        actions=[]
        nodes=[]
        image_grid=np.zeros((64,64))
        grid_shape=image_grid.shape
        while node.parent is not None:
            #print(node.parent.flow-node.flow)
            actions.append(node.flow-node.parent.flow)
            nodes.append(node)
            list_=np.linspace(node.flow, node.parent.flow, 5, endpoint=False)
            for l in list_:
                idx_grid=self.convert_coord2Index(l, grid_shape)
                image_grid[idx_grid[0],idx_grid[1]]=1
            node = node.parent
            path.append(node.flow)




        path.reverse()
        actions.reverse()
        return path,actions,image_grid
    def interp_between_2points(self, sample_point, parent_point,num_interp):
        # print("!",sample_point[0],parent_point.shape)
        x_interp = np.linspace(sample_point[0], parent_point[0], num_interp)
        y_interp = np.linspace(sample_point[1], parent_point[1], num_interp)
        interp_point = np.vstack((x_interp, y_interp)).T
        # print("points",interp_point.shape)
        return interp_point

    def get_nearest_node_by_dist(self, flows, target, dict):
        # t = np.random.randn(4, 2)
        # print("flows",flows)
        # print("target",target)
        diff = flows - target
        # print("diff", diff)
        dist = np.linalg.norm(diff, axis = 1)
        # print("dictionary")
        # for x in dict.values():
        # print(x.flow)
        idx = np.argmin(dist)
        # print("dist", dist,idx)
        return dict[idx]



    def scale_target_flows(self, target_flows, parent_node):
        # print("----",target_flows,parent_node)
        dist = np.linalg.norm(target_flows - parent_node.flow)
        # print("1",dist)
        if dist > self.max_dist:
            target_flows = parent_node.flow + (target_flows - parent_node.flow) / dist * self.max_dist
        return target_flows

    def new_optimized_path(self,path):
        if len(path)<=2:
            return path
        else:
            new_path=[]
            diff = path[0] - path[-1]
            dist = np.linalg.norm(diff)
            num=ceil(dist/2.0)
            #print(dist)
            interp_points=self.interp_between_2points(path[0],path[-1],num)
            obstacle_collideing = True
            count=0
            for item in interp_points:
                if self.env._collision_check(item) == True:
                    break
                count = count + 1
            if count == len(interp_points):
                obstacle_collideing = False
            if obstacle_collideing==False:
                inner_points=np.linspace(path[0], path[-1], num)
                #print(inner_points)
                return list(inner_points)
            else:
                half_num=ceil(len(path)/2.0)
                left_p=self.new_optimized_path(path[:half_num])
                right_p=self.new_optimized_path(path[half_num:])
                left_p.extend(right_p[1:])
                return left_p
        return path 

    def plan(self,optimized):
        first_flow = self.curr_pos
        find=False
        # grid_map = self.init_gridMap()
        # self.convert_coord2Index(first_flow, grid_map)


        first_node = Node(flow = first_flow, parent = None, cost = None, is_root = True)
        self.tree = []
        self.nodes = [first_node]

        dic = {}
        node = first_node
        temp = node.flow.reshape(1, 2)
        dic[temp.shape[0] - 1] = first_node
        image=np.zeros((64,64))

        # if np.all(image==0):
        #     print("---------------------")
        for i in range(5000):
            target_flow = self.sample()
            parent_node = self.get_nearest_node_by_dist(temp, target_flow, dic)
            target_flow = self.scale_target_flows(target_flow, parent_node)
            interp_points = self.interp_between_2points(target_flow, parent_node.flow,2)
            obstacle_collideing = True
            count=0
            for item in interp_points:
                if self.env._collision_check(item) == True:
                    break
                count = count + 1
            if count == len(interp_points):
                obstacle_collideing = False
            if obstacle_collideing==False:
                new_node = Node(flow = target_flow, parent = parent_node, cost = None, is_root = False)
                self.tree.append(((new_node.flow[0], new_node.flow[1]),
                                  (parent_node.flow[0], parent_node.flow[1])))
                node = new_node
                node.parent.children.append(node)

                # print(temp.shape,temp)
                # print(node.flow.shape,node.flow)
                t = node.flow.reshape(1, 2)
                temp = np.concatenate((temp, t), axis = 0)
                # print("temp",temp)
                dic[temp.shape[0] - 1] = node
                self.nodes.append(node)
                # print(temp.shape,temp)
                # temp = torch.cat((temp, torch.unsqueeze(node.flows, dim = 0)), 0)

                diff = target_flow - self.goal_flow
                dist_ = np.linalg.norm(diff)
                # print(dist_)
                if dist_ < self.goal_threshold:
                    if self.opt == "puck":
                        #print("puck find")
                        find=True
                    if self.opt == "goal":
                        #print("goal find")
                        find=True
                    self.path,self.actions,image = self.get_path(new_node)
                    if optimized:
                        self.path=self.new_optimized_path(self.path)
                        self.actions,image=self.new_optimized_action_image(self.path)
                    # print(path)
                    break

        
        return self.path,self.actions,find,image

class MazeEnv(gym.Env):
    """ An openai gym environment with a moving block and an obstacle
    """

    def __init__(self, hand_env = True, large = True, random_max_objects = False):
        self.action_space = spaces.Box(-2, 2, shape = (2,), dtype = np.float32)
        self.width = 32
        if large:
            self.width = 64
        self.observation_space = spaces.Box(0, 1, shape = (3, self.width, self.width), dtype = np.float32)
        self.object_size = 4

        self.obs_width = 2
        self.min_length = 8
        self.max_length = 16

        self.min_objects = 2
        self.max_objects = 4
        if large:
            self.max_objects *= 3

        if random_max_objects:
            self.min_objects = 1
            self.max_objects = 6
            # self.max_objects=random.randint(self.min_objects, self.max_objects)

        self.goal_threshold = 2.#self.object_size  # 0.5
        self.hand_env = hand_env

        self.start_coord=None
        self.puck_coord=None
        self.goal_coord=None
        self.reset()

    def reset(self):

        self.obs_image = np.zeros((self.width, self.width))
        center = np.random.randint(0, self.width, size = 2)
        for i in range(np.random.randint(self.min_objects, self.max_objects)):

            if i % 2 == 0:  # np.random.random() < 0.5:
                obs_shape = np.array([self.obs_width // 2, np.random.randint(self.min_length, self.max_length) // 2])
            else:
                obs_shape = np.array([np.random.randint(self.min_length, self.max_length) // 2, self.obs_width // 2])

            if i != 0:
                if np.random.random() < 0.5:
                    center += obs_shape
                else:  # if np.random.random() < 0.5:
                    center -= obs_shape

            min_coords = np.clip(center - obs_shape, 0, self.width)
            max_coords = np.clip(center + obs_shape, 0, self.width)
            self.obs_image[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]] = 1.0
            if np.any(max_coords == 0) or (np.all(min_coords != 0) and np.random.random() < 0.5):
                center = min_coords
            else:
                center = max_coords
        colliding = True
        while colliding:
            self.curr_pos = np.random.random(size = 2) * (self.observation_space.shape[2] - self.object_size)
            #print("------------------",self.observation_space.shape[2] - self.object_size)
            colliding = self._collision_check(self.curr_pos)
        self.start_coord=deepcopy(self.curr_pos)
        colliding = True
        while colliding:
            self.goal_pos = np.random.random(size = 2) * (self.observation_space.shape[2] - self.object_size)
            colliding = self._collision_check(self.goal_pos)
        self.goal_coord=deepcopy(self.goal_pos)
        colliding = True
        while colliding:
            self.puck_pos = np.random.random(size = 2) * (self.observation_space.shape[2] - self.object_size)
            colliding = self._collision_check(self.puck_pos)
        self.puck_coord=deepcopy(self.puck_pos)


        return self.render()
        
    def path_tree_vis(self,tree_puck,tree_goal,path,idx,image1,image2, action_num,out_dir):
        plt.figure(figsize=(8.0, 8.0))
        plt.xlim((0, 64))
        plt.ylim((64, 0))
        for pos in tree_puck:
            plt.plot((pos[0][1], pos[1][1]), (pos[0][0], pos[1][0]), 'b')
        for pos in tree_goal:
            plt.plot((pos[0][1], pos[1][1]), (pos[0][0], pos[1][0]), 'orange')
        for i in range(1, len(path)):
            plt.plot((path[i - 1][1], path[i][1]), (path[i - 1][0], path[i][0]), 'r')   
        plt.imshow(image1, interpolation='nearest',alpha=0.3) 
        plt.imshow(image2, interpolation='nearest',alpha=0.3)
        plt.savefig(out_dir+"/"+"_traj_{}_action_{}.png".format(idx,action_num))




    def _collision_check(self, tmp_pos):
        tmp_image = np.zeros(self.observation_space.shape, dtype = np.float32)
        tmp_image[0, :, :] = self.obs_image
        # print("tmp_pos[0]",tmp_pos[0])
        # print("tmp_pos[1]",tmp_pos[1])
        tmp_image = draw_rect(tmp_image, tmp_pos[0], tmp_pos[1], 1, self.object_size)
        return np.any(np.sum(tmp_image, axis = 0) > 1)

    def step(self, action):
        tmp_pos = self.curr_pos + action

        if True:
            hand_diff = np.abs(self.curr_pos - self.puck_pos)

        # If out of bounds, don't move
        if np.any(tmp_pos < 0) or np.any(tmp_pos + self.object_size > self.observation_space.shape[1:]):
            tmp_pos = self.curr_pos

        # If collision, don't move
        if self._collision_check(tmp_pos):
            tmp_pos = self.curr_pos

        self.curr_pos = tmp_pos

        # If hand in contact, set puck pos to hand pos
        if True:
            if np.all(hand_diff < self.object_size):
                self.puck_pos = self.curr_pos
                #print("+++++++")

        im = self.render()
        dist = np.linalg.norm(self.curr_pos - self.puck_pos)
        # if self.hand_env:
        #     dist = np.linalg.norm(self.goal_pos - self.puck_pos)
        # else:
        #     dist = np.linalg.norm(self.goal_pos - self.curr_pos)
        reward = float(dist < self.goal_threshold)
        done = dist < self.goal_threshold
        return im, reward, done, {'curr_pos': self.curr_pos, 'obs_image': self.obs_image, 'goal_pos': self.goal_pos,
                                  'dist': dist}

    def render(self):
        image = np.zeros(self.observation_space.shape, dtype = np.float32)
        image[0, :, :] = self.obs_image
        image = draw_rect(image, self.curr_pos[0], self.curr_pos[1], 1, self.object_size)
        #image = draw_rect(image, self.goal_pos[0], self.goal_pos[1], 2, self.object_size)
        if self.hand_env:
            image = draw_rect(image, self.puck_pos[0], self.puck_pos[1], 2, self.object_size)
        return image

    def vis_goal(self):
        image = np.zeros(self.observation_space.shape, dtype = np.float32)
        image[0, :, :] = self.obs_image
        if self.hand_env:
            image = draw_rect(image, self.goal_pos[0], self.goal_pos[1], 2, self.object_size)
        else:
            image = draw_rect(image, self.goal_pos[0], self.goal_pos[1], 1, self.object_size)
        return image

class MazeDataset(Dataset):
    """ This dataset will return a sequence of images with a moving block
    """

    def __init__(self,
                 is_train = True,
                 sequence_length = 10,
                 context_length = 3,
                 num_rrt=1,
                 optimized=False,
                 **kwargs):
        self.is_train = is_train
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.env = MazeEnv(**kwargs)
        self.go_to_puck_prob = 1.0
        self.go_to_goal_prob = 1.0
        self.reach_puck = False
        self.num_rrt=num_rrt
        self.optimized=optimized


        if is_train == False:
            main_dir = "./dataset_"
            self.out_dir=main_dir+str(self.num_rrt)+"_rrt"
            os.makedirs(self.out_dir,exist_ok=True)
    def get_env(self):
        return self.env

    def __len__(self):
        return 10000
    
    def rrt_plan(self,idx,image,actions,action_num,rrt_node_puck,rrt_node_goal,optimized):
        np.random.seed(action_num)
        random.seed(action_num)
        findPath=False
        total_len_toUse = self.sequence_length + self.context_length
        total_len=30
        #rrt_node_puck = RRT(self.env,opt = "puck")
        path_puck,action_rrt_puck,find_puck,image_puck=rrt_node_puck.plan(optimized)
        tree_puck=rrt_node_puck.tree

        #rrt_node_goal = RRT(self.env, opt = "goal")
        env=rrt_node_puck.get_env()
        path_goal, action_rrt_goal,find_goal,image_goal = rrt_node_goal.plan(optimized)
        tree_goal=rrt_node_goal.tree

        done = True
        temp_path_puck=deepcopy(path_puck)

        action_len_puck=len(action_rrt_puck)
        action_len_goal = len(action_rrt_goal)

        action_len=action_len_puck+action_len_goal

        path_puck.extend(path_goal)
        action_rrt_puck.extend(action_rrt_goal)
        action_rrt=action_rrt_puck

        done=False
        if (action_len_puck!=0) and (action_len_goal !=0):
            for i in range(1, total_len):
                if i-1<action_len:
                    actions[i-1,:]=np.array(action_rrt[i-1])
                    image[i], _, done, inf = env.step(actions[i - 1, :])

                else:
                    actions[i - 1, :]=np.zeros_like(action_rrt[0])
                    image[i], _, done, inf = env.step(actions[i - 1, :])

        if done==True and np.all(image_puck==0)!=True and np.all(image_goal==0)!=True :
            findPath=True

            if idx>=0 and self.is_train==False:
                env.path_tree_vis(tree_puck,tree_goal,path_puck,idx,image_puck,image_goal,action_num,self.out_dir)

        image_toUse=image[:total_len_toUse]
        actions_toUse=actions[:total_len_toUse-1]



        sample = {'target_frames': image_toUse[self.context_length:],
                  'actions': actions_toUse[self.context_length - 1:],
                  'context_frames': image_toUse[:self.context_length],
                  'context_actions': actions_toUse[:self.context_length]}
        return sample,findPath,image_puck,image_goal

    def __getitem__(self, idx):
        findPath=False

        while findPath!=True:
            env=self.env.reset()
            total_len = 30  
            goal_=self.env.vis_goal()

            start_point=self.env.start_coord
            puck_point=self.env.puck_coord
            goal_point=self.env.goal_coord
            cur_point=self.env.curr_pos
            sample_list=[]
            rrt_node_puck = RRT(self.env, opt = "puck")
            rrt_node_goal = RRT(self.env, opt = "goal")
            for i in range(self.num_rrt):
                image = np.zeros((total_len, 3, self.env.width, self.env.width), dtype = np.float32)
                image[0]=env
                actions = np.zeros((total_len - 1, 2), dtype = np.float32)


                sample,findPath,image_puck,image_goal=self.rrt_plan(idx,image,actions,i,deepcopy(rrt_node_puck),deepcopy(rrt_node_goal),self.optimized)
                sample_list.append(sample)
            print("--------------------------------")
        print("idx",idx)
        if self.is_train==False:
            if idx>=0:
                im = (np.transpose(goal_, (1, 2, 0)) * 255.).astype(np.uint8) 
                plt.savefig(self.out_dir +"/goal{}.png".format(idx))

            for j in range(len(sample_list)):
                with imageio.get_writer(self.out_dir +'/traj_{}_action_{}.gif'.format(idx,j), mode='I',duration=1) as writer1:
                    for i in range(3):#data['target_frames'].shape[0]):
                        im = (np.transpose(sample['context_frames'][i], (1, 2, 0)) * 255.).astype(np.uint8)
                        writer1.append_data(im)
                    for i in range(10):#data['target_frames'].shape[0]):
                        im = (np.transpose(sample['target_frames'][i], (1, 2, 0)) * 255.).astype(np.uint8)
                        writer1.append_data(im)
        plt.close('all')
        if self.is_train==False:
            return sample,image_puck,image_goal,[start_point,puck_point,goal_point]
        else:
            return sample

if __name__ == '__main__':
    import cv2
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rrt", type=int, default=1)
    parser.add_argument("--dataset", action='store_false')
    parser.add_argument("--optimized", action='store_true')

    args = parser.parse_args()

    d = MazeDataset(is_train=args.dataset,num_rrt=args.num_rrt,optimized=args.optimized)

    # out_dir = "tmp5"
    # os.mkdir(out_dir)
    

    total_=0
    c=0
    amount_train=10000
    amount_test=100
    total_image_puck_train=np.zeros((amount_train,64,64))
    total_image_goal_train=np.zeros((amount_train,64,64))
    total_image_puck_test=np.zeros((amount_test,64,64))
    total_image_goal_test=np.zeros((amount_test,64,64))
    
    total_coord_puck_train=np.zeros((amount_train,2,2))
    total_coord_goal_train=np.zeros((amount_train,2,2))

    total_coord_puck_test=np.zeros((amount_test,2,2))
    total_coord_goal_test=np.zeros((amount_test,2,2))

    for j in range(amount_train+amount_test):
        if j < amount_train:
            data,image_puck,image_goal,coord_list = d[j]
            train_coord_puck_array=np.stack((coord_list[0],coord_list[1]))
            train_coord_goal_array=np.stack((coord_list[1],coord_list[2]))
            total_image_puck_train[j]= image_puck
            total_image_goal_train[j]= image_goal
            total_coord_puck_train[j]=train_coord_puck_array
            total_coord_goal_train[j]=train_coord_goal_array
        else:
            data,image_puck,image_goal,coord_list = d[j]
            test_coord_puck_array=np.stack((coord_list[0],coord_list[1]))
            test_coord_goal_array=np.stack((coord_list[1],coord_list[2]))
            total_image_puck_test[j-amount_train]= image_puck
            total_image_goal_test[j-amount_train]= image_goal
            total_coord_puck_test[j-amount_train]=test_coord_puck_array
            total_coord_goal_test[j-amount_train]=test_coord_goal_array
    
    
    total_image_puck_train=total_image_puck_train.reshape(amount_train,-1)
    total_image_goal_train=total_image_goal_train.reshape(amount_train,-1)
    total_coord_puck_train=total_coord_puck_train.reshape(amount_train,-1)
    total_coord_goal_train=total_coord_goal_train.reshape(amount_train,-1)
    
    np.savetxt('image_puck_train.csv', total_image_puck_train, delimiter=',')
    np.savetxt('image_goal_train.csv', total_image_goal_train, delimiter=',')
    np.savetxt('coord_puck_train.csv', total_coord_puck_train, delimiter=',')
    np.savetxt('coord_goal_train.csv', total_coord_goal_train, delimiter=',')


    total_image_puck_test=total_image_puck_test.reshape(amount_test,-1)
    total_image_goal_test=total_image_goal_test.reshape(amount_test,-1)
    total_coord_puck_test=total_coord_puck_test.reshape(amount_test,-1)
    total_coord_goal_test=total_coord_goal_test.reshape(amount_test,-1)

    np.savetxt('image_puck_test.csv', total_image_puck_test, delimiter=',')
    np.savetxt('image_goal_test.csv', total_image_goal_test, delimiter=',')
    np.savetxt('coord_puck_test.csv', total_coord_puck_test, delimiter=',')
    np.savetxt('coord_goal_test.csv', total_coord_goal_test, delimiter=',')

    #my_data_puck = np.genfromtxt('image_puck_train.csv', delimiter=',')
    # my_data_puck=my_data_puck.reshape(3,64,64)
    
    

    
    # my_data_coord = np.genfromtxt('coord_puck_train.csv', delimiter=',')
    # my_data_coord=my_data_coord.reshape(3,2,2)

    # fig = plt.figure()
    # for i in range(10):
    #     x=my_data_puck[i]
    #     c1=my_data_coord[i][0][0] 
    #     c2=my_data_coord[i][0][1]
    #     c3=my_data_coord[i][1][0]
    #     c4=my_data_coord[i][1][1]
    #     plt.figure(figsize=(8.0, 8.0))
    #     plt.xlim((0, 64))
    #     plt.ylim((64, 0))
    #     #print(x[].shape,c1.shape)
    #     plt.imshow(x)
    #     # plt.plot(my_data_coord_1[i][0][0],my_data_coord_1[i][0][1],'rp',markersize = 14)
    #     # plt.plot(my_data_coord_1[i][1][0],my_data_coord_1[i][1][1],'o',markersize = 14)
    #     plt.plot(c2,c1,'rp',markersize = 20,alpha=0.3)
    #     plt.plot(c4,c3,'o',markersize = 20,alpha=0.3)
    #     plt.show()


    
    
    # i=0
    # for data in my_data_puck:
    #     plt.figure(figsize=(8.0, 8.0))
    #     plt.xlim((0, 64))
    #     plt.ylim((0, 64))
    #     plt.imshow(data)
    #     plt.plot(my_data_coord[i][0][0],my_data_coord[i][0][1],'rp',markersize = 14)
    #     plt.plot(my_data_coord[i][1][0],my_data_coord[i][1][1],'o',markersize = 14)
    #     plt.show()
    #     i=i+1
    # my_data_goal = np.genfromtxt('image_puck.csv', delimiter=',')
    # my_data_goal=my_data_goal.reshape(3,64,64)
    # for data in my_data_goal:
    #     plt.figure(figsize=(8.0, 8.0))
    #     plt.xlim((0, 64))
    #     plt.ylim((0, 64))
    #     plt.imshow(data)
    #     plt.show() 
    #print(my_data.shape)

        # if inf is not None:
        #     print("dist:",inf["dist"])
    #     if done==True:
    #         total_+=length
    #         c+=1
    #
    #     print(done)
    # print(total_/c)
        # print(j)

        # for i in range(30):#data['target_frames'].shape[0]):
        #         #print(data['target_frames'][i].shape)
        #     im = (np.transpose(data['target_frames'][i], (1, 2, 0)) * 255.).astype(np.uint8)
        #     cv2.imwrite("tmp5/im_{}_{}.png".format(j, i), im)
        # print("----------------------")

