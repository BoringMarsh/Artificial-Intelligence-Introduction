import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)  # 初始化游戏环境
    model = DeepQNetwork()  # 实例化DQN网络
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)  # 优化器为Adam

    criterion = nn.MSELoss()

    state = env.reset()  # 重置游戏
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)  # 初始化回访缓存
    epoch = 0          # 训练时期数
    t1 = time.time()   # 总训练时间计时起点
    total_time = 0     # 总训练时间
    best_score = 1000  # 当前最高分

    while epoch < opt.num_epochs:
        start_time = time.time()  # 本轮计时起点
        next_steps = env.get_next_states()  # 获取下一状态
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)  # 计算epsilon
        u = random()  # 取随机数
        random_action = u <= epsilon  # 判断是否随机选择动作

        next_actions, next_states = zip(*next_steps.items())  # 取下一步所有动作和状态
        next_states = torch.stack(next_states)

        if torch.cuda.is_available():
            next_states = next_states.cuda()

        model.eval()

        with torch.no_grad():  # 根据下一步状态预测Q值
            predictions = model(next_states)[:, 0]

        model.train()

        # 有epsilon的可能随机选择动作，否则选择预测Q值最大的动作
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]  # 根据决策取下一步状态
        action = next_actions[index]  # 取下一步动作

        reward, done = env.step(action, render=True)  # 计算奖励值和游戏是否结束

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])  # 回放至replay_memory

        if done:
            final_score = env.score                  # Score
            final_tetrominoes = env.tetrominoes      # Pieces
            final_cleared_lines = env.cleared_lines  # Cleared Lines
            state = env.reset()  # 重置游戏
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:  # 若epoch到replay_memory大小的倍数则继续执行，否则开始新循环
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))  # 从replay_memory中随机取一批样本
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))  # 取状态批次
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])  # 取奖励批次
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))  # 取下一步状态批次

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        print("state_batch",state_batch.shape)
        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        # 计算loss并执行梯度下降
        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        use_time = end_time - t1 - total_time
        total_time = end_time - t1
        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}, Used time: {}, total used time: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines,
            use_time,
            total_time))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:  # 在更新频率倍数时保存模型
            print("save interval model: {}".format(epoch))
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))
        elif final_score > best_score:  # 最高分记录打破时保存模型
            best_score = final_score
            print("save best model: {}".format(best_score))
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, best_score))


if __name__ == "__main__":
    opt = get_args()
    train(opt)