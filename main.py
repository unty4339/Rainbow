# -*- coding: utf-8 -*-
from __future__ import division

import argparse
import bz2
import os
import pickle
from datetime import datetime
from test import test

import atari_py
import numpy as np
import torch
from tqdm import trange

from agent import Agent
from env import Env
from memory import ReplayMemory

# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
# ハイパーパラメータは、エージェント ステップではなく ATARI ゲーム フレームで最初に報告される可能性があることに注意してください
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='乱数シード値')
parser.add_argument('--disable-cuda', action='store_true', help='CUDAを無効化する')
parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='ゲーム フレームの最大エピソード長 (0 で無効)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='連続して処理する状態の数')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='ネットワーク アーキテクチャ')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='NNの隠れ層の大きさ')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='noisy linear layersの初期標準偏差')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='価値分布学習のatomの大きさ')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='価値分布の最小値')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='価値分布の最大値')
parser.add_argument('--model', type=str, metavar='PARAMS', help='事前トレーニング済みモデル (状態辞書)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replayメモリ容量')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='メモリからのサンプリングの頻度')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='優先経験リプレイ指数 (元はα)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='優先経験リプレイの重点サンプリングの初期の重み')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='multi-step returnのステップ数')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='減衰率')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='targetネットワークを更新するまでのステップ数')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='報酬クリッピング (無効にする場合は 0)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='学習率')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adamのイプシロン')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='バッチサイズ')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='勾配クリッピングの最大 L2 ノルム')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='トレーニング開始前のステップ数')
parser.add_argument('--evaluate', action='store_true', help='評価のみ')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='評価と評価の間に行うトレーニングステップ数')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='平均する評価エピソードの数')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
# TODO: DeepMind の評価方法は、1M ステップごとに 500K フレームの最新エージェントを実行していることに注意してください
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Q の検証に使用するトランジションの数')
parser.add_argument('--render', action='store_true', help='画面表示(テストのみ)')
parser.add_argument('--enable-cudnn', action='store_true', help='cuDNN を有効にする (高速だが非決定的)')
parser.add_argument('--checkpoint-interval', default=0, help='モデルをチェックポイントする頻度。デフォルトは 0 (チェックポイントなし)')
parser.add_argument('--memory', help='メモリを保存/ロードするパス')
parser.add_argument('--disable-bzip-memory', action='store_true', help='メモリ ファイルを圧縮しないでください。推奨されません (圧縮は少し遅く、はるかに小さくなります)')

"""
# parser = argparse.ArgumentParser(description='Rainbow')
# parser.add_argument('--id', type=str, default='default', help='Experiment ID')
# parser.add_argument('--seed', type=int, default=123, help='Random seed')
# parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
# parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
# parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
# parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
# parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
# parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
# parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
# parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
# parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
# parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
# parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
# parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
# parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
# parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
# parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
# parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
# parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
# parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
# parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
# parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
# parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
# parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
# parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
# parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
# parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
# parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
# parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
# parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
# parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
# parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
# parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
# parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
# parser.add_argument('--memory', help='Path to save/load the memory from')
# parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
"""

# Setup
# セットアップ

# argsのインスタンス変数としてオプションで設定した値を取得できる
# --idオプションならargs.idなど
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
# seed値をランダムに振っている
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
# シンプルな ISO 8601 タイムスタンプ付きロガー
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


# Environment
# gymの環境
env = Env(args)
env.train()
action_space = env.action_space()

# Agent
dqn = Agent(args, env)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
# トレーニング済モデル用の引数modelが設定され、かつevaluateがfalseの場合、モデルの学習の再開と判断しメモリをロードする
if args.model is not None and not args.evaluate:
    if not args.memory:
        raise ValueError('Cannot resume training without memory save path. Aborting...')
    elif not os.path.exists(args.memory):
        raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

    mem = load_memory(args.memory, args.disable_bzip_memory)

else:
    mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
# 評価メモリを構築する
# スコア計測用にのみ使われる、更新しないリプレイバッファ？
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
    if done:
        state = env.reset()

    next_state, _, done = env.step(np.random.randint(0, action_space))
    val_mem.append(state, -1, 0.0, done)
    state = next_state
    T += 1

if args.evaluate:
    # Set DQN (online network) to evaluation mode
    # DQN(オンラインネットワーク)を評価モードに設定
    dqn.eval()
    avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
    print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
    # Training loop
    # 訓練用ループ
    dqn.train()
    done = True
    for T in trange(1, args.T_max + 1):
        if done:
            state = env.reset()

        # 引数replay_freaquency <- メモリからのサンプリングの頻度
        if T % args.replay_frequency == 0:
            # Draw a new set of noisy weights
            # ノイズの多い重みの新しい集合を描画する
            dqn.reset_noise()

        # Choose an action greedily (with noisy weights)
        # アクションを貪欲に選択する (ノイズ入りウェイトを使用)
        action = dqn.act(state)

        # Step
        next_state, reward, done = env.step(action)
        if args.reward_clip > 0:
            # Clip rewards
            # 報酬のクリップ
            reward = max(min(reward, args.reward_clip), -args.reward_clip)
        # Append transition to memory
        # 遷移情報をメモリに加える
        mem.append(state, action, reward, done)

    # Train and test
    # 訓練とテスト
        if T >= args.learn_start:
            # Anneal importance sampling weight β to 1
            # 重要度サンプリング重み β を 1 にアニールする
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

            # 引数replay_freaquency <- メモリからのサンプリングの頻度
            if T % args.replay_frequency == 0:
                # Train with n-step distributional double-Q learning
                # n ステップの分布型 double-Q 学習でトレーニングする
                dqn.learn(mem)

            # 引数evaluation_interval <- 評価と評価の間に行うトレーニングステップ数
            if T % args.evaluation_interval == 0:
                # Set DQN (online network) to evaluation mode
                # DQN(オンラインネットワーク)を評価モードに設定
                dqn.eval()
                # Test
                # テスト
                avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                # Set DQN (online network) back to training mode
                # DQN (オンライン ネットワーク) をトレーニング モードに戻す
                dqn.train()

                # If memory path provided, save it
                # メモリ パスが提供されている場合は保存する
                if args.memory is not None:
                    save_memory(mem, args.memory, args.disable_bzip_memory)

            # Update target network
            # ターゲット ネットワークを更新する
            if T % args.target_update == 0:
                dqn.update_target_net()

            # Checkpoint the network
            # ネットワークのチェックポイント（セーブ地点）
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                dqn.save(results_dir, 'checkpoint.pth')

        state = next_state

env.close()
