
import networkx as nx
import gym
# from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor
# from IPython.display import SVG, display
import numpy as np
# import pogema
from pogema import GridConfig
# import matplotlib.pyplot as plt


# from custom_networkx import astar_path, grid_graph
from history import History
from model import Model


def main(grid_config):
    # Define random configuration
    
    from pogema.wrappers.metrics import MetricsWrapper
    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)
    env = MetricsWrapper(env)
    obs = env.reset()

    done = [False for k in range(len(obs))]
    history = History([i[0] for i in obs])
    solver = Model(history) #history

    while not all(done):
        # Используем AStar
        obs, reward, done, info = env.step(solver.act(obs, done,
                                                      env.get_agents_xy_relative(),
                                                      env.get_targets_xy_relative()),
                                                      )                                           
    # сохраняем анимацию и рисуем ее
    env.save_animation("render_1.svg", egocentric_idx=None)
    CSR = info[0]['metrics'].get('CSR')
    ISR = np.mean([x['metrics'].get('ISR',0) for x in info])
    return CSR, ISR



if __name__ == '__main__':
    csr = []
    isr= []
    for i in range(10):
        print('\n>>>',i)
        grid_config = GridConfig(num_agents=32,  # количество агентов на карте
                                size=64,  # размеры карты
                                density=0.2,  # плотность препятствий
                                seed=np.random.randint(100),  # сид генерации задания
                                max_episode_steps=256,  # максимальная длина эпизода
                                obs_radius=5,  # радиус обзора
                                )
        CSR, ISR =  main(grid_config)
        print(CSR, ISR)
        csr.append(CSR)
        isr.append(ISR)
    print(csr)
    print('CSR = ', np.mean(csr), 'ISR = ',  np.mean(isr),'\n')
