
import networkx as nx
import gym
# from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor
# from IPython.display import SVG, display
import numpy as np
# import pogema
from pogema import GridConfig
# import matplotlib.pyplot as plt


from custom_networkx import astar_path, grid_graph
from history import History

# 0. Cкладываем матрицы агентов и препядствий (они пересеаться не будут)
# 1. Добавляем рамку по краям
# 2. Получаем координаты старта финиша
# 3. Получаем следующий шаг от А*
# 4. повторили упражнение
# 5. Завтра добавлю персонального картографа)

class Model:
    def __init__(self, history=None):
        self.agents = None
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id
        
        self.history = history
        count_agents = len(self.history.maps) 
    
        self.steps = [[0 for i in range(count_agents)]] if self.history is not None else None

        self.steps_corr = []

    def act(self, obs, dones, positions_xy, targets_xy, ) -> list:
        
        # Достаем предыдущий шаг
        
        # try:
        if self.history is not None:
            steps = self.steps[-1]  
            
            obs_new = self.history.update_history([i[0] for i in obs], steps)
            print('obs_new>>>\n', obs_new)
        # except:
        #     print('первый шаг')
       
        
        list_corr_agents = self.history.curr_pos
        # print('!!!!', list_corr_agents)

        # for i in obs[...,0]:

        # Для поcтройки всего пути - для штрафов. можем использовать 
        # self.steps_corr 

        
        def custom_concat_matrix(obs, agents):
            ...


        



        # Подготавливаем расширенную матрицу с окантовкой и смещенными точками старта\финиша
        a_ = []
        b_ = []
        edging_ = []
        for i in obs:
            pos_i_temp, pos_j_temp = np.where(i[2] == 1)
            a_temp = (int(i.shape[1]/2)+1,int(i.shape[1]/2)+1)

            # if not 0 in [z for z in i[0][...,0]]:
            #     edging_temp = np.pad(i[0]+i[1], pad_width=1, mode='constant', constant_values=0)[...,2:]
            #     a_temp = (a_temp[0], a_temp[1]-2)

            # elif not 0 in [z for z in i[0][...,-1]]:
            #     edging_temp = np.pad(i[0]+i[1], pad_width=1, mode='constant', constant_values=0)[...,:-2]

            # elif not 0 in [z for z in i[0][0,]]:
            #     edging_temp = np.pad(i[0]+i[1], pad_width=1, mode='constant', constant_values=0)[2:]
            #     a_temp = (a_temp[0]-2, a_temp[1])

            # elif not 0 in [z for z in i[0][-1,]]:
            #     edging_temp = np.pad(i[0]+i[1], pad_width=1, mode='constant', constant_values=0)[:-2]

            # else:
            edging_temp = np.pad(i[0]+i[1], pad_width=1, mode='constant', constant_values=0) #custom_concat_matrix(i[0],i[1])

            edging_.append(edging_temp)
            a_.append(a_temp)
            b_.append((pos_i_temp[0]+1, pos_j_temp[0]+1))

    
        # основной код - генерим граф делаем поиск А*
        next_step = []
        for edging, a, b in zip(edging_, a_, b_):
            G = nx.grid_graph(edging.shape)
            # G = grid_graph(edging.shape)
            x, y = np.where(edging ==1) 
            for i, j in zip(x,y):
                if (i,j) != a and (i,j) != b:
                    try:
                        G.remove_node((i,j))
                    except:
                        print('error', i, j)
            try:
                path = np.array(nx.astar_path(G, a, b)[1]) - a
                steps = {
                    0: (0, 0),
                    2: (1, 0),   
                    1: (-1, 0),  
                    4: (0, 1),   
                    3: (0, -1),  
                }
                next_step.append([k for k in steps if (steps[k] == path).all()][0])
            except:
                # исправить на действие доступное в области видимости
                next_step.append(np.random.randint(4))
        # проверка на пропуск хода
        for ind, i in enumerate(obs[:][1]):
            if i[5,6] == 1 or i[4,5] == 1 or i[4,6] == 1:
                # print('стоим на месте', obs[ind], f'\nsteps = {next_step}')
                next_step[ind] = 0
                path = (0,0)

        # Сохраняем текущий шаг
        self.steps.append(next_step)
        self.steps_corr.append(path)

        return next_step


def main():
    # Define random configuration
    grid_config = GridConfig(num_agents=32,  # количество агентов на карте
                             size=64,  # размеры карты
                             density=0.2,  # плотность препятствий
                             seed=6,  # сид генерации задания
                             max_episode_steps=256,  # максимальная длина эпизода
                             obs_radius=5,  # радиус обзора
                             )
    from pogema.wrappers.metrics import MetricsWrapper
    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)
    env = MetricsWrapper(env)
    obs = env.reset()

    done = [False for k in range(len(obs))]
    history = History([i[0] for i in obs])
    solver = Model(history) 

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
    print()

    print(CSR, ISR)




if __name__ == '__main__':
    main()