
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
    
        self.steps = [[0 for i in range(count_agents)]] if self.history is not None else []
        self.steps_corr = []
        self.errors = 0

    def act(self, obs, dones, positions_xy, targets_xy, ) -> list:
        
        # Достаем предыдущий шаг

        if self.history is not None:
            steps = self.steps[-1]  
            
            self.history.update_history([i[0] for i in obs], steps)

        
        def custom_concat_matrix(iter_, agents_one):
             # обновляем карту агента
            x, y  = self.history.curr_pos[iter_]
            # print(x, y)
            temp_maps = self.history.maps[iter_].copy()
            temp_maps[x-5:x+6, y-5:y+6] += agents_one
            # print(temp_maps.shape)
            return temp_maps

        def veiw_shot_list(res, x, y):
            x, y = np.where(res[x-1:y+2,x-1:y+2] == 0)
            path = np.array([(i,j) for i, j in zip(x, y)]) - (1,1)
            steps = {
                    0: (0, 0),
                    2: (1, 0),   
                    1: (-1, 0),  
                    4: (0, 1),   
                    3: (0, -1),  
                }
            x_ = [[k for k in steps if (steps[k] == i).all()] for i in path]
            return [i for j in x_ for i in j]

        

        # Подготавливаем расширенную матрицу с окантовкой и смещенными точками старта\финиша
        a_ = []
        b_ = []
        edging_ = []
        for ind, i in enumerate(obs):
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
            edging_temp = np.pad(custom_concat_matrix(ind, i[1]), pad_width=1, mode='constant', constant_values=0) #custom_concat_matrix(ind, i[1])   i[0]+i[1]
            edging_.append(edging_temp)
            # a_.append(a_temp)
            x, y  = self.history.curr_pos[ind]
            a_.append((x+1, y+1))
            b_.append((x-5+pos_i_temp[0]+1, y-5+pos_j_temp[0]+1))
            # b_.append((pos_i_temp[0]+1, pos_j_temp[0]+1))

        # основной код - генерим граф делаем поиск А*
        k = 0
        next_step = []
        for edging, a, b in zip(edging_, a_, b_):
            G = nx.grid_graph(edging.shape)
            # G = grid_graph(edging.shape)
            x, y = np.where(edging ==1) 
            for i, j in zip(x,y):
                if (i,j) != a and (i,j) != b:
                    try:
                        G.remove_node((i,j))
                    except:...
                        # print('error', i, j)

            if edging[a[0],a[1]] == 1:
                try:
                    try:
                        path_all = nx.astar_path(G, a, b)
                    except:...

                    path = np.array(path_all[1]) - a
                    steps = {
                        0: (0, 0),
                        2: (1, 0),   
                        1: (-1, 0),  
                        4: (0, 1),   
                        3: (0, -1),  
                    }
                    # nx.draw(G,)
                    step_bot = [k for k in steps if (steps[k] == path).all()][0]
                    if step_bot == self.steps[-1][k]:
                        true_step = veiw_shot_list(obs[k][0]+obs[k][1], 5,5)
                        true_step.pop(step_bot)
                        true_step.pop(self.steps[-2][k])
                        next_step.append(np.random.choice(true_step)) 
                        print(step_bot,'--', true_step)
                    else:
                        next_step.append(step_bot)
                except:
                    self.errors +=1
                    
                    # исправить на действие доступное в области видимости
                    # print('AAAAAAuhhh    a = ', a)
                    true_step = veiw_shot_list(obs[k][0]+obs[k][1], 5,5)
                    # v = [1 for i in range(len(true_step))]
                    # v_ = np.exp(v)/np.sum(np.exp(v))
                    next_step.append(np.random.choice(true_step)) 
            else:
                next_step.append(0)
            k+=1

        # проверка на пропуск хода
        for ind, i in enumerate([i[1] for i in obs]):
            # if (i[4,4] == 1 or i[5,6] == 1 or i[4,5] == 1 or i[4,6] == 1 or i[6,6] == 1) and  i[5,5] == 1:
            if (i[5,6] == 1 or i[4,5] == 1) and  i[5,5] == 1:
                # print('стоим на месте')
                next_step[ind] = 0
                path = (0,0)

        # Проверка на зацикленность
        try:
            # по агентам
            for ind, i in enumerate(range(len(obs))):
                history_path = [j[i] for j in self.steps[-4:]]
                x_ = np.unique(history_path[::2])
                x_1 = np.unique(history_path[1::2])
                if len(x_) == 1  and len(x_1) == 1:
                    # print('Врубаем жесткий рандом!!!!!')
                    temp_step = next_step[ind]
                    true_step = veiw_shot_list(edging, 5, 5)
                    true_step.pop(temp_step)
                    v = [1 for i in range(len(true_step))]
                    try:
                        ind_1 = true_step.index(x_)
                        v[ind_1] = v[ind_1]/2
                    except:...
                    try:
                        ind_2 =  true_step.index(x_)
                        v[ind_2] = v[ind_2]/2
                    except:...
                    try:
                        ind_3 = true_step.index(0)
                        v[ind_3] = v[ind_3]/2
                    except:...
                    v_ = np.exp(v)/np.sum(np.exp(v))
                    next_step[ind] = np.random.choice(true_step, p=v_)
        except:...
        # Сохраняем текущий шаг
        self.steps.append(next_step)
        # self.steps_corr.append(path)

        return next_step


def main():
    # Define random configuration
    grid_config = GridConfig(num_agents=32,  # количество агентов на карте
                             size=64,  # размеры карты
                             density=0.3,  # плотность препятствий
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
    print('CSR = ', CSR, 'ISR = ',  ISR,'\n')
    print(f'errors = {solver.errors}')

if __name__ == '__main__':
    main()