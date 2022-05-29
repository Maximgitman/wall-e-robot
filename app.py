from crypt import methods
from flask import Flask, render_template, request
from pogema import GridConfig
from pogema.animation import AnimationMonitor
import gym
import numpy as np


app = Flask(__name__)


# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/create", methods=["GET", "POST"])
def create():
    num_agents = request.form.get("num_agents")
    map_size = request.form.get("map_size")
    density = request.form.get("density")

    # Const
    seed = np.random.randint(0, 345546)

    grid_config = GridConfig(num_agents=num_agents,  # количество агентов на карте
                                size=map_size,  # размеры карты
                                density=density,  # плотность препятствий
                                seed=seed,  # сид генерации задания
                                max_episode_steps=256,  # максимальная длина эпизода
                                obs_radius=5,  # радиус обзора
                                )
    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    done = [False, ...]

    while not all(done):
        # Используем случайную стратегию
        obs, reward, done, info = env.step([np.random.randint(4) for _ in range(len(obs))])

    env.save_animation("static/render.svg", egocentric_idx=0)
    
    img_path = "static/render.svg"

    return render_template("create.html",
                            img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)