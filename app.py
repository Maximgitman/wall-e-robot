from flask import Flask, render_template, request
import gym
from pogema.animation import AnimationMonitor
import numpy as np
from pogema import GridConfig

app = Flask(__name__)

# Ensure templates are auto-reloaded
app.secret_key = "not-so-secret"
app.config["TEMPLATES_AUTO_RELOAD"] = True


# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
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
            obs, reward, done, info = env.step([get_action(o, i) for i, o in enumerate(obs)])

        img_path = "renders/render.svg"

        return render_template("index.html",
                               img_path=img_path)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)