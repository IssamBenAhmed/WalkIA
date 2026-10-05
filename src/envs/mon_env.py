import numpy as np
import os
import time
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class MonRobotEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        xml_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "mon_robot.xml")
        observation_space = Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=4,
            observation_space=observation_space,
            **kwargs
        )

    def _torso_height(self):
        torso_id = self.model.body("torso").id
        return float(self.data.xpos[torso_id][2])

    def _body_height(self, body_name):
        body_id = self.model.body(body_name).id
        return float(self.data.xpos[body_id][2])

    def _count_contacts_with_geoms(self, geom_names):
        count = 0
        for c in self.data.contact:
            geom1 = self.model.geom(c.geom1).name
            geom2 = self.model.geom(c.geom2).name
            if geom1 in geom_names or geom2 in geom_names:
                count += 1
        return count

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]

        torso_height = self._torso_height()
        torso_pitch = float(self.data.qpos[2])

        # Hauteurs utiles pour détecter les mauvais appuis (genoux)
        knee_names = [
            "mollet_avant_gauche",
            "mollet_avant_droite",
            "mollet_arriere_gauche",
            "mollet_arriere_droite",
        ]
        knee_height = min(self._body_height(name) for name in knee_names)

        # Geoms de palmes / pieds
        palm_geoms = [
            "palme_avant_gauche_geom",
            "palme_avant_droite_geom",
            "palme_arriere_gauche_geom",
            "palme_arriere_droite_geom",
        ]
        palm_contacts = self._count_contacts_with_geoms(palm_geoms)

        # Vitesse avant
        vitesse_x = (x_after - x_before) / self.dt

        # --------------------------------------------------
        # Récompense de stabilité "d'abord"
        # --------------------------------------------------
        # 1. Le torse doit rester haut, mais pas trop haut
        if 0.7 <= torso_height <= 0.9:
            reward_stability = 5.0
        elif torso_height > 0.9:
            reward_stability = -2.0
        elif torso_height > 0.55:
            reward_stability = 1.0
        else:
            reward_stability = -5.0

        # 2. Le torse doit rester droit
        if abs(torso_pitch) < 0.35:
            reward_posture = 3.0
        elif abs(torso_pitch) < 0.6:
            reward_posture = 1.0
        else:
            reward_posture = -4.0

        # 3. Les palmes doivent porter le poids
        reward_palmes = 1.5 * (palm_contacts / 4.0)

        # 4. L'IA doit avancer doucement, sans se vautrer
        if vitesse_x > 0:
            reward_avancer = 2.0 * vitesse_x
        else:
            reward_avancer = -1.0 * abs(vitesse_x)

        # 5. Coût d'énergie
        cout_energie = 0.03 * np.sum(np.square(action))

        # 6. Pénalité si le robot s'appuie sur ses genoux au lieu des palmes
        # Le robot "essaye de tricher" en restant sur ses genoux
        bad_knee_support = (
            torso_height < 0.75
            and knee_height < 0.55
            and palm_contacts == 0
        )
        punition_genoux = 12.0 if bad_knee_support else 0.0

        # 7. Chute
        punition_chute = 20.0 if torso_height < 0.55 else 0.0

        reward = (
            reward_stability
            + reward_posture
            + reward_palmes
            + reward_avancer
            - cout_energie
            - punition_genoux
            - punition_chute
        )

        # -------------------------------
        # Condition de fin d'épisode
        # -------------------------------
        terminated = bool(torso_height < 0.55)

        # Dans le mode visuel, on peut garder l'observation sans tuer l'épisode immédiatement
        if self.render_mode == "human":
            self.render()
            terminated = False  # Permet d'observer les ratés sans Game Over instantané

        observation = self._get_obs()

        info = {
            "vitesse_x": vitesse_x,
            "cout_energie": float(cout_energie),
            "distance_parcourue_m": float(x_after),
            "angle_torse": float(abs(torso_pitch)),
            "hauteur_torse": float(torso_height),
            "palmes_contacts": int(palm_contacts),
            "genoux_basses": bool(bad_knee_support),
            "est_tombe": bool(terminated),
            "norme_actions": float(np.mean(np.abs(action))),
        }

        return observation, reward, terminated, False, info

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
