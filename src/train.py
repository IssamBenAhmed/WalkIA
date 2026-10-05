import gymnasium as gym
import os
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.envs.registration import register

# --- NOTRE CALLBACK SUR MESURE ---
class CustomInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_print_time = time.time()
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_distances = []
        self.ep_chutes = []
        
        self.step_angles = []
        self.step_hauteurs = []
        self.step_vitesses = []
        self.step_actions = []
        
        # Gestion du CSV avec pandas
        self.csv_path = "data/stats_robot.csv"
        self.nombre_total_episodes = 0
        self.episodes_session = 0
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                if not df.empty and "nombre_total_episodes" in df.columns:
                    self.nombre_total_episodes = int(df["nombre_total_episodes"].max())
            except Exception:
                pass

    def _on_step(self) -> bool:
        new_eps = 0
        for info in self.locals.get("infos", []):
            if "angle_torse" in info:
                self.step_angles.append(info["angle_torse"])
            if "hauteur_torse" in info:
                self.step_hauteurs.append(info["hauteur_torse"])
            if "vitesse_x" in info:
                self.step_vitesses.append(info["vitesse_x"])
            if "norme_actions" in info:
                self.step_actions.append(info["norme_actions"])
                
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                self.ep_distances.append(info.get("distance_parcourue_m", 0.0) * 100.0)
                self.ep_chutes.append(1 if info.get("est_tombe", True) else 0)
                new_eps += 1
                
        if new_eps > 0:
            self.episodes_session += new_eps
            self.nombre_total_episodes += new_eps

        current_time = time.time()
        if current_time - self.last_print_time >= 10.0:
            if len(self.ep_rewards) > 0:
                avg_rew = np.mean(self.ep_rewards[-20:]) 
                avg_len = np.mean(self.ep_lengths[-20:])
                avg_dist = np.mean(self.ep_distances[-20:])
                
                nb_eps_recent = min(20, len(self.ep_rewards))
                total_chutes = sum(self.ep_chutes[-20:])
                taux_succes = ((nb_eps_recent - total_chutes) / nb_eps_recent) * 100.0
                
                # Nouveaux indicateurs "camarade" (> 2s = 50 steps, > 5s = 125 steps)
                recent_lengths = self.ep_lengths[-20:]
                debout_2s = sum(1 for l in recent_lengths if l >= 50)
                debout_5s = sum(1 for l in recent_lengths if l >= 125)
                
                avg_angle = np.mean(self.step_angles) if len(self.step_angles) > 0 else 0.0
                avg_hauteur = np.mean(self.step_hauteurs) if len(self.step_hauteurs) > 0 else 0.0
                avg_vitesse = np.mean(self.step_vitesses) if len(self.step_vitesses) > 0 else 0.0
                avg_action = np.mean(self.step_actions) if len(self.step_actions) > 0 else 0.0
                
                stabilite = max(0.0, 100.0 - (avg_angle * 100.0))
                temps_debout = avg_len * 0.04
                
                log_data = {
                    "date_heure": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "nombre_total_episodes": self.nombre_total_episodes,
                    "episodes_session": self.episodes_session,
                    "recompense_moyenne": round(avg_rew, 2),
                    "duree_episode_moyenne_sec": round(temps_debout, 2),
                    "distance_moyenne_cm": round(avg_dist, 1),
                    "vitesse_avant_moyenne": round(avg_vitesse, 3),
                    "hauteur_torse_moyenne": round(avg_hauteur, 2),
                    "angle_torse_moyen": round(avg_angle, 2),
                    "force_action_moyenne": round(avg_action, 2),
                    "nombre_chutes_recents": total_chutes,
                    "taux_reussite_debout_pourcent": round(taux_succes, 1),
                    "succes_debout_2sec": debout_2s,
                    "succes_debout_5sec": debout_5s,
                    "score_stabilite_pourcent": round(stabilite, 1)
                }
                
                df = pd.DataFrame([log_data])
                if not os.path.isfile(self.csv_path):
                    df.to_csv(self.csv_path, index=False)
                else:
                    df.to_csv(self.csv_path, mode='a', header=False, index=False)
                
                print(f"📊 [10s] Réc: {avg_rew:5.1f} | Survie: {temps_debout:4.1f}s | Haut: {avg_hauteur:4.2f}m | Vit: {avg_vitesse:5.2f} | >2s: {debout_2s}/{nb_eps_recent} | Chutes: {total_chutes}")
                
                self.step_angles = []
                self.step_hauteurs = []
                self.step_vitesses = []
                self.step_actions = []
            else:
                print(f"⏳ [Bilan] En cours... Épisodes globaux atteints : {self.nombre_total_episodes}")
            self.last_print_time = current_time
        return True

# On déclare notre nouvel environnement
register(
    id="MonRobot-v0",
    entry_point="envs.mon_env:MonRobotEnv",
)

print("🌍 Chargement de l'environnement personnalisé...")

render_mode = "human" if os.getenv("MON_ROBOT_RENDER") == "1" else None

# En mode visuel (human), on force l'épisode à durer 10 secondes (250 actions de 0.04s).
# En arrière-plan (None), on lui laisse jusqu'à 40 secondes (1000 actions).
limite_temps = 250 if render_mode == "human" else 1000

env = gym.make("MonRobot-v0", render_mode=render_mode, max_episode_steps=limite_temps)

print("🧠 Initialisation de l'IA (PPO)...")
brain_file = "models/cerveau_mon_robot.zip"

# Chargement intelligent : On reprend là où on s'était arrêté !
if os.path.exists(brain_file):
    print("✅ Ancien cerveau trouvé ! Chargement en cours...")
    model = PPO.load("models/cerveau_mon_robot", env=env)
else:
    print("🆕 Aucun cerveau trouvé. Création d'un cerveau vierge.")
    model = PPO("MlpPolicy", env, verbose=0)


# =====================================================================
# SÉPARATION STRICTE : OBSERVATION vs ENTRAÎNEMENT
# =====================================================================
if render_mode == "human":
    print("👀 MODE OBSERVATION : L'IA utilise son cerveau mais N'APPREND PAS (pour ne pas fausser l'entraînement).")
    try:
        obs, info = env.reset()
        while True:
            # deterministic=True : Le robot utilise ses meilleures connaissances sans faire de hasard
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\n🛑 Observation interrompue manuellement (Ctrl+C).")
    except Exception as e:
        print(f"\n🛑 Observation arrêtée (Fenêtre fermée) : {e}")
    finally:
        env.close()
        import sys
        sys.exit(0)
else:
    print("🚀 MODE ENTRAÎNEMENT : Le robot apprend à toute vitesse en arrière-plan !")
    try:
        model.learn(total_timesteps=5000000, callback=CustomInfoCallback())
    except KeyboardInterrupt:
        print("\n🛑 Entraînement interrompu manuellement (Ctrl+C).")
    except Exception as e:
        print(f"\n🛑 Entraînement arrêté : {e}")
    finally:
        print("💾 Sauvegarde de sécurité du cerveau en cours...")
        model.save("models/cerveau_mon_robot")
        print("✅ Le cerveau a été sauvegardé avec succès dans 'cerveau_mon_robot.zip' !")
        env.close()
        import sys
        sys.exit(0)
