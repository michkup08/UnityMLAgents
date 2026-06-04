from typing import Dict, cast, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import attr

import copy
from mlagents.torch_utils import default_device
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.timers import timed

from mlagents.trainers.torch_entities.networks import NetworkBody
from mlagents.trainers.settings import NetworkSettings
from mlagents_envs.base_env import ObservationSpec

@attr.s(auto_attribs=True)
class TD3Settings(OffPolicyHyperparamSettings):
    """Specyficzne hiperparametry dla algorytmu TD3."""
    tau: float = 0.005  # Szybkość aktualizacji sieci docelowych (Soft Update)
    policy_noise: float = 0.2  # Szum dodawany do akcji docelowych (wygładzanie)
    noise_clip: float = 0.5  # Limit szumu (żeby Crawler nie zwariował)
    policy_delay: int = 2  # Co ile kroków aktualizować Aktora względem Krytyka
    gamma: float = 0.99  # Zniżka (Discount factor) - jak bardzo dbamy o przyszłe nagrody.

# ==========================================================
# 2. ARCHITEKTURA SIECI KRYTYKA DLA TD3 (Stan + Akcja)
# ==========================================================
class TD3CriticNetwork(nn.Module):
    """
    Sieć oceniająca funkcję Q. 
    Przyjmuje na wejściu: Obserwacje (co Crawler widzi) + Akcje (jak chce ruszyć nogami).
    Zwraca: Pojedynczą liczbę (przewidywaną sumę nagród z tego punktu).
    """
    def __init__(
        self, 
        observation_specs: List[ObservationSpec], 
        network_settings: NetworkSettings, 
        action_size: int
    ):
        super().__init__()
        
        # 1. Wbudowany moduł ML-Agents (NetworkBody)
        # Automatycznie obsługuje normalizację i kodowanie zmysłów (np. wizji i wektorów)
        self.network_body = NetworkBody(observation_specs, network_settings)
        
        # Po przepuszczeniu przez NetworkBody, obserwacje są "skompresowane" 
        # do wektora o rozmiarze zdefiniowanym w YAML (hidden_units)
        encoding_size = network_settings.hidden_units
        hidden_size = network_settings.hidden_units

        # 2. Właściwa sieć Krytyka (Q-network)
        # Wejście: zakodowane (i ZNORMALIZOWANE) obserwacje + akcje
        self.layer1 = nn.Linear(encoding_size + action_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)

    def forward(self, obs_list: List[torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        # UWAGA: NetworkBody przyjmuje LISTĘ obserwacji (tak jak zwraca środowisko), 
        # a nie ręcznie sklejony wektor!
        encoded_obs, _ = self.network_body(obs_list)
        
        # Łączymy zakodowany wektor środowiska z akcjami
        x = torch.cat([encoded_obs, action], dim=-1)
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        q_value = self.layer3(x)
        return q_value
    
# ==========================================================
# 3. GŁÓWNY OPTYMALIZATOR TD3 (MATEMATYKA)
# ==========================================================
class TorchTD3Optimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        
        # Pobieramy twarde ustawienia TD3 z naszej nowej klasy konfiguracyjnej
        self.td3_settings: TD3Settings = cast(TD3Settings, trainer_settings.hyperparameters)
        
        # Wyciągamy informacje o środowisku (ile oczu ma Crawler, ile ma stawów)
        # Zakładamy płaskie obserwacje (wektorowe, np. floaty z czujników)
        self.obs_size = sum([spec.shape[0] for spec in policy.behavior_spec.observation_specs])
        self.action_size = policy.behavior_spec.action_spec.continuous_size
        
        # --- TWORZENIE SIECI GŁÓWNYCH ---
        # Aktor to nasz decydent. Używamy wbudowanego z ML-Agents.
        self.actor = self.policy.actor
        
        # Tworzymy Podwójnych Krytyków (Twin Critics)
        obs_specs = policy.behavior_spec.observation_specs
        net_settings = trainer_settings.network_settings
        
        self.critic_1 = TD3CriticNetwork(obs_specs, net_settings, self.action_size).to(default_device())
        self.critic_2 = TD3CriticNetwork(obs_specs, net_settings, self.action_size).to(default_device())
        
        # --- TWORZENIE SIECI DOCELOWYCH (TARGET NETWORKS) ---
        # To są wolniej uczące się kopie głównych sieci. 
        # Służą do stabilizowania matematyki (jak stały punkt odniesienia).
        # --- TWORZENIE SIECI DOCELOWYCH (TARGET NETWORKS) ---
        self.actor_target = copy.deepcopy(self.actor).to(default_device())
        self.critic_1_target = copy.deepcopy(self.critic_1).to(default_device())
        self.critic_2_target = copy.deepcopy(self.critic_2).to(default_device())
        
        # --- OPTYMALIZATORY PYTORCH (Narzędzia liczące gradienty) ---
        lr = self.td3_settings.learning_rate
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.update_step = 0  # Licznik kroków potrzebny do mechanizmu opóźnienia (Delay)
        
    def _soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float):
        """Mechanizm płynnego aktualizowania sieci docelowych."""
        # Bierze po trochu (tau) z nowych wag i zostawia większość (1-tau) ze starych wag.
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _get_flattened_obs(self, batch: AgentBuffer, key: str) -> torch.Tensor:
        """Funkcja pomocnicza: pobiera z bufora listę zmysłów i skleja je w jeden wektor."""
        # ML-Agents traktuje zmysły jako listę. Odpakowujemy ją.
        n_obs = len(self.policy.behavior_spec.observation_specs)
        obs_list = ObsUtil.from_buffer(batch, n_obs, key)
        # Konwertujemy na tensory w pamięci GPU
        tensor_list = [ModelUtils.list_to_tensor(obs) for obs in obs_list]
        # Sklejamy wszystkie wektory zmysłów horyzontalnie (dim=1)
        return torch.cat(tensor_list, dim=1)

    # @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """GŁÓWNA PĘTLA UCZENIA TD3 z wstrzykniętą logiką OpenAI Spinning Up."""
        self.update_step += 1

        # --- Przygotowanie Danych ---
        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs_raw = ObsUtil.from_buffer(batch, n_obs)
        current_obs_list = [ModelUtils.list_to_tensor(o) for o in current_obs_raw]
        next_obs_list = ObsUtil.from_buffer_next(batch, n_obs)
        next_obs_list = [ModelUtils.list_to_tensor(o) for o in next_obs_list]
        actions = AgentAction.from_buffer(batch).continuous_tensor
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE]).unsqueeze(-1)
        
        rewards = torch.zeros_like(dones)
        for name in self.reward_signals:
            stream_rewards = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            ).unsqueeze(-1)
            rewards += stream_rewards

        # ---------------------------------------------------------
        # KROK 1: Aktualizacja Krytyków (Oceniaczy) - Równanie Bellmana
        # ---------------------------------------------------------
        with torch.no_grad(): 
            next_action_out, _, _ = self.actor_target.get_action_and_stats(next_obs_list)
            # OpenAI: Akcje zawsze muszą być w zakresie [-1, 1]
            next_actions = torch.tanh(next_action_out.continuous_tensor)
            
            # OpenAI: Target Policy Smoothing (Szum wygładzający politykę docelową)
            epsilon = torch.randn_like(next_actions) * self.td3_settings.policy_noise
            epsilon = torch.clamp(epsilon, -self.td3_settings.noise_clip, self.td3_settings.noise_clip)
            
            a2 = next_actions + epsilon
            a2 = torch.clamp(a2, -1.0, 1.0) # Zakładamy limity akcji Unity [-1, 1]
            
            target_Q1 = self.critic_1_target(next_obs_list, a2)
            target_Q2 = self.critic_2_target(next_obs_list, a2)
            
            target_Q = torch.min(target_Q1, target_Q2)
            backup = rewards + self.td3_settings.gamma * (1.0 - dones) * target_Q

        current_Q1 = self.critic_1(current_obs_list, actions)
        current_Q2 = self.critic_2(current_obs_list, actions)

        # OpenAI używa MSE (Mean Squared Error) zamiast Smooth L1
        critic_1_loss = F.mse_loss(current_Q1, backup)
        critic_2_loss = F.mse_loss(current_Q2, backup)

        # Optymalizacja Krytyka 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # Optymalizacja Krytyka 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ---------------------------------------------------------
        # KROK 2: Aktualizacja Aktora (Delayed Policy Update)
        # ---------------------------------------------------------
        policy_loss_val = 0.0
        
        if self.update_step % self.td3_settings.policy_delay == 0:
            # Zamrażamy Krytyka, żeby nie liczyć dla niego niepotrzebnych gradientów
            for param in self.critic_1.parameters():
                param.requires_grad = False

            actor_action_out, _, _ = self.actor.get_action_and_stats(current_obs_list)
            actor_actions = torch.tanh(actor_action_out.continuous_tensor)
            
            actor_Q1 = self.critic_1(current_obs_list, actor_actions)
            
            # WSTRZYKNIĘCIE OPENAI: Czysta funkcja straty aktora. 
            # Bez udziwnień, bez kar L2 za logity, bez normalizacji std!
            actor_loss = -actor_Q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Odmrażamy Krytyka
            for param in self.critic_1.parameters():
                param.requires_grad = True
            
            policy_loss_val = actor_loss.item()

            # Soft Update sieci docelowych (Polyak Averaging z OpenAI)
            self._soft_update(self.actor_target, self.actor, self.td3_settings.tau)
            self._soft_update(self.critic_1_target, self.critic_1, self.td3_settings.tau)
            self._soft_update(self.critic_2_target, self.critic_2, self.td3_settings.tau)

        # ---------------------------------------------------------
        # KROK 3: Statystyki
        # ---------------------------------------------------------
        return {
            "Losses/Critic 1 Loss": critic_1_loss.item(),
            "Losses/Critic 2 Loss": critic_2_loss.item(),
            "Losses/Actor Loss": policy_loss_val,
        }
    
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Aktualizuje mechanizmy normalizacji w sieciach Krytyków poprzez 
        skopiowanie najświeższych statystyk od Aktora.
        """
        # Aktor ma już zaktualizowane statystyki (robione w trainer_td3.py),
        # więc wystarczy je bezpiecznie skopiować do wszystkich naszych Krytyków.
        self.critic_1.network_body.copy_normalization(self.actor.network_body)
        self.critic_2.network_body.copy_normalization(self.actor.network_body)
        
        self.critic_1_target.network_body.copy_normalization(self.actor.network_body)
        self.critic_2_target.network_body.copy_normalization(self.actor.network_body)

        self.actor_target.network_body.copy_normalization(self.actor.network_body)

    def get_trajectory_value_estimates(
        self, batch: AgentBuffer, next_obs: List[np.ndarray], done: bool
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[np.ndarray]]:
        """
        Ocenia zebraną właśnie trajektorię. 
        Zwraca strukturę potrzebną klasie bazowej do narysowania wykresów TensorBoard.
        """
        with torch.no_grad():
            n_obs = len(self.policy.behavior_spec.observation_specs)
            
            # Pobieramy i sklejamy zmysły
            obs_raw = ObsUtil.from_buffer(batch, n_obs)
            obs_list = [ModelUtils.list_to_tensor(o) for o in obs_raw]
            
            # Pobieramy wykonane akcje z tej ścieżki
            actions = AgentAction.from_buffer(batch).continuous_tensor
            
            # Pytamy pierwszego Krytyka: "Ile przewidujesz punktów za to, co on właśnie zrobił?"
            q_values = self.critic_1(obs_list, actions)

        # Odklejamy to z GPU, przerabiamy na numpy, bo TensorBoard przyjmuje czyste liczby
        value_estimates = {"extrinsic": q_values.cpu().numpy()}

        # Zwracamy słownik dla loggera. Dwa ostatnie parametry to puste wartości, 
        # bo TD3 (w przeciwieństwie do PPO) nie używa skomplikowanej wbudowanej pamięci PPO (critic memories)
        return value_estimates, {}, None

    def get_modules(self):
        """Rejestruje stworzone moduły (dla zapisywania i wznawiania modelu)."""
        # Samodzielnie budujemy słownik ze wszystkimi elementami, które chcemy zapisać
        modules = {
            "Optimizer:actor_optimizer": self.actor_optimizer,
            "Optimizer:critic_1_optimizer": self.critic_1_optimizer,
            "Optimizer:critic_2_optimizer": self.critic_2_optimizer,
            "Optimizer:td3_critic_1": self.critic_1,
            "Optimizer:td3_critic_2": self.critic_2,
        }
        
        # Musimy też pamiętać o zapisaniu stanu modułów od nagród (np. extrinsic)
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
            
        return modules