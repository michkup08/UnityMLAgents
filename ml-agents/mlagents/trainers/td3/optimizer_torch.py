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
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings, TrainerSettings
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.timers import timed

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
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        # Prosta 3-warstwowa sieć Multi-Layer Perceptron (MLP)
        # Wejście to suma rozmiaru czujników i rozmiaru wektora akcji
        self.layer1 = nn.Linear(obs_size + action_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)  # Wyjście to zawsze 1 liczba (ocena)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Sklejamy obserwacje i akcje w jeden długi wektor: [obserwacje, akcje]
        x = torch.cat([obs, action], dim=-1)
        
        # Przepuszczamy sygnał przez warstwy z funkcją aktywacji ReLU (łamie liniowość)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        q_value = self.layer3(x)  # Ostatnia warstwa bez aktywacji (chcemy surowy wynik)
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
        self.critic_1 = TD3CriticNetwork(self.obs_size, self.action_size).to(default_device())
        self.critic_2 = TD3CriticNetwork(self.obs_size, self.action_size).to(default_device())
        
        # --- TWORZENIE SIECI DOCELOWYCH (TARGET NETWORKS) ---
        # To są wolniej uczące się kopie głównych sieci. 
        # Służą do stabilizowania matematyki (jak stały punkt odniesienia).
        self.actor_target = copy.deepcopy(self.actor).to(default_device())
        self.critic_1_target = copy.deepcopy(self.critic_1).to(default_device())
        self.critic_2_target = copy.deepcopy(self.critic_2).to(default_device())
        
        # --- OPTYMALIZATORY (Narzędzia liczące gradienty i zmieniające wagi) ---
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
        """GŁÓWNA PĘTLA UCZENIA TD3. Uruchamiana z paczką zebranych danych."""
        self.update_step += 1

        # ---------------------------------------------------------
        # KROK 1: Przygotowanie Danych z Bufora
        # ---------------------------------------------------------
        n_obs = len(self.policy.behavior_spec.observation_specs)
        
        # Stan obecny (S)
        current_obs_raw = ObsUtil.from_buffer(batch, n_obs)
        current_obs_list = [ModelUtils.list_to_tensor(o) for o in current_obs_raw]
        obs = torch.cat(current_obs_list, dim=1)  # Sklejony wektor dla Krytyka

        # Stan następny (S') - tam gdzie ziutek wylądował po zrobieniu kroku
        next_obs_list = ObsUtil.from_buffer_next(batch, n_obs)
        next_obs_list = [ModelUtils.list_to_tensor(o) for o in next_obs_list]
        next_obs = torch.cat(next_obs_list, dim=1)
        
        # Akcje (A)
        actions = AgentAction.from_buffer(batch).continuous_tensor
        
        # Nagrody (R) i status końca epizodu (D)
        rewards = ModelUtils.list_to_tensor(batch[BufferKey.ENVIRONMENT_REWARDS]).unsqueeze(-1)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE]).unsqueeze(-1)

        # ---------------------------------------------------------
        # KROK 2: Aktualizacja Krytyków (Oceniaczy)
        # ---------------------------------------------------------
        with torch.no_grad(): 
            # UWAGA: Aktor dostaje LISTĘ (next_obs_list), a nie sklejony wektor!
            next_action_out, _, _ = self.actor_target.get_action_and_stats(next_obs_list)
            # next_actions = next_action_out.continuous_tensor
            next_actions_raw = next_action_out.continuous_tensor
            next_actions = torch.tanh(next_actions_raw)
            
            # Target Policy Smoothing
            noise = torch.randn_like(next_actions) * self.td3_settings.policy_noise
            noise = noise.clamp(-self.td3_settings.noise_clip, self.td3_settings.noise_clip)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            
            # Krytyk dostaje sklejony wektor (next_obs)
            target_Q1 = self.critic_1_target(next_obs, next_actions)
            target_Q2 = self.critic_2_target(next_obs, next_actions)
            
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1.0 - dones) * self.td3_settings.gamma * target_Q

        current_Q1 = self.critic_1(obs, actions)
        current_Q2 = self.critic_2(obs, actions)

        critic_1_loss = F.mse_loss(current_Q1, target_Q)
        critic_2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ---------------------------------------------------------
        # KROK 3: Aktualizacja Aktora
        # ---------------------------------------------------------
        policy_loss_val = 0.0
        
        if self.update_step % self.td3_settings.policy_delay == 0:
            # UWAGA: Aktor znów dostaje LISTĘ
            actor_action_out, _, _ = self.actor.get_action_and_stats(current_obs_list)
            # actor_actions = actor_action_out.continuous_tensor
            actor_actions_raw = actor_action_out.continuous_tensor
            actor_actions = torch.tanh(actor_actions_raw)
            
            # Krytyk ocenia wymyślone akcje na podstawie sklejonego wektora
            actor_Q1 = self.critic_1(obs, actor_actions)
            
            # Penalizujemy akcje dążące do skrajności (-1.0 lub 1.0)
            # Zmusza to sieć do unikania martwych stref funkcji Tanh.
            action_l2_penalty = (actor_actions_raw ** 2).mean()
            
            # Obliczamy błąd Aktora: chce maksymalizować Q, minimalizując spięcie mięśni
            # Waga 0.01 jest wystarczająca, by zapobiec zastyganiu bez psucia chodu.
            actor_loss = -actor_Q1.mean() + (0.01 * action_l2_penalty)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            policy_loss_val = actor_loss.item()

            self._soft_update(self.actor_target, self.actor, self.td3_settings.tau)
            self._soft_update(self.critic_1_target, self.critic_1, self.td3_settings.tau)
            self._soft_update(self.critic_2_target, self.critic_2, self.td3_settings.tau)

        # ---------------------------------------------------------
        # KROK 4: Statystyki
        # ---------------------------------------------------------
        update_stats = {
            "Losses/Critic 1 Loss": critic_1_loss.item(),
            "Losses/Critic 2 Loss": critic_2_loss.item(),
            "Losses/Actor Loss": policy_loss_val,
        }

        return update_stats
    
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Aktualizuje mechanizmy normalizacji w sieciach Krytyków.
        """
        # Jeśli w TD3CriticNetwork są czyste warstwy nn.Linear
        # i nie ma wpiętych ML-Agents ObservationNormalizer, po prostu pass.
        # Gdy zrobi się normalizację (powinna być), tutaj przekażesz bufor do statystyk Krytyka.
        pass

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
            obs_tensor = torch.cat([ModelUtils.list_to_tensor(o) for o in obs_raw], dim=1)
            
            # Pobieramy wykonane akcje z tej ścieżki
            actions = AgentAction.from_buffer(batch).continuous_tensor
            
            # Pytamy pierwszego Krytyka: "Ile przewidujesz punktów za to, co on właśnie zrobił?"
            q_values = self.critic_1(obs_tensor, actions)

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