# Uczenie ze wzmocnieniem na potrzeby generowania ruchu w animacji 3D (Praca Magisterska)

https://github.com/user-attachments/assets/86bc7f3c-3798-45fc-901a-993305f5bee0

## ⚠️ Oświadczenie o autorstwie i źródłach
Niniejsze repozytorium zawiera kod wykorzystany na potrzeby pracy magisterskiej. Projekt bazuje na oficjalnym środowisku Unity ML-Agents Toolkit. 

Ten projekt jest modyfikacją oficjalnego środowiska Unity ML-Agents stworzoną na potrzeby pracy magisterskiej. Mój autorski wkład (środowiska testowe, agenci, systemy nagród, niezaimplementowane domyślnie algorytmy) znajduje się w dedykowanych folderach, a reszta repozytorium to rdzenny, niemodyfikowany kod biblioteki ML-Agents.

## 📖 O projekcie
Głównym celem projektu była analiza mechanizmu emergencji zachowań lokomocyjnych w zależności od złożoności środowiska. W ramach pracy przeprowadzono eksperymenty symulacyjne na autorskich środowiskach 3D o różnej charakterystyce topograficznej (m.in. środowisko płaskie, losowy zakrzywiony teren oraz środowisko z obserwacjami wizualnymi).

W ramach badań zrealizowano następujące kroki:
* Zaprojektowanie i stworzenie autorskich środowisk treningowych w silniku Unity.
* Implementacja oraz integracja z silnikiem Unity wybranych, niezaimplementowanych domyślnie algorytmów RL: Phasic Policy Gradient (PPG) i Dropout Q-Function (DroQ).
* Zestawienie zaimplementowanych algorytmów z rozwiązaniami domyślnymi ze środowiska, takimi jak PPO i SAC.
* Przeprowadzenie eksperymentów symulacyjnych oraz analiza porównawcza wyników.

## 📂 Struktura autorskiego kodu

Poniżej wyszczególniono lokalizacje plików i folderów, które stanowią autorski wkład w repozytorium:

### 1. Środowiska i skrypty w silniku Unity (C#)
Autorskie obiekty środowisk, modele postaci, definicje przestrzeni obserwacji i akcji agenta oraz skrypty odpowiedzialne za inżynierię funkcji nagrody znajdują się w katalogu:
* `MLAgentsProject/Assets/Examples/Vertebrate/Scenes/Enviroments/`

### 2. Implementacja algorytmów w Pythonie
Kod backendu treningowego został rozszerzony o obsługę algorytmów DroQ oraz PPG. Do nowych folderów z plikami algorytmów należą:
* `ml-agents/mlagents/trainers/droQ/`, `ml-agents/mlagents/trainers/ppg/`
Zmiany nastąpiły też w istniejących już pplikach gdzie jak np:
* `ml-agents\mlagents\trainers\torch_entities\networks.py`
Kod eksportujący wyniki do plików csv i generujący wykresy:
* `config/plotsGenerator.py`, `config/statsExporterAllInOne.py`

### 3. Pliki konfiguracyjne (YAML)
Hiperparametry dla przeprowadzonych eksperymentów dla modeli PPO, SAC, PPG i DroQ znajdują się w katalogu:
* `config/`.

## 📂 Wyniki

Wyniki wykonanych pomiarów są w osobnym repozytorium: https://github.com/michkup08/UnityMLAgentsResults

## 📂 Sposób korzystania

### 1. Konfiguracja i instalacja
Całą instalację pakietów oraz niezbędną konfigurację środowiska (w tym języka Python i biblioteki PyTorch) należy przeprowadzić ściśle zgodnie z oficjalną dokumentacją ML-Agents dostępną pod poniższym adresem:
https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Installation.html

### 2. Dostępne środowiska i uruchamianie
Pod ścieżką `MLAgentsProject/Assets/Examples/Vertebrate/Scenes/Enviroments/` znajdują się środowiska badawcze do eksperymentów z generowaniem ruchu. Projekt zawiera warianty o zróżnicowanym poziomie trudności: środowisko podstawowe, środowisko z losowo generowanym terenem oraz środowisko z obserwacjami wizualnymi (wymagające od agenta analizy obrazu).

Uruchamianie treningów w tych środowiskach odbywa się w sposób w pełni standardowy, zgodnie z instrukcjami z podlinkowanej wyżej dokumentacji ML-Agents (za pomocą komendy mlagents-learn i wskazania odpowiedniego pliku konfiguracyjnego).

### 3. Konfiguracja algorytmów (YAML)
Poza wbudowanymi w pakiet algorytmami PPO oraz SAC, aplikacja została rozszerzona o implementację metod PPG (Phasic Policy Gradient) oraz DroQ. Aby użyć nowych algorytmów, wystarczy wpisać ppg lub droq w odpowiednim pliku konfiguracyjnym .yaml. Model rozpozna te parametry i rozpocznie standardowy proces uczenia.

Uwaga dotycząca PPG: Algorytm ten charakteryzuje się rozszerzoną strukturą i posiada dodatkowe parametry konfiguracyjne w pliku YAML względem klasycznego PPO. Należą do nich w szczególności:

* num_policy_updates_per_aux – parametr określający częstotliwość uruchamiania fazy pomocniczej (wskazuje, po ilu standardowych aktualizacjach polityki ma zostać przeprowadzona faza optymalizacji współdzielonej reprezentacji).
* kl_penalty_coef – współczynnik wagowy (w literaturze opisywany często jako $\beta_{clone}$), który kontroluje karę za nadmierne odchylenia polityki (klonowanie zachowania) podczas aktualizacji w fazie pomocniczej.
