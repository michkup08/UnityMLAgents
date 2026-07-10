# Uczenie ze wzmocnieniem na potrzeby generowania ruchu w animacji 3D (Praca Magisterska)

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
* `Project/Assets/Examples/Vertebrate/Scenes/Enviroments/`

### 2. Implementacja algorytmów w Pythonie
Kod backendu treningowego został rozszerzony o obsługę algorytmów DroQ oraz PPG. Do nowych folderów z plikami algorytmów należą:
* `ml-agents/mlagents/trainers/droQ/`, `ml-agents/mlagents/trainers/ppg/`
Zmiany nastąpiły też w istniejących już pplikach gdzie jak np:
* `ml-agents\mlagents\trainers\torch_entities\networks.py`
Kod eksportujący wyniki do plików csv i generujący wykresy:
* `config/plotsGenerator.py`, `ml-agents/mlagents/trainers/ppg/`

### 3. Pliki konfiguracyjne (YAML)
Hiperparametry dla przeprowadzonych eksperymentów dla modeli PPO, SAC, PPG i DroQ znajdują się w katalogu:
* `config/`.

## 📂 Wyniki

Wyniki wykonanych pomiarów są w osobnym repozytorium: https://github.com/michkup08/UnityMLAgentsResults
