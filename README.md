# PCB_Flaw_Detection

## Opis programu
Program wykorzystuje struktury uczenia maszynowego do tworzenia modelu AI stwierdzającego rodzaj uszkodzenia płyty PCB. Porozumiewa się przy tym z użytkownikiem za pomocą przyjaznego interfejsu na stronie internetowej.

## Użyte technologie
- Do odpowiedniej komunikacji z użytkownikiem użyliśmy środowiska React.js, które oprócz prostoty i szybkości pracy pozwala na połączenie z serwerem. React jest rozwiązaniem często używanym na hackathonach, ponieważ 
jest samohostowany oraz umożliwia podział projektu na komponenty dające przejrzystość kodu.
- Jako środowisko hostujące serwer użyliśmy tailwinda ze względu na jego prostote i możliwość używania języka python, który jest najbardziej popularny jeśli chodzi o tworzenie modeli AI.
- Aby stworzyć model sztucznej inteligencji należy używać algorytmów uczenia maszynowago. Rozwiązaniem stworzonym do pracy na grafikach jest TensorFlow, z dodatkowym zintegrowanym API - Keras. Dzięki niemu możliwe było szybkie wyszkolenie modelu bez konieczności posiadania wiedzy z zakresu matematyki. Tensorflow posiada również wiele poradników i świetną dokumentacje, co ostatecznie zadecydowało o jego użyciu.

## Jak uruchomić program
Program działa na 2 płaszczyznach - Frontend (React.js) i Backend (Flask).
- W folderze backend należy stworzyć środowisko wirtualne ``` python -m venv venv ```, a następnie w nie wejść ```venv/Scripts/activate``` (Windows) albo ```venv/bin/activate``` (Linux) i pobrać wymagane biblioteki z pliku requirements.txt ```pip install -r requirements.txt```. W celu uruchomienia serwera należy użyć polecenie ```flask --app main.py run``` aby uruchomić główny plik operacyjny.

- W folderze PCB_Flaw_Detection_UI po pobraniu Node.js, należy wykonać komende ```npm install``` oraz zbudować ```npm run build``` i uruchomić projekt ```npm run start```.

## Zbiory danych:
- Użyto zestawu danych: 
https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset/data

- Zbiory znajdują się odpowiednio w folderach testing_data i training_data gdzie przeznaczone są do testowania i trenowania modelu AI.

Wyszczególnione wady konstrukcyjne:
- mouse bites
- defekty na ścieżkach
- braki produkcyjne
- spięcia
- przerwanie obwodu
- błędy miedzi

### Mocne strony:
- wysoka jakość obrazów, którą można skalować
- różnorodność problemów ukazanych na obrazach
- popularny typ jpg
- ogromna ilość plików
- darmowy dostęp do plików po zalogowaniu

### Szanse:
- możliwa łatwa aktualizacja danych
- możliwe łatwe rozszerzenie danych o nowe przypadki

### Słabe strony:
- pliki zajmują sporo miejsca
- pliki nie mają nałożonych efektów ułatwiających postrzeganie defektów przez co trzeba je modyfikować
- niektóre zdjęcia mogą posiadać więcej niż jeden defekt co może wprowadzać model w błąd
- dane nie są posortowane kategoriami przez co trzeba to robić kodem
- przypadki defektów trudne do wykrycia (mouse_bite i open_circuit)

### Zagrożenia:
- możliwe późniejsze nałożenie licencji przez twórcę
- możliwe zawieszenie dostępu do plików przez twórcę

## Utworzenie modelu
Aby utworzyć model należy uruchomić program classification.py ```python classification.py``` znajdujący się w folderze backend. Model o nazwie "model.h5" pojawi się w folderze ```backend/models```.

## Trenowanie modelu
Aby trenować model należy uruchomić program train.py ```python train.py``` znajdujący się w folderze backend. Program przeprowadzi domyślnie 200 szkoleń na zbiorze ```training_data``` zawierającym ponad 1000 zdjęć testowych.

## Sprawdzanie modelu
Aby sprawdzić jakość modelu należy uruchomić program check_prediction.py ```python check_prediction.py``` znajdujący się w folderze backend. Program sprawdzi dokładność modelu dla danych, które są dla neigo zupełnie obce.