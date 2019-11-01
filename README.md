# Wildfire competition
Соревнование алгоритмов классификации лесных пожаров по данным о температурных аномалиях со спутников.
 
https://wildfire.sberbank.ai/competition

https://github.com/sberbank-ai/mchs-wildfire

## Baselines
### 1. catboost_simple 
Catboost, только тренировочные данные (Public LB 0,9060).

### 2. catboost_ncep 
Catboost, тренировочные данные и данные о погоде (Public LB 0,9183).

## Usage
#### 1. Скачиваем данные
Тренировочный датасет скачиваем вручную с сайта https://wildfire.sberbank.ai/competition. 
В архиве будет два файла - `wildfires_train.csv` и `wildfires_check.csv`, распаковываем их в папку `data`.

Дополнительно можно использовать наборы открытых данных.
Скрипт `download_ncep.sh` в папке `data` скачает исторические данные о погоде с сайта [NCEP Reanalysis 2].

[NCEP Reanalysis 2]: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis2.html

#### 2. Тренируем модель

`train_in_jupyter.ipynb` - тренировка в блокноте.

`train.py` - то же самое в формате скрипта.

`train_in_docker.sh` - запускает `train.py` в docker-контейнере.

#### 3. Тестируем предикт

`predict.py` - скрипт для предикта.

`test_predict.sh` - проверка работы `predict.py` на контрольных данных в контейнере.

#### 4. Упаковываем submission.

`pack_submission.sh` - упаковывает нужные файлы в `submission.zip` для загрузки на сайт.

## Docker
Решение запускается в контейнере. Организаторы предлагают использовать образ `sberbank/python`. 
В этом образе установлена версия `tensorflow`, собранная с использованием команд современных процессоров Intel.
При локальном тестировании на менее современных процессорах возникает ошибка:
```
The TensorFlow library was compiled to use AVX2 instructions, but these aren't available on your machine.
```
Ошибка возникает даже без использования `tensorflow` - при запуске `python`, `pip`, `conda`,  т.е. удалить 
или переустановить его цивилизованными способами невозможно.

Поэтому скрипты в этом репозитории используют образ `efimov/wildfire`, который собирается из этого [Dockerfile]. 
Он основан на образе `kaggle/python` со следующими изменениями:
 - пакет `tensorflow` заменён на стандартный из `conda`
 - добавлены/обновлены некоторые пакеты.
 
[Dockerfile]: https://github.com/mikhailefimov/wildfire/blob/master/docker/Dockerfile
 