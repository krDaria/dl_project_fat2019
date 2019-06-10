## DL project

Соревнование: [FAT2019](https://www.kaggle.com/c/freesound-audio-tagging-2019)

Бейзлайн: https://github.com/m12sl/kaggle-freesound-2019-baseline
#
### **LSML**:

Вся инструкция из [readme.md](https://github.com/m12sl/kaggle-freesound-2019-baseline/blob/master/README.md)  выполнена.
1.	Научились запускать тренировки (через tmux)
```
# синхронизовать код
Через tmux
# зайти в нужную папку
cd kaggle-freesound-2019-baseline/
#ввести
python main.py --outpath ./runs/
```
2.	Cмотреть графики в tensorboard 
```
Через tmux
# зайти в папку с запусками и запустить TB.  
cd kaggle-freesound-2019-baseline/runs/
CUDA_VISIBLE_DEVICES= tensorboard --logdir=./
# На своей машине запустить
ssh -L 9009:localhost:6006 ubuntu@[public_server_ip]
# зайти в браузере на http://localhost:9009 
```
3.	Отлаживать скрипты. 
```
Через tmux
./start_notebook.sh
# зайти в браузере на https://[public_server_ip]:9999/tree/kaggle-freesound-2019-baseline
```
4.	Отправлять лучшие чекпоинты как датасет на Kaggle через [Kaggle.api](https://github.com/Kaggle/kaggle-api) (_пример для runs/0_)
```
#На сервере
kaggle datasets init -p ~/kaggle-freesound-2019-baseline/runs/0
cd ~/kaggle-freesound-2019-baseline/runs/0
vim dataset-metadata.json (ввести название датасета)
kaggle datasets create -p ./

#Для обновления существующего датасета
kaggle datasets version -p ./ -m "Updated data"
```
5.	Сабмитить ответы через скрипты в кернеле. И это все несмотря на 2 питон 🤔
```
* создать приватный кернел с кодом из kernel_infer.py
* залить лучший чекпоинт best.pth в качестве датасета
* проверить пути, запустить, дождаться просчета и на вкладке Outputs засабмитить ответы
```
Для избежания накладок со 2 питоном, записывался новый chkp
```
#На сервере
cd
python 
import torch
path = './kaggle-freesound-2019-baseline/runs/1/last.pth'
x = torch.load(path)
torch.save(x['model_state_dict'],'./kaggle-freesound-2019-baseline/runs/0/last1.pth')
```
#
### **DL**:

1. Для начала необходимо было добавить сохранение лучшей модели `best.pth` для того, чтобы можно было не заботиться о возможном переобучении сетки и спокойно спать ночью 😴, пока сеть занимается полезностями, а именно - учится
2. Запустили бейзлайн, для того, чтобы понимать от чего нам отталкиваться и с чем работать. Кроме этого, все последующие улучшения, можно сравнивать с этой моделью с помощью tensorboard, зачастую недообучая модель. 
![Image alt](https://github.com/{username}/{repository}/raw/{branch}/{path}/image.png)

3. Идеи возможного улучшения качества модели
  - Поменять `learning rate`
  ```
  ```
  - Дольше учиться 
  ```
  ```
  - Больше батч 
  ```
  ```
  - Изменить функцию активации `ReLU -> Sigmoid`
  ```
  ```
  - Поставить `batchnorm` после функции активации 
  ```
  ```
  - Добавить регуляризатор `dropout`
  ```
  ```
  - Попробовать другую архитектуру последнего слоя fc
  ```
  ```
  - Изменить предобработку данных (изменить окно преобразования Фурье, размер наложения окна)
  ```
  ```
  - Добавить аугментацию `mixup`
  ```
  ```
  - Изменить  `loss`
  ```
  ```
  - _Не успела попробовать - изменение `sampler` в `DataLoader`_

