# Manga Translator V10.4

Локальный Windows-инструмент для перевода английских manga/manhwa PDF на русский язык. Основной сценарий рассчитан на CPU: распознавание пузырей, OCR, машинный перевод и сборка PDF выполняются локально.

> \*\*Статус:\*\* рабочий экспериментальный прототип. Перед массовой обработкой обязательно проверьте несколько страниц. Ошибки OCR не исправляются автоматически переводчиком, а художественные шрифты и сложная верстка требуют ручной проверки.

## Быстрый старт

```powershell
# 1. Клонировать репозиторий
git clone https://github.com/determinati0n/Manga-Translator.git
cd manga-translator

# 2. Подготовить Windows CPU-окружение
Set-ExecutionPolicy -Scope Process Bypass
.\\setup\_windows.ps1

# 3. Активировать venv
\& D:\\AI\\venv\\Scripts\\Activate.ps1

# 4. Положить исходный PDF
#    D:\\manhva\\input.pdf

# 5. Сначала проверить одну страницу
python .\\manga\_translator\_v10\_4.py `
  --input D:\\manhva\\input.pdf `
  --output D:\\manhva\\test.pdf `
  --pages 4
```

Если одна страница выглядит нормально, запускайте диапазон или весь PDF.

## Как работает pipeline

```text
PDF
 │
 ▼
Рендер страницы
 │
 ▼
YOLO11n instance segmentation
поиск speech bubbles
 │
 ▼
PaddleOCR English
несколько OCR-вариантов
 │
 ▼
NLLB-200 distilled 600M
eng\_Latn → rus\_Cyrl
 │
 ▼
Удаление исходного текста
только внутри маски пузыря
 │
 ▼
Рендер русского текста
 │
 ▼
Новый PDF
```

Qwen2.5-3B-Instruct-GGUF остаётся **опциональным редактором**, но отключён по умолчанию: `USE\_QWEN\_EDITOR = False`.

## Возможности

* English → Russian;
* YOLO11n instance segmentation для speech bubbles;
* PaddleOCR с fallback-вариантами изображения;
* NLLB-200 600M как основной переводчик;
* удаление исходного текста только внутри найденной маски пузыря;
* обработка одной страницы, диапазона или выбранных страниц;
* отсутствие обязательных облачных API;
* необязательный локальный Qwen-редактор.

## Требования

* Windows 10/11, 64-bit;
* Python 3.12, 64-bit;
* 16 GB RAM желательно;
* SSD и несколько GB свободного места;
* CPU с несколькими потоками.

GPU для текущей CPU-конфигурации не требуется.

PyTorch официально поддерживает Windows с Python 3.9–3.12; PaddlePaddle для Windows указывает поддержку Python 3.9–3.13 и 64-bit x86\_64. cite-not-in-readme

## Установка

### 1\. Python

Установите Python 3.12 (64-bit) и проверьте:

```powershell
py -3.12 --version
```

### 2\. Git

Установите Git for Windows, затем клонируйте репозиторий:

```powershell
git clone https://github.com/determinati0n/Manga-Translator.git
cd manga-translator
```

### 3\. Основная установка

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\\setup\_windows.ps1
```

Скрипт:

1. создаёт `D:\\AI\\venv`(я устанавливал все на диск D, так как не хватало памяти на основном диске, sry);
2. устанавливает CPU-сборку PyTorch из официального индекса PyTorch;
3. устанавливает PaddlePaddle CPU;
4. ставит остальные зависимости;
5. переносит основные кэши на `D:`;
6. сохраняет переменные кэша в пользовательское окружение Windows;
7. запускает базовую проверку импортов и `pip check`.

По умолчанию создаются:

```text
D:\\AI
├── venv
├── huggingface
├── torch
├── ultralytics
├── pip-cache
└── models

D:\\manhva
```

Если нужен другой диск, поменяйте `MANGA\_AI\_DIR` и `MANGA\_TRANSLATOR\_HOME` в `setup\_windows.ps1` до запуска.

### 4\. Активация

```powershell
\& D:\\AI\\venv\\Scripts\\Activate.ps1
python --version
python -m pip check
```

## Модели

Модели скачиваются автоматически при первом использовании.

### Speech bubbles

`huyvux3005/manga109-segmentation-bubble` — YOLO11n instance segmentation, один класс `Speech Bubble`.

### Перевод

`facebook/nllb-200-distilled-600M`

```text
eng\_Latn → rus\_Cyrl
```

### Qwen — необязательно

`Qwen/Qwen2.5-3B-Instruct-GGUF`

Файл:

```text
qwen2.5-3b-instruct-q4\_k\_m.gguf
```

Для обычной установки Qwen не нужен.

Если хотите протестировать редактор:

```powershell
.\\setup\_qwen\_windows.ps1
```

Этот скрипт использует готовый CPU wheel `llama-cpp-python`, чтобы не требовать локальную сборку C/C++.

После установки отдельно включите:

```python
USE\_QWEN\_EDITOR = True
```

Но сначала рекомендуется сравнить результат с режимом NLLB-only.

## Где лежит PDF

По умолчанию:

```text
D:\\manhva\\input.pdf
```

Файл можно назвать иначе — путь задаётся через `--input`.

## Запуск

### Одна страница

```powershell
python .\\manga\_translator\_v10\_4.py `
  --input D:\\manhva\\input.pdf `
  --output D:\\manhva\\test.pdf `
  --pages 4
```

### Диапазон

```powershell
python .\\manga\_translator\_v10\_4.py `
  --input D:\\manhva\\input.pdf `
  --output D:\\manhva\\test\_4\_10.pdf `
  --pages 4-10
```

### Выбранные страницы

```powershell
python .\\manga\_translator\_v10\_4.py `
  --input D:\\manhva\\input.pdf `
  --output D:\\manhva\\selected.pdf `
  --pages 4,6,10
```

### Весь PDF

```powershell
python .\\manga\_translator\_v10\_4.py `
  --input D:\\manhva\\input.pdf `
  --output D:\\manhva\\translated.pdf `
  --pages all
```

## Настройки

Основные параметры находятся в начале `manga\_translator\_v10\_4.py`.

### Пути

```python
AI\_ROOT = Path(os.environ.get("MANGA\_AI\_DIR", r"D:\\AI"))
WORK\_ROOT = Path(os.environ.get("MANGA\_TRANSLATOR\_HOME", r"D:\\manhva"))
```

Лучше менять пути через `setup\_windows.ps1` или `config.example.ps1`, а не редактировать код.

### OCR

```python
MIN\_OCR\_CONF = 0.40
MIN\_TEXT\_LEN = 2
OCR\_UPSCALE = 2.0
OCR\_DEBUG = True
```

Для отладки OCR:

```python
OCR\_DEBUG = True
```

Фрагменты сохраняются в:

```text
D:\\manhva\\ocr\_debug
```

После отладки можно выключить:

```python
OCR\_DEBUG = False
```

### Speech bubbles

```python
BUBBLE\_IMGSZ = 1600
BUBBLE\_CONF = 0.25
BUBBLE\_IOU = 0.50
```

### Перевод

```python
TRANSLATION\_MODEL = "facebook/nllb-200-distilled-600M"
SRC\_LANG = "eng\_Latn"
TGT\_LANG = "rus\_Cyrl"
```

### Qwen

```python
USE\_QWEN\_EDITOR = False
```

Оставляйте `False`, если нужен наиболее предсказуемый базовый режим V10.4.

## Первый запуск и диагностика

Рекомендуемый порядок:

1. одна страница;
2. 5–10 страниц;
3. целая глава;
4. массовая обработка нескольких глав.

Проверяйте одновременно:

* нашёл ли YOLO все пузыри;
* правильно ли OCR прочитал английский текст;
* соответствует ли русский смыслу оригинала;
* не вышел ли русский текст за пределы пузыря;
* не повреждён ли фон вокруг текста.

Лог:

```text
D:\\manhva\\translation\_v10.log
```

## Частые проблемы

### `torch` не найден

Снова выполните:

```powershell
.\\setup\_windows.ps1
```

И проверьте:

```powershell
python -c "import torch; print(torch.\_\_version\_\_)"
```

### PaddlePaddle не проходит проверку

```powershell
python -c "import paddle; print(paddle.\_\_version\_\_); paddle.utils.run\_check()"
```

Если установка сломалась, используйте актуальную официальную инструкцию PaddlePaddle для Windows.

### OCR прочитал ерунду

Например:

```text
OCR: ILTAK...
```

NLLB не может надёжно восстановить исходную английскую фразу, если OCR уже потерял текст. Смотрите `D:\\manhva\\ocr\_debug` и улучшайте OCR для конкретной страницы/шрифта.

### Перевод слишком буквальный

NLLB — универсальная модель машинного перевода. Короткие разговорные реплики, сленг, междометия и обрывки фраз могут потребовать ручной правки.

### `max\_side\_limit` от PaddleOCR

V10.4 ограничивает размер OCR-подготовки, но отдельные сложные страницы всё равно могут выдавать предупреждения. Сохраните лог и OCR-debug конкретной страницы.

### Qwen устанавливается с ошибкой сборки

Не запускайте просто:

```powershell
pip install llama-cpp-python
```

На Windows это может начать локальную сборку. Используйте:

```powershell
.\\setup\_qwen\_windows.ps1


## Лицензия

Код этого репозитория распространяется под **MIT License**. См. `LICENSE`.

MIT-лицензия относится к коду проекта и **не заменяет** лицензии сторонних библиотек и моделей. Для каждой модели проверяйте её собственную карточку и условия использования.

## Права на исходные материалы

Инструмент предназначен для обработки материалов, которые пользователь имеет право обрабатывать. Не публикуйте в репозитории сканы или производные материалы защищённых авторским правом произведений без соответствующих прав.

## Структура проекта

```text
manga-translator/
├── manga\_translator\_v10\_4.py
├── README.md
├── requirements.txt
├── requirements-qwen.txt
├── setup\_windows.ps1
├── setup\_qwen\_windows.ps1
├── config.example.ps1
├── run\_example.ps1
├── LICENSE
├── NOTICE.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── VERSION
└── .gitignore
```

## Сторонние проекты

* PaddleOCR / PaddlePaddle
* Ultralytics
* Hugging Face Transformers / Hub
* NLLB-200
* MangaLens speech-bubble segmentation model
* Qwen2.5-3B-Instruct-GGUF
* llama-cpp-python

Подробные ссылки и лицензии сторонних компонентов см. в `NOTICE.md`.

## Полезные официальные ссылки

* PyTorch installation: https://docs.pytorch.org/get-started/locally/
* PaddlePaddle Windows installation: https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip\_en.html
* Ultralytics installation: https://docs.ultralytics.com/quickstart
* PaddleOCR: https://www.paddleocr.ai/
* NLLB-200: https://huggingface.co/facebook/nllb-200-distilled-600M
* Speech-bubble model: https://huggingface.co/huyvux3005/manga109-segmentation-bubble
* Qwen2.5-3B-Instruct-GGUF: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
* llama-cpp-python: https://github.com/abetlen/llama-cpp-python
* GitHub Docs — publishing local code: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
* GitHub Docs — licensing: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository

