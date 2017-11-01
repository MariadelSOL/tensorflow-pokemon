# TensorPokemon
Web-service which matches pictures of people with pictures of pokemons using Keras.

## How does it work
Project uses the high-level neural networks API [Keras](https://keras.io/) and the neural network ResNet which was developed by Microsoft.

On startup neural network computes vectors for all pokemon images. When user uploads new photo, network computes vector for this photo too and compares it to all vectors of pokemon images. It outputs the image which has closes distance to uploaded photo.

## О проекте
Этот сайт был сделан на хакатоне по машинному обучению.

В его создании принимали участие 

* [Дмитрий Миттов](https://github.com/dmittov)
* [Андрей Евлампиев](https://github.com/andyzt)
* [Никита Путинцев](https://vk.com/id39947099)
* [Людмила Корнилова](https://github.com/kornilova-l)

## Как он работает
Сайт использует библиотеку для машнного обучения Keras. Для распознования образов в него загружена нейронная сеть ResNet от компании Microsoft, которая выйграла Imagenet Recognition Challenge.

При запуске кода нейросеть считает вектора всех изображений покемонов и запоминает их. Когда пользователь загружает фотографию, сеть считает ее вектор, сравнивает со всеми векторами покемонов и выбирает то изображение, вектор которого оказался ближе всего к изображению.

Код с комментариями можено посмотреть здесь: [run-server.py](https://github.com/kornilova-l/tensorflow-pokemon/blob/master/run-server.py)

## Что он распознает
Сама сеть ResNet не была натренирована специально для распознований черт лица. Но она может увидеть на изображении такие образы как цветок, спираль, бабочка и др.
![скриншот с сайта. Девушка с цветком и покемон-цветок](http://news.ifmo.ru/images/news/big/646202.jpg)
![скриншот с сайта. Существо на фоне спирали и покемон со спиралью](http://oi66.tinypic.com/4ihf7d.jpg)
