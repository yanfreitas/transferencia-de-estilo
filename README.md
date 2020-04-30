# transferencia de estilo
O código de trânsferencia de estilo(style trannsfer) tem como objetivo criar uma imagem a partir de 2 outras imagens.
O algoritmo recebe como input uma imagem principal(aquela em que desejamos manter seus componentes principais)
e uma imagem de estilo(aqui conseguimos extrair apenas a essência da imagem, seu estilo).
Como output, recebemos uma imagem totalmente nova com componentes da primeira imagem e estilo da segunda.
Como exemplo podemos pegar uma imagem pessoal como imagem principal e uma imagem de uma pintura famosa que conhecemos, assim
iremos adquirir em pouco tempo o que poderia demorar horas para ser feito por um pintor profissional.
Esse algoritmo usa conceitos como trânsferencia de aprendizado, gradiente descendente, redes neurais e aprendizado de máquina.
O algoritmo foi feito durante o curso: Deep Learning: Advanced Computer Vision (GANs, SSD, +More!) na Udemy.

Libs usadas:
matplotlib
numpy
scipy
tensorflow/keras

Ambiente:
Pycharm

No repositório temos:
main_code: Onde está todo o código utilizado.
ny.jpg: Imagem principal(uma imagem de Nova York)
pintura.jpg: Imagem de estilo
final_img.jpg: Imagem de resultado(output)
