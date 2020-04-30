from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from skimage.transform import resize
from scipy.optimize import fmin_l_bfgs_b
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

tf.compat.v1.disable_eager_execution()


def vgg16_avgpool(shape):
    """
    Essa função visa criar um modelo sequencial baseado no VGG16,
    com a alteração das camadas de MaxPooling do modelo original(VGG16),
    por camadas de AveragePooling com o objetivo de perder o mínimo de informação possível,
    coisa que acontece com uma de forma aguda nas camadas de MaxPooling.
    """
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)

    new_model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)

    return new_model


def unpreprocess(img):
    """
    Matplotlib espera receber imagens com o formato R.G.B para exibir a mesma,
    enquanto o modelo VGG16 espera receber as imagens em um formato diferente.
    Essa função faz o oposto da função do keras chamada preprocess_input(função que adapta a imagem aos formatos aceitaveis pelo modelo VGG16)
    tornando a imagem novamente em formato R.G.B para poder ser exibida sem problemas.
    """
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]

    return img


def scale_img(x):
    x = x - x.min()
    x = x / x.max()

    return x


def gram_matrix(img):
    """
    A imagem de input tem o formato (height, width, c) onde c = feature maps.
    Primeiro precisamos converter o formato para (c, height * width) = X
    depois calculamos gram matrix = XX ^ T = G.
    """
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
    G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
    return G


def style_loss(y, t):
    """
    Aqui calculamos gram matrix da imagem alvo, e a imagem gerada pelo modelo
    e logo após calculamos o mean square error entre ambos, essa função lida com a imagem de estilo.
    """
    return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


def minimize(fn, epochs, batch_shape):
    """
    Essa função tem como objetivo otimizar a imagem de input.
    Primeiro inicializamos um vetor 1D para ser o valor inicial no minimizer.
    a função fmin retorna um novo valor para x e a perda(loss).
    Quando o loop é completo nós exibimos as perdas(losses), damos um novo formato ao vetor x
    e usa a função unpreprocess para exibir a imagem.
    """
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = fmin_l_bfgs_b(func=fn, x0=x, maxfun=20)
        x = np.clip(x, -127, 127)
        print(f'iter{i}, loss={l}')
        losses.append(l)

    print('duração:', datetime.now() - t0)
    plt.plot(losses)
    plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)

    return final_img[0]


def load_img_and_preprocess(img, shape=None):
    """
    Essa função carrega a imagem e faz o preprocessamento da masma para o formato
    exigido pelo modelo VGG16.
    """
    img = image.load_img(img, target_size=shape)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


# Aqui carregamos a imagem principal
content_img = load_img_and_preprocess('ny.jpg')

# Aqui carregamos e alteramos o tamanho da imagem de estilo
h, w = content_img.shape[1:3]
style_img = load_img_and_preprocess('pintura.jpg', (h, w))

batch_shape = content_img.shape
shape = content_img.shape[1:]


# Aqui criamos o modelo vgg16 com camadas de maxpooling alteradas.
vgg = vgg16_avgpool(shape)

# Aqui criamos o modelo que irá lidar com a imagem principal e selecionamos
# a camada que queremos usar, quando mais profunda a camada, mais borrada a imagem,
# quando mais raza mais nítida a imagem.
# também definimos o alvo(target).
content_model = Model(vgg.input, vgg.layers[10].get_output_at(0))
content_target = K.variable(content_model.predict(content_img))

# Aqui criamos o modelo para a imagem de estilo
# nesse modelo, diferentemente do modelo para a imagem principal onde temos
# apenas 1 output, teremos varios outputs.
symbolic_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]

# Criando o modelo que terá multíplos outputs.
style_model = Model(vgg.input, symbolic_conv_outputs)

# Calculando os alvos que são outputs de cada camada
style_layers_output = [K.variable(y) for y in style_model.predict(style_img)]

# Criando total loss.
loss = K.mean(K.square(content_model.output - content_target))

# Pesos para a imagem de estilo(opcionais)
style_weights = [2, 2, 3, 3, 4]

for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_output):
    loss += w * style_loss(symbolic[0], actual[0])

grads = K.gradients(loss, vgg.input)

# Função para obter gradients e loss.
# Retorna um formato de imagem
get_loss_and_grads = K.function(inputs=[vgg.input], outputs=[loss] + grads)


# Retorna e aceita vetores
def get_loss_and_grads_wrapper(x_vec):
    l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)


final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
plt.imshow(scale_img(final_img))
plt.savefig('final_img.jpg')
plt.show()
