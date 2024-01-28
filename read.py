import tensorflow as tf
import keras
def pca_tensorflow(data, n_components):
    mean=tf.reduce.mean('./dataset/mel2_format/blues.00000.jpg',axis=0)
    data=data-mean
    s,u,v=tf.linalg.svd(data)
    principal_components=tf.slice(v,[0,0],[tf.shape(v)[0],n_components])
    return tf.matmul(data,principal_components)
train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255).flow_from_directory('./dataset/mel2_format',
                                                                color_mode="grayscale",
                                                                target_size=(1293,128),
                                                                batch_size=1,
                                                                class_mode="categorical",
                                                                )
print(train_generator.next()[0].shape)