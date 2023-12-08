from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.src.layers import Dropout


class ImageClassifier:
    def __init__(self):
        self.model = Sequential()
        self._add_layers()
        self._compile_model()

    def _add_layers(self):
        # Primeira camada convolutiva e pooling
        self.model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Segunda camada convolutiva e pooling
        self.model.add(Conv2D(64, (3, 3), activation='relu'))  # Aumentando o número de filtros
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Terceira camada convolutiva e pooling
        self.model.add(Conv2D(128, (3, 3), activation='relu'))  # Mais filtros
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Achatamento das características para a camada densa
        self.model.add(Flatten())

        # Primeira camada densa com Dropout
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dropout(0.5))  # Adicionando Dropout

        # Camada de saída
        self.model.add(Dense(units=1, activation='sigmoid'))

    def _compile_model(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, training_path, validation_path, batch_size=32, steps_per_epoch=7, epochs=10, validation_steps=3):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory(
            training_path,
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='binary'
        )
        validation_set = validation_datagen.flow_from_directory(
            validation_path,
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='binary'
        )

        self.model.fit(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_set,
            validation_steps=validation_steps
        )


if __name__ == '__main__':
    classifier = ImageClassifier()
    classifier.train('dataset_personagens/training_set', 'dataset_personagens/test_set')
