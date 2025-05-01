import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class DeepfakeDetector:
    def __init__(self, img_size=(128, 128), model_path=None):
        self.img_size = img_size
        self.model = self.build_model()
        
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            print(f"Loaded pre-trained model from {model_path}")
    
    #Building CNN
    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data_generators(self, train_dir, validation_dir, batch_size=32):
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        #validation data generator 
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            classes=['real', 'fake']
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            classes=['real', 'fake']
        )
        
        return train_generator, validation_generator
    
    def train(self, train_generator, validation_generator, epochs=20, checkpoint_path=None):
        callbacks = []
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='max'
            )
            callbacks.append(checkpoint)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, test_generator):
        return self.model.evaluate(test_generator)
    
    def predict(self, image_path):
        from tensorflow.keras.preprocessing import image
        
        img = image.load_img(image_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = self.model.predict(img_array)[0][0]
        return prediction
    
    def predict_batch(self, image_folder):
        results = {}
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(image_folder, filename)
                fake_prob = self.predict(file_path)
                results[filename] = fake_prob
        
        return results
    
    def visualize_training_history(self, history):
        # Plot training & validation accuracy
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_confusion_matrix(self, test_generator):
        test_generator.reset()
        y_pred = self.model.predict(test_generator)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        y_true = test_generator.classes
        
        cm = confusion_matrix(y_true, y_pred_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], 
                    yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        print(classification_report(y_true, y_pred_classes, target_names=['Real', 'Fake']))


if __name__ == "__main__":
    # Paths to dataset
    DATASET_PATH = "path/to/your/dataset"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VALIDATION_DIR = os.path.join(DATASET_PATH, "validation")
    TEST_DIR = os.path.join(DATASET_PATH, "test")
    
    #Calling Class DeepfakeDetector
    detector = DeepfakeDetector(img_size=(128, 128))
    
    train_generator, validation_generator = detector.prepare_data_generators(
        TRAIN_DIR, VALIDATION_DIR, batch_size=32
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        classes=['real', 'fake'],
        shuffle=False
    )

    history = detector.train(
        train_generator,
        validation_generator,
        epochs=20,
        checkpoint_path="models/deepfake_detector_best.h5"
    )
    
    test_loss, test_acc = detector.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")
    
    detector.visualize_training_history(history)
    
    detector.visualize_confusion_matrix(test_generator)

    sample_image = "path/to/test/image.jpg"
    fake_probability = detector.predict(sample_image)
    print(f"Probability of being fake: {fake_probability:.4f}")
    print(f"Image is {'fake' if fake_probability > 0.5 else 'real'}")