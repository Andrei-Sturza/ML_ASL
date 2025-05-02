import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time


# --------------------------
# PARTEA 1: ANTRENARE MODEL
# --------------------------
def train_model():
    # Parametri
    DATA_DIR = "asl_alphabet_train"
    IMG_SIZE = 200
    BATCH_SIZE = 32
    EPOCHS = 5

    # Verificare existenta dataset
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Directorul cu dataset '{DATA_DIR}' nu există!")

    # Preprocesare date
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset='training',
        class_mode='categorical'
    )

    val_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        class_mode='categorical'
    )

    # Arhitectură CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(29, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Antrenare
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # Salvare model
    model.save('asl_cnn.h5')
    print("Model antrenat și salvat cu succes!")


# --------------------------
# PARTEA 2: RECUNOAȘTERE TIMP REAL
# --------------------------
def real_time_recognition():
    # Configurații
    MODEL_PATH = 'asl_cnn.h5'
    CLASSES_PATH = "asl_alphabet_train"
    ROI_SIZE = 300
    TARGET_SIZE = (200, 200)
    CONFIDENCE_THRESHOLD = 0.8
    FPS_UPDATE_INTERVAL = 10

    # Verificare existență model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelul '{MODEL_PATH}' nu există!")

    try:
        # Încărcare model cu optimizare pentru inferență
        model = models.load_model(MODEL_PATH)
        model.trainable = False  # Dezactivează training mode pentru performanță
    except Exception as e:
        raise RuntimeError(f"Eroare la încărcarea modelului: {str(e)}")

    # Validare structură model
    if len(model.input_shape) != 4 or model.input_shape[1:3] != TARGET_SIZE:
        raise ValueError("Modelul are o structură neașteptată!")

    # Încărcare și validare clase
    class_names = sorted(os.listdir(CLASSES_PATH))
    if len(class_names) != model.output_shape[-1]:
        raise ValueError("Numărul de clase nu corespunde cu modelul!")

    # Inițializare camera cu validare
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nu s-a putut deschide camera!")

    # Variabile pentru performanță
    frame_count = 0
    start_time = time.time()
    fps = 0

    roi_cache = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        x_center, y_center = w // 2, h // 2
        x_min = max(0, x_center - ROI_SIZE // 2)
        x_max = min(w, x_center + ROI_SIZE // 2)
        y_min = max(0, y_center - ROI_SIZE // 2)
        y_max = min(h, y_center + ROI_SIZE // 2)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            continue

        # Procesare imagine
        try:
            # Convertire RGB și redimensionare
            processed_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            processed_img = cv2.resize(processed_img, TARGET_SIZE)

            # Normalizare și batch dimension
            processed_img = np.expand_dims(processed_img / 255.0, axis=0)

            # Predictie cu cache
            if not np.array_equal(roi_cache, processed_img):
                pred = model.predict(processed_img, verbose=0)[0]
                roi_cache = processed_img.copy()

            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
        except Exception as e:
            print(f"Eroare procesare: {str(e)}")
            continue

        # Calcul FPS
        frame_count += 1
        if frame_count % FPS_UPDATE_INTERVAL == 0:
            fps = FPS_UPDATE_INTERVAL / (time.time() - start_time)
            start_time = time.time()

        label = f"{class_names[class_idx]} ({confidence:.2f})" if confidence > CONFIDENCE_THRESHOLD else "Se incarca..."
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Q - Iesire", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # train_model()
    real_time_recognition()