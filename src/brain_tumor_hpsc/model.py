from __future__ import annotations


def build_resnet50_classifier(
    image_size: tuple[int, int],
    num_classes: int,
    learning_rate: float = 0.001,
    freeze_backbone: bool = True,
):
    """Build and compile a ResNet50 transfer learning classifier."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    backbone.trainable = not freeze_backbone

    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
