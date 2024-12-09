from typing import Optional

import keras
import numpy as np
import stellargraph as sg
from tensorflow import keras


def attri2vec(
    SG: sg.StellarGraph,
    lr: float = 1e-3,
    epochs: int = 1,
    number_of_walks: int = 80,
    length: int = 10,
    layer_sizes: list = [128],
    batch_size: Optional[int] = None,
    workers: int = 1,
    bias: bool = False,
    normalize: bool = None,
    use_multiprocessing: bool = False,
    shuffle: bool = True,
) -> np.ndarray:

    unsupervised_samples = sg.data.UnsupervisedSampler(
        SG,
        nodes=list(SG.nodes()),
        length=length,
        number_of_walks=number_of_walks,
    )

    node_gen = sg.mapper.Attri2VecNodeGenerator(SG, batch_size)
    link_gen = sg.mapper.Attri2VecLinkGenerator(SG, batch_size)

    layer = sg.layer.Attri2Vec(
        layer_sizes=layer_sizes,
        generator=link_gen,
        bias=bias,
        normalize=normalize,
    )
    x_in, x_out = layer.in_out_tensors()

    x_pred = sg.layer.link_classification(
        output_dim=1,
        output_act="sigmoid",
        edge_embedding_method="ip",
    )(x_out)

    model = keras.Model(inputs=x_in, outputs=x_pred)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )
    model.fit(
        link_gen.flow(unsupervised_samples),
        epochs=epochs,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        verbose=2,
    )

    embedding_model = keras.Model(
        inputs=x_in[0],
        outputs=x_out[0],
    )

    node_embeddings = embedding_model.predict(
        node_gen.flow(list(SG.nodes())),
        workers=workers,
        verbose=1,
    )

    return node_embeddings
