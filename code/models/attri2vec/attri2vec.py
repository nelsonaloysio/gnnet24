import logging as log
import random
from typing import Optional

import keras
import numpy as np
import stellargraph as sg
import tensorflow as tf
from stellargraph.data import UnsupervisedSampler
from stellargraph.layer import Attri2Vec, link_classification
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from tensorflow import keras


def attri2vec(
    SG: sg.StellarGraph,
    epochs: int = 1,
    number_of_walks: int = 80,
    length: int = 10,
    layer_sizes: list = [128],
    batch_size: Optional[int] = None,
    workers: int = 1,
    verbose: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:

    log.info(SG.info())

    nodes = list(SG.nodes())
    if not batch_size:
        batch_size = 20000

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        keras.utils.set_random_seed(812)

    unsupervised_samples = UnsupervisedSampler(
        SG,
        nodes=nodes,
        length=length,
        number_of_walks=number_of_walks
    )

    generator = Attri2VecLinkGenerator(SG, batch_size)

    model = Attri2Vec(
        layer_sizes=layer_sizes,
        generator=generator,
        bias=False,
        normalize=None
    )
    x_inp, x_out = model.in_out_tensors()

    prediction = link_classification(
        output_dim=1,
        output_act="sigmoid",
        edge_embedding_method="ip"
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    history = model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=1,
        shuffle=True,
    )

    x_inp_src = x_inp[0]
    x_out_src = x_out[0]

    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_gen = Attri2VecNodeGenerator(SG, batch_size).flow(nodes)
    node_embeddings = embedding_model.predict(node_gen, workers=workers, verbose=verbose)
    return node_embeddings
