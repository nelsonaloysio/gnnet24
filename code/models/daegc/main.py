import logging as log
import random
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
# from sklearn.cluster import KMeans

from .daegc import DAEGC
from ..gat import GAT
from ..kmeans import KMeans
from ..utils.earlystop import EarlyStop
from ..utils.evaluate import evaluate


def getargs(argv: list = sys.argv[1:]) -> Tuple[Namespace, list]:
    """ Get arguments. """
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILEPATH",
                        type=torch.load)

    parser.add_argument("--hidden-size",
                        help="Hidden size of the GAT model.",
                        default=256,
                        type=int)

    parser.add_argument("--out-features",
                        help="Output features of the GAT model.",
                        default=16,
                        type=int)

    parser.add_argument("--lr",
                        help="Learning rate of the optimizer.",
                        type=float,
                        default=0.005)

    parser.add_argument("--weight-decay",
                        help="Weight decay of the optimizer.",
                        type=float,
                        default=0.005)

    parser.add_argument("--update-interval",
                        help="Interval to update centroids.",
                        default=5,
                        type=int)

    parser.add_argument("--epochs",
                        help="Number of epochs to train before early stop.",
                        type=int,
                        default=500)

    parser.add_argument("--patience",
                        help="Number of epochs to wait for improvement before early stop.",
                        type=int,
                        default=50)

    parser.add_argument("--alpha",
                        help="Alpha value for the GAT model.",
                        type=float,
                        default=0.2)

    parser.add_argument("--gamma",
                        help="Gamma value for the DAEGC model.",
                        type=float,
                        default=10.0)

    parser.add_argument("--t",
                        help="T value for the DAEGC model.",
                        type=int,
                        default=2)

    parser.add_argument("--v",
                        help="V value for the DAEGC model.",
                        type=float,
                        default=1.0)

    parser.add_argument("--n-init",
                        help="Number of initializations for KMeans.",
                        type=int,
                        default=20)

    parser.add_argument("--seed",
                        help="Random seed number.",
                        type=int)

    parser.add_argument("--split",
                        help="Split data into train, validation and test sets.",
                        action="store_true")

    parser.add_argument("--stage",
                        help="Model stage to run.",
                        choices=["pretrain+train", "pretrain", "train"],
                        default="pretrain+train")

    parser.add_argument("--weights",
                        default="weights_daegc",
                        help="File path to save or load weights.")

    parser.add_argument("--device",
                        help="Device to run the model.",
                        choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--device-kmeans",
                        help="Device to run the clusterer.",
                        choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args(argv)


def main(args: Namespace) -> None:
    """ Main function. """
    data = args.data

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    gat = GAT(
        in_features=data.num_features,
        hidden_size=args.hidden_size,
        out_features=args.out_features,
        alpha=args.alpha,
    ).to(args.device)

    x = data.x.to(args.device, dtype=torch.float32)
    y = data.y.cpu().numpy()
    n_clusters = len(np.unique(y))

    if args.split:
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    else:
        train_mask = val_mask = test_mask = np.ones(x.shape[0], dtype=bool)

    A = DAEGC.get_A(data.edge_index)
    B = DAEGC.get_B(A)
    M = DAEGC.get_M(B, args.t).to(args.device, dtype=torch.float32)
    B = B.to(args.device, dtype=torch.float32)

    weights_pretrain = f"{args.weights}_pretrain_seed={args.seed}.pkl"
    weights_train = f"{args.weights}_train_seed={args.seed}.pkl"

    # Pretrain model.
    if "pretrain" in args.stage.split("+"):
        log.info("Pretraining model...")

        optimizer = Adam(gat.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        es = EarlyStop(patience=args.patience)

        for epoch in range(args.epochs):
            gat.train()

            # Forward pass.
            A_pred, Z = gat(x[train_mask], B[:, train_mask][train_mask], M[:, train_mask][train_mask])

            # Backward pass.
            loss = F.binary_cross_entropy(A_pred.view(-1).cpu(), A[:, train_mask][train_mask].view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation.
            with torch.no_grad():
                A_pred, Z = gat(x[val_mask], B[:, val_mask][val_mask], M[:, val_mask][val_mask])
                y_pred = KMeans(n_clusters=n_clusters, n_init=args.n_init, seed=args.seed, device=args.device_kmeans).fit_predict(Z)
                eva = evaluate(y[val_mask], y_pred)

                if es(eva, loss=loss.item()):
                    break
                if epoch == es.best_epoch:
                    torch.save(gat.state_dict(), weights_pretrain)

        gat.load_state_dict(torch.load(weights_pretrain, map_location="cpu"))
        log.info(f"Best on pretrain: {es.best}")

        # Test.
        if args.split:
            A_pred, Z = gat(x[test_mask], B[:, test_mask][test_mask], M[:, test_mask][test_mask])
            y_pred = KMeans(n_clusters=n_clusters, n_init=args.n_init, seed=args.seed, device=args.device_kmeans).fit_predict(Z)
            eva = evaluate(y[test_mask], y_pred)
            log.info(f"Test on pretrain: {eva}")

    # Train model.
    if "train" in args.stage.split("+"):
        log.info("Training model...")

        if weights_pretrain is None:
            log.info("Loading weights from pretrain...")
            gat.load_state_dict(torch.load(args.weights, map_location="cpu"))

        daegc = DAEGC(
            n_clusters=n_clusters,
            n_features=args.out_features,
            v=args.v,
        ).to(args.device)

        optimizer = Adam(daegc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        es = EarlyStop(patience=args.patience)

        # Obtain initial cluster centers.
        with torch.no_grad():
            A_pred, Z = gat(x, B, M)

            daegc.cluster_layer.data = KMeans(
                n_clusters=n_clusters,
                n_init=args.n_init,
                seed=args.seed,
                device=args.device_kmeans
            )\
            .fit(Z)\
            .cluster_centers_\
            .to(args.device)

        for epoch in range(args.epochs):
            gat.train()

            # Forward pass.
            A_pred, Z = gat(x[train_mask], B[:, train_mask][train_mask], M[:, train_mask][train_mask])
            Q = daegc(Z)
            if epoch % args.update_interval == 0:
                P = daegc.get_P(Q.detach())

            # Backward pass.
            r_loss = F.binary_cross_entropy(A_pred.view(-1).cpu(), A[:, train_mask][train_mask].view(-1))
            c_loss = F.kl_div(Q.log(), P, reduction="batchmean")
            loss = r_loss + (args.gamma * c_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation.
            with torch.no_grad():
                A_pred, Z = gat(x[val_mask], B[:, val_mask][val_mask], M[:, val_mask][val_mask])
                Q = daegc(Z)
                y_pred = Q.detach().data.cpu().numpy().argmax(1)
                eva = evaluate(y[val_mask], y_pred)

                if es(eva, loss=loss.item(), r_loss=r_loss.item(), c_loss=c_loss.item()):
                    break
                if epoch == es.best_epoch:
                    torch.save(gat.state_dict(), weights_train)

        gat.load_state_dict(torch.load(weights_train, map_location="cpu"))
        log.info(f"Best on train: {es.best}")

        # Test.
        if args.split:
            A_pred, Z = gat(x[test_mask], B[:, test_mask][test_mask], M[:, test_mask][test_mask])
            Q = daegc(Z)
            y_pred = Q.detach().data.cpu().numpy().argmax(1)
            eva = evaluate(y[test_mask], y_pred)
            log.info(f"Test on train: {eva}")


if __name__ == "__main__":
    sys.exit(main(getargs()))
