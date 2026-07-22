import torch
from torch_geometric.datasets import KarateClub

from .gcn import GCN
from ...utils.plot import plot_embedding, plot_loss


def test_gcn(epochs=100):
    data = KarateClub()
    device = torch.device("cuda")
    data.data = data.data.to(device)
    data.num_features

    model = GCN(num_features=data.num_features,
                num_classes=data.num_classes,
                hidden_size=16,
                embedding_size=8,
                learning_rate=0.05,
                device="cuda",
                seed=42)

    embeddings, losses = [], []

    for epoch in range(epochs):
        out, h, loss = model.train(data)
        embeddings.append(h)
        losses.append(float(f"{loss:4f}"))

        if epoch % 10 == 0 or epoch+1 == epochs:
            print(f"Epoch: {epoch}\tLoss: {loss:4f}")

        # plot_loss(losses, show=True)
        # plot_embedding(h.detach().cpu().numpy(), data.y.cpu().numpy(), title=f"Epoch {epoch} (loss={loss:4f})", show=True)

    # return out.detach().cpu().numpy(), h.detach().cpu().numpy()

if __name__ == "__main__":
    test_gcn()
