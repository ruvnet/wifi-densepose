# The following code is partially copied from the official example.
# https://github.com/pytorch/examples/blob/ca1bd9167f7216e087532160fc5b98643d53f87e/mnist/main.py

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


model = torch.jit.load("model.pt")
device = torch.device("cuda")
test_kwargs = {"batch_size": 100}
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
test(model, device, test_loader)
