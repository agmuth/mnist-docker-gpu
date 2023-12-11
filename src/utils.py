import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from src.cnn import CNN
from src.constants import MODEL_DIR


class DataUtils:
    _dataset = datasets.MNIST
    _tmp_dir = "../data"
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((28, 28), antialias=True),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # train_dataset = _dataset(
    #     _tmp_dir, train=True, download=False, transform=img_transform
    # )
    # test_dataset = _dataset(_tmp_dir, train=False, transform=img_transform)


class Trainer:
    def __init__(self, loader_kwargs):
        self._loader = torch.utils.data.DataLoader(
            DataUtils._dataset(
                DataUtils._tmp_dir, train=True, download=True, transform=DataUtils.img_transform
            ),
            **loader_kwargs
        )

    def train(self, args, model, device, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(self._loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(self._loader.dataset),
                        100.0 * batch_idx / len(self._loader),
                        loss.item(),
                    )
                )
                if args.dry_run:
                    break


class Tester:
    def __init__(self, loader_kwargs):
        self._loader = torch.utils.data.DataLoader(
            DataUtils._dataset(DataUtils._tmp_dir, train=False, transform=DataUtils.img_transform), **loader_kwargs
        )

    def test(self, model, device):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self._loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self._loader.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(self._loader.dataset),
                100.0 * correct / len(self._loader.dataset),
            )
        )


class ModelUtils:
    @staticmethod
    def load_model_from_file(model_name: str):
        model = CNN()
        model.load_state_dict(torch.load(MODEL_DIR.joinpath(model_name)))
        model.eval()
        return model
