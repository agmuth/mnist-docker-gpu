from pydantic import BaseModel


class ModelBuildConfig(BaseModel):
    batch_size: int = 64  # 'input batch size for training (default: 64)'
    test_batch_size: int = 1000  # 'input batch size for testing (default: 1000)'
    epochs: int = 14  # 'number of epochs to train (default: 14)'
    lr: float = 1.0  # 'learning rate (default: 1.0)'
    gamma: float = 0.7  # 'Learning rate step gamma (default: 0.7)'
    no_cuda: bool = False  # 'disables CUDA training'
    dry_run: bool = False  # 'quickly check a single pass'
    seed: int = 1  # 'random seed (default: 1)'
    log_interval: int = 10  # 'how many batches to wait before logging training status'
    save_model: bool = True  # 'For Saving the current Model'
    file_name: str = "mnist_cnn.pt"
