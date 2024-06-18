import torch
from torchvision import transforms
from code_v2.dataset import ImageDataset, Prefetcher

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = torch.device("cuda", 0)

class MockDataloader:
    def __init__(self, data=[]):
        self.idx = -1
        self.data = data
        self.iter = None

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        return self.data[self.idx] if self.idx >= len(self.data) else None
    
    def __len__(self):
        return len(self.data)
    
class PrefetcherStub(Prefetcher):
    def preload(self):
        return None

def test_image_dataset_init() -> None:
    """
    Test ImageDataset contructor.
    """
    mock_set = ImageDataset('train', mean, std)
    assert mock_set.mode == 'train'
    assert isinstance(mock_set.post_transform, transforms.Compose)

def test_image_dataset_len() -> None:
    """
    Test ImageDataset length method.
    """
    mock_paths = ['a.png','b.png','c.png']
    mock_set = ImageDataset('train', mean, std)
    assert len(mock_set) == 0
    mock_set.image_file_paths = mock_paths
    assert len(mock_set) == len(mock_paths)

def test_prefetcher() -> None:
    """
    Test Prefetcher constructor.
    """
    mock_prefetcher = PrefetcherStub(MockDataloader(), device)
    assert len(mock_prefetcher) == 0
    assert mock_prefetcher.device == device

def test_prefetcher_next(mocker) -> None:
    """
    Test Prefetcher next method.
    """
    mock_prefetcher = PrefetcherStub(MockDataloader(), device)
    preload_spy = mocker.spy(mock_prefetcher, "preload")
    mock_prefetcher.next()
    preload_spy.assert_called_once()

def test_prefetcher_reset(mocker) -> None:
    """
    Test Prefetcher reset method.
    """
    mock_prefetcher = PrefetcherStub(MockDataloader(), device)
    preload_spy = mocker.spy(mock_prefetcher, "preload")
    mock_prefetcher.reset()
    preload_spy.assert_called_once()

def test_prefetcher_len() -> None:
    """
    Test Prefetcher length method.
    """
    mock_data = ['a','b','c']
    mock_prefetcher = PrefetcherStub(MockDataloader(mock_data), device)
    assert len(mock_prefetcher) == len(mock_data)
