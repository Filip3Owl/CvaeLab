import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_SIZE = 64

# Nome da pasta em italiano → rótulo em inglês
LABEL_MAP = {
    "cane":       "dog",
    "cavallo":    "horse",
    "elefante":   "elephant",
    "farfalla":   "butterfly",
    "gallina":    "chicken",
    "gatto":      "cat",
    "mucca":      "cow",
    "pecora":     "sheep",
    "ragno":      "spider",
    "scoiattolo": "squirrel",
}

# Índice inteiro determinístico por classe (ordenado alfabeticamente pelo nome em italiano)
CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(sorted(LABEL_MAP))}
IDX_TO_CLASS: dict[int, str] = {i: LABEL_MAP[cls] for cls, i in CLASS_TO_IDX.items()}

NUM_CLASSES = len(CLASS_TO_IDX)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class AnimalsDataset(Dataset):
    """
    Dataset Animals-10 que retorna pares (tensor_imagem, índice_de_classe).

    Args:
        root_dir  : Caminho para a raiz do dataset (uma subpasta por classe).
        transform : Pipeline de transformações torchvision aplicado a cada imagem.
    """

    def __init__(self, root_dir: str, transform=train_transform):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        for class_name in os.listdir(root_dir):
            if class_name not in CLASS_TO_IDX:
                continue
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            label = CLASS_TO_IDX[class_name]
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_dataloader(
    root_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
    augment: bool = True,
) -> DataLoader:
    transform = train_transform if augment else eval_transform
    dataset = AnimalsDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)
