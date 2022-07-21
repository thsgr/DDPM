class DataloaderGenerator:
    """
    Base abstract class for data loader generators
    dataloaders
    """
    def __init__(self):
        pass

    def dataloaders(self, batch_size, num_workers=0, shuffle_train=True,
                    shuffle_val=False):
        """
        Returns: triplet (dataloader_train, dataloader_val, dataloader_test) of dataloaders
        """
        raise NotImplementedError