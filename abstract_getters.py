class AbstractGetters:
    """
    This abstract class defines getter methods that need to be implemented for every multimodal dataset separately.
    """
    def get_encs_decs(self, flags, liks):
        """
        Getter for lists with encoders and decoders for all modalities.

        Args:
            flags: argparse.Namespace with input arguments.
            liks: List with likelihoods for every modality.

        Returns:
            Lists with newly initialized encoders and decoders for all modalities.
        """
        raise NotImplementedError

    def get_img_to_digit_clfs(self, flags):
        """
        Getter for the list with pre-trained image-to-digit classifiers.

        Args:
            flags: argparse.Namespace with input arguments.

        Returns:
            A list with pre-trained image-to-digit classifiers for all modalities.
        """
        raise NotImplementedError

    def get_data_loaders(self, batch_size, num_modalities, num_workers, shuffle=True, device="cuda",
                         random_noise=False):
        """
        Getter for train and test set DataLoaders.

        Args:
            batch_size: Batch size to use when loading data.
            num_modalities: Number of modalities.
            num_workers: How many subprocesses to use for data loading.
            shuffle: Flag identifying whether to shuffle the data.
            device: Which device to use for storing tensors, "cuda" (by default) or "cpu".
            random_noise: Flag identifying whether to augment images with Gaussian white noise.

        Returns:
            DataLoader for training and test sets.
        """
        raise NotImplementedError
