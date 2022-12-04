import torch
import torch.nn as nn

class Correlator(nn.Module):
    """ The basic siamese network joining network, that takes the outputs of
    two embedding branches and joins them applying a correlation operation.
    Should always be used with tensors of the form [B x C x H x W], i.e.
    you must always include the batch dimension.
    """

    def __init__(self, dsift=False, device=None):
        super(Correlator, self).__init__()
        self.epsilon = 1e-6
        self.dsift = dsift
        self.device = device

    def normalize_batch_zero_mean(self, batch):
        b,c,w,h = batch.shape
        mean = batch.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        batch1 = batch + torch.randn(batch.shape).to(self.device)*self.epsilon
        std = batch1.view(b, c, -1).std(dim=2).view(b, c, 1, 1)

        return (batch - mean) / (std + self.epsilon)

    def normalize_batch_zero_mean_dsift(self, batch):
        b,h,w,f = batch.shape
        mean = batch.view(b, h*w*f).mean(dim=1).view(b, 1, 1, 1)
        std = batch.view(b, h*w*f).std(dim=1).view(b, 1, 1, 1)

        return (batch - mean) / (std + self.epsilon)

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The reference patch of dimensions [B, C, H, W].
                Usually the shape is [8, 3, 127, 127].
            x2 (torch.Tensor): The search region image of dimensions
                [B, C, H', W']. Usually the shape is [8, 3, 255, 255].
        Returns:
            match_map (torch.Tensor): The score map for the pair. For the usual
                input shapes, the output shape is [8, 1, 33, 33].
        """

        match_map = self.match_corr2(x1, x2)
        return match_map

    def match_corr2(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].

        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.

        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape

        if self.dsift:
            embed_ref = self.normalize_batch_zero_mean_dsift(embed_ref).view(b, 1, c*h*w)
            embed_srch = self.normalize_batch_zero_mean_dsift(embed_srch).view(b, c*h*w, 1)
        else:
            embed_ref = self.normalize_batch_zero_mean(embed_ref).view(b, 1, c*h*w)
            embed_srch = self.normalize_batch_zero_mean(embed_srch).view(b, c*h*w, 1)

        match_map = torch.matmul(embed_ref, embed_srch)
        match_map /= (h*w*c)

        return match_map