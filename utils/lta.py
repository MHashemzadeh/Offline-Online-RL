import torch

import torch.nn as nn


def construct_interval_starts(lower_bound=0.0, upper_bound=1.0, num_intervals=10):
    """
    :return: Tensor of shape (1, num_intervals), each element indicating the start
    of intervals evenly spaced between the upper and lower bound.
    """
    delta = (upper_bound - lower_bound) / num_intervals
    return torch.linspace(lower_bound, upper_bound - delta, num_intervals)


def exact_indicator_scalar(x):
    if x > 0:
        return 1
    else:
        return 0

def exact_indicator(x, threshold=0.0):
    """
    :param x: Tensor of any shape.
    :param threshold:  The threshold for indication.
    :return: A tensor of the same shape, with values 1 where x was above the threshold,
    and zero where x was below the threshold.
    """
    indicated = torch.zeros_like(x)
    indicated[x > threshold] = 1

    return indicated


def approximate_indicator(x, threshold=0.0):
    """
    Identical to exact_indicator, but with values the same as x when below the threshold.
    """
    indicated = x.clone()
    indicated[x > threshold] = 1

    return indicated


def tile_activation_scalar(x, bin_interval_starts, threshold):
    """
    :param x: A scalar value.
    :param bin_interval_starts: A tensor shaped (1, k) with evenly spaced values. See: construct_interval_starts
    :param threshold: A sparsity-controlling threshold, which also proportional the fraction of the
    function's domain with nonzero derivatives.
    :return: A tensor with the same shape as bin_interval_starts, with a different tile activation
    applied to each unit.
    NOTE: An alternative, batch-vectorized call is also supported here.  If x has been repeated in an interleaved way,
    and bin_interval starts has been repeated in a tiled way, such that the dimension of both is (-1, k * d), then this
    will return the tile activation of the whole batch.  See bins_to_activations() for an example.
    """
    zeros = torch.zeros_like(bin_interval_starts)
    bin_width = bin_interval_starts.flatten()[1] - bin_interval_starts.flatten()[0]

    shifted_decreasing = bin_interval_starts - x
    shifted_increasing = (x - bin_width) - bin_interval_starts

    clamped_decreasing = torch.max(shifted_decreasing, zeros)
    clamped_increasing = torch.max(shifted_increasing, zeros)

    return 1 - approximate_indicator(clamped_decreasing + clamped_increasing, threshold)


def bins_to_activations(x, bin_interval_starts, sparsity_control):
    """
    The vectorized, batch-friendly version of tile_activation_scalar.
    :param x: A batched input tensor of shape (-1, d)
    :param bin_interval_starts: A tensor shaped (1, k) with evenly spaced values. See: construct_interval_starts
    :param sparsity_control: A sparsity-controlling threshold, which also proportional the fraction of the
    function's domain with nonzero derivatives.
    :return: A batched vector of shape (-1, d * k), where every k elements is a tile activation from an input element.
    """
    batch_size = x.shape[0]
    input_dimension = x.shape[1]
    num_bins = bin_interval_starts.shape[0]

    repeated_bin_interval_starts = bin_interval_starts.repeat(batch_size, input_dimension)

    # NOTE: `x.repeat_interleave(num_bins, dim=1)` is the more readable, desireable call.  But right now it's so
    # slow that it dominates execution time.  See https://github.com/pytorch/pytorch/issues/31980
    repeated_inputs = x[..., None].expand(-1, -1, num_bins).flatten(1)

    return tile_activation_scalar(repeated_inputs, repeated_bin_interval_starts, sparsity_control)


def tile_activation(x, lower_bound=0, upper_bound=1, num_intervals=5, sparsity_control=0.05):
    """
    :param x: A batched input tensor of shape (-1, d)
    :param lower_bound: Lower bound on domain.
    :param upper_bound: Upper bound on domain.
    :param num_intervals: Number of intervals/bins to split the input domain into.
    :param sparsity_control: A sparsity-controlling threshold, which also proportional the fraction of the
    function's domain with nonzero derivatives.
    :return: :return: A batched vector of shape (-1, d * num_intervals), where every num_intervals elements is a tile
    activation from an input element.
    """
    bin_interval_starts = construct_interval_starts(lower_bound, upper_bound, num_intervals).to(x.device)

    return bins_to_activations(x, bin_interval_starts, sparsity_control)


class LTALayer(nn.Module):
    def __init__(self, input_dim, output_dim, pre_tiling_width, bins, eta, tile_min, tile_max):
        """
        A network approximating ℝ^n --> ℝ^m functions using a linear transformation, then LTA, followed
        by a linear transformation.

        EX:
        model = LTALayer(
            input_dim = 5,
            output_dim = 10,
            pre_tiling_width = 15,
            bins = 20,
        )

        ^This will induce the following chain of transformations:

        5 dim ->  5x15 matrix -> 15 dim -> 15x20 LTA -> 300 dim -> 300x10 matrix -> 10 dim
        -----                    ------                 -------                     -----

        where activation layers are underlined, and 10x20 LTA means each of the 10 dimensions
        input to the LTA is transformed to 20 dimensions by the "leaky tiled" binning process
        described in the paper.

        :param input_dim:
            The input dimensionality of the layer.
        :param output_dim:
            The output dimensionality of the layer.
        :param pre_tiling_width:
            The linear transformation output dimensionality, as input to LTA.
        :param bins: (k in the paper)
            Number of bins per input unit of LTA.
        :param eta: (η in the paper)
            Controls sparsity, proportion of each bin's domain with nonzero derivative,
            and overlap of each bin.
            Each bin has a sloped tail off either end of the indicator cap, and the
            domain under each tail has length (tile_max - tile_min) * eta
            eta=0 means the activation is disjoint binning with globally vanishing gradients.
            If eta<0 is passed, then it is automatically computed as 1/bins, which is
            a reasanable starting point, and usually sufficient.
        :param tile_min: (l in the paper)
            The lowest value an input can take before vanishing across all bins.
        :param tile_max: (u in the paper)
            The highest value an input can take before vanishing across all bins.
        """
        super().__init__()

        self.bins = bins
        if eta < 0:
            self.eta = 1.0 / self.bins
        else:
            self.eta = eta

        self.tile_min = tile_min
        self.tile_max = tile_max

        self.in_linear = nn.Linear(input_dim, pre_tiling_width)
        self.out_linear = nn.Linear(pre_tiling_width * bins, output_dim)

    def forward(self, x):
        x = self.in_linear(x)
        x = tile_activation(x, self.tile_min, self.tile_max, self.bins, self.eta)
        x = self.out_linear(x)

        return x


class ReluLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_width):
        """
        A network approximating ℝ^n --> ℝ^m functions using a linear transformation, then Relu, followed
        by a linear transformation.  This corresponds to LTALayer for fair comparison across architectures.

        EX:
        model = ReluLayer(
            input_dim = 5,
            output_dim = 10,
            hidden_width = 15,
        )

        ^This will induce the following chain of transformations:

        5 dim ->  5x15 matrix -> 15 dim -> Relu -> 15 dim -> 15x10 matrix -> 10 dim
        -----                    ------            ------                    ------

        where activation layers are underlined.

        :param input_dim:
            The input dimensionality of the layer.
        :param output_dim:
            The output dimensionality of the layer.
        :param hidden_width:
            The linear transformation output dimensionality, as input to Relu.
        """
        super().__init__()

        self.in_linear = nn.Linear(input_dim, hidden_width)
        self.out_linear = nn.Linear(hidden_width, output_dim)

    def forward(self, x):
        x = self.in_linear(x)
        x = torch.relu(x)

        x = self.out_linear(x)

        return x
