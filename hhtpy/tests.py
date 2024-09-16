import unittest
import numpy as np
from hhtpy.emd import decompose
from hhtpy.plot import plot_imfs
import matplotlib.pyplot as plt


class TestEMDAndPlotting(unittest.TestCase):

    def setUp(self):
        # Create the sample data used in the original code
        T = 5  # sec
        f_s = 15000  # Hz
        n = np.arange(T * f_s)
        t = n / f_s  # sec

        self.y = (
            0.3 * np.cos(2 * np.pi * 5 * t**2) + 2 * np.cos(2 * np.pi * 1 * t) + 1 * t
        )

    def test_emd_decomposition(self):

        imfs, residue = decompose(self.y)

        self.assertIsNotNone(imfs)
        self.assertGreater(len(imfs), 0, "IMFs should not be empty after decomposition")

        self.assertIsNotNone(residue, "Residue should not be None after decomposition")
        self.assertEqual(
            len(residue),
            len(self.y),
            "Residue length should match the input signal length",
        )

    def test_plot_imfs(self):
        imfs, residue = decompose(self.y)

        fig, axs = plot_imfs(imfs, self.y, residue, x_axis=self.t, show_plot=False)

        self.assertIsInstance(
            fig, plt.Figure, "The output should be a matplotlib Figure object"
        )
        self.assertIsInstance(
            axs, np.ndarray, "The output should be an array of matplotlib Axes"
        )


if __name__ == "__main__":
    unittest.main()
