import unittest
import numpy as np
from hhtpy import EmpiricalModeDecomposition
from hhtpy.plot import plot_imfs
import matplotlib.pyplot as plt


class TestEMDAndPlotting(unittest.TestCase):

    def setUp(self):
        # Create the sample data used in the original code
        self.T = 5  # sec
        self.f_s = 15000  # Hz
        self.n = np.arange(self.T * self.f_s)
        self.t = self.n / self.f_s  # sec

        self.y = (
                0.3 * np.cos(2 * np.pi * 5 * self.t ** 2) +
                2 * np.cos(2 * np.pi * 1 * self.t) +
                1 * self.t
        )

    def test_emd_decomposition(self):
        emd = EmpiricalModeDecomposition(self.y)
        emd.decompose()

        self.assertIsNotNone(emd.imfs)
        self.assertGreater(len(emd.imfs), 0, "IMFs should not be empty after decomposition")

        self.assertIsNotNone(emd.residue, "Residue should not be None after decomposition")
        self.assertEqual(len(emd.residue), len(self.y), "Residue length should match the input signal length")

    def test_plot_imfs(self):
        emd = EmpiricalModeDecomposition(self.y)
        emd.decompose()

        fig, axs = plot_imfs(emd.imfs, self.y, emd.residue, x_axis=self.t, show_plot=False)

        self.assertIsInstance(fig, plt.Figure, "The output should be a matplotlib Figure object")
        self.assertIsInstance(axs, np.ndarray, "The output should be an array of matplotlib Axes")



if __name__ == '__main__':
    unittest.main()