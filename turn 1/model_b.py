import unittest
import numpy as np

class TestFeatureIntegrity(unittest.TestCase):

    def setUp(self):
        # Dummy data for features
        self.features = np.array([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0],
                                  [7.0, 8.0, 9.0]])

    def test_feature_shape(self):
        """
        Verify that the features array has the expected shape.
        """
        self.assertEqual(self.features.shape, (3, 3),
                         msg="Features array should have shape (n_samples, n_features).")

    def test_feature_data_type(self):
        """
        Verify that the features array contains the expected data type.
        """
        self.assertEqual(self.features.dtype, np.float64,
                         msg="Features array should contain float64 data type.")

    def test_feature_values(self):
        """
        Verify that the features array contains the expected values.
        """
        expected_features = np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(self.features, expected_features,
                                      err_msg="Features array contains incorrect values.")

if __name__ == '__main__':
    unittest.main()
