"""Unit test package for tensors."""
import unittest
import torch


class TensorsTestCase(unittest.TestCase):

    def test_basics(self):
        a = torch.ones(3, 3)
        self.assertEqual(2, a.ndim)

        points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
        # 3 rows, 2 columns
        self.assertEqual(torch.Size([3, 2]), points.shape)
        self.assertEqual(1., points[0, 1])  # first row, second column
        self.assertTrue(torch.equal(torch.tensor([4., 1.]), points[0]))  # first row, all columns

        # Indexing
        self.assertTrue(torch.equal(torch.tensor([[5.0, 3.0], [2.0, 1.0]]), points[1:]))  # first row default
        self.assertTrue(torch.equal(torch.tensor([[5.0, 3.0], [2.0, 1.0]]), points[1:, :]))  # first row same
        self.assertTrue(torch.equal(torch.tensor([5., 2.]), points[1:, 0]))  # 1D of rows 1 and 2
        self.assertEqual(torch.Size([1, 3, 2]), points[None].shape)  # Leading dimension with None

    def test_broadcasting(self):
        # Scalar addition
        # Tensor X: shape (1, 3) row vector
        x = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        # Scalar Y
        y = 10

        # Broadcasted scalar addition
        result = x + y
        expected = torch.tensor([[11, 12, 13]], dtype=torch.int64)

        self.assertEqual(torch.Size([1, 3]), result.shape)
        self.assertTrue(torch.equal(result, expected))

        # Tensor multiplication
        # Tensor A: shape (1, 3) row vector
        a = torch.tensor([[1, 2, 3]], dtype=torch.int64)

        # Tensor B: shape (3, 1) column vector
        b = torch.tensor([[10],
                          [20],
                          [30]], dtype=torch.int64)

        # Broadcasted multiplication: (3, 1) * (1, 3) -> (3, 3)
        c = b * a
        expected = torch.tensor([[10, 20, 30],
                                 [20, 40, 60],
                                 [30, 60, 90]], dtype=torch.int64)

        self.assertEqual(torch.Size([3, 3]), c.shape)
        self.assertTrue(torch.equal(c, expected))

    def test_named_tensors(self):
        img_t = torch.randn(3, 5, 5)  # shape [channels, rows, columns]
        weights = torch.tensor([0.2126, 0.7152, 0.0722])
        print(weights)

        weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
        weights_named
        print(weights_named)
        img_named = img_t.rename(..., 'channels', 'rows', 'columns')
        self.assertEqual(torch.Size([3, 5, 5]), img_named.shape)

    def test_tensor_api(self):
        a = torch.ones(3, 2)
        a_t = torch.transpose(a, 0, 1)
        self.assertEqual(torch.Size([3, 2]), a.shape)
        self.assertEqual(torch.Size([2, 3]), a_t.shape)

    def test_tensor_more(self):
        # .size(), .offset(), .stride()
        # Storage: size, offset stride.
        # .transpose or .t
        # .numpy and .from_numpy also .save() and .load()
        # torch.unsqueeze - add a new dimension
        # img.permutate(2, 0, 1) - swap indexes
        # .view() returns a new tensor that changes the number of dimensions, without changing the storage.
        # .cat() concatenate sequence of tensors along the existing location
        pass


if __name__ == '__main__':
    unittest.main()
