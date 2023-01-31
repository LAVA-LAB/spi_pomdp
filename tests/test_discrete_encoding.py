import unittest

from utils import EncoderDecoder


class DiscreteEncodingTestCase(unittest.TestCase):
    def test_binary_variable_encode(self):
        enc_dec = EncoderDecoder([range(2)])
        self.assertEqual(enc_dec.encode(0), 0)
        self.assertEqual(enc_dec.encode(1), 1)

    def test_size(self):
        enc_dec = EncoderDecoder([range(2)])
        self.assertEqual(enc_dec.size, 2)

    def test_binary_variable_decode(self):
        enc_dec = EncoderDecoder([range(2)])
        self.assertEqual(enc_dec.decode(0), [0])
        self.assertEqual(enc_dec.decode(1), [1])

    def test_empty_encode(self):
        enc_dec = EncoderDecoder([])
        self.assertEqual(enc_dec.encode(0), 0)

    def test_empty_size(self):
        enc_dec = EncoderDecoder([])
        self.assertEqual(enc_dec.size, 1)

    def test_empty_decode(self):
        enc_dec = EncoderDecoder([])
        self.assertEqual(enc_dec.decode(0), [])

    def test_two_variables_encode(self):
        enc_dec = EncoderDecoder([range(2), range(3)])
        self.assertEqual(enc_dec.encode(0, 0), 0)
        self.assertEqual(enc_dec.encode(1, 0), 3)
        self.assertEqual(enc_dec.encode(1, 2), 5)

    def test_two_variables_size(self):
        enc_dec = EncoderDecoder([range(2), range(3)])
        self.assertEqual(enc_dec.size, 6)

    def test_two_variables_decode(self):
        enc_dec = EncoderDecoder([range(2), range(3)])
        self.assertEqual(enc_dec.decode(0), [0, 0])
        self.assertEqual(enc_dec.decode(3), [1, 0])
        self.assertEqual(enc_dec.decode(5), [1, 2])


if __name__ == '__main__':
    unittest.main()
