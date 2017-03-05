import unittest, h5py
import os.path as P
import generate as gen
class TestGenerate(unittest.TestCase):
	def test_strtensor(self):
		for task in gen.PERM + gen.ARITH:
			strdata, trans = gen.generate_set(3, 5, 2, 4, task)
			self.assertEqual(
				gen.tensor2str(
					gen.str2tensor(strdata, trans), trans, csv=',' in strdata),
				strdata,
				'str/tensor conversion failed at task {}'.format(task))
	def test_reconstr(self):
		testdir = '.test/'
		datadir = gen.generate_all('repeatCopy', testdir, 7,
			15, 3, 5,
			11, 6, 9,
			23,
            train_nrepeat_low=2, train_nrepeat_high=10,
            valid_nrepeat_low=11, valid_nrepeat_high=20)
		traintensor, validtensor = gen.reconstruct_tensor(datadir)
		f = h5py.File(P.join(datadir, 'tensor.hdf5'), 'r')
		truetrain, truevalid = f['train'], f['valid']
		self.assertTrue((traintensor==truetrain).all())
		self.assertTrue((validtensor==truevalid).all())

if __name__ == '__main__':
	unittest.main()