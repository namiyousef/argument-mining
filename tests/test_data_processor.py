# -- public imports

import unittest
import sys
from itertools import permutations

# -- private imports
from argminer.data import DataProcessor

# -- functions
class TestLogger:
    def __init__(self):
        print(f'Running test "{sys._getframe(1).f_code.co_name}"')
        print("=" * 50)
        self.counter = 1

    def log(self, description):
        print(f'{self.counter}: {description}')
        self.counter += 1

class TestDataProcessor(unittest.TestCase):

    def test_base_processor(self):
        logger = TestLogger()

        logger.log('Test the preproc, proc, postproc cannot be called in incorrect order')

        # define dummy subclass to test superclass features
        class DummyProcessor(DataProcessor):
            def __init__(self, path=''):
                super().__init__(path)


        processor = DummyProcessor()

        status = ['preprocess', 'process', 'postprocess']
        status_perm = list(permutations(status, r=2))

        for s1, s2 in status_perm:
            processor.status = s1
            try:
                getattr(processor, s2)
                raise Exception(f'Failure: Successfully ran {s2} after {s1}')
            except:
                pass
            processor.status = None

        logger.log('Test saving')

        logger.log('Test loading')


        logger.log('Test default train test split')

        logger.log('Test custom train test split')

    def test_tu_darmstadt_processor(self):
        pass

    def test_persuade_processor(self):
        pass

    def test_arg2020_processor(self):
        pass


if __name__ == '__main__':
    unittest.main()