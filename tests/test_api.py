import unittest

class TestAlias(unittest.TestCase):
    def test_alias(self):
        try:
            from tatpulsar.pulse import barycor
        except Exception as e:
            self.fail(f"Failed to call the function via alias, error: {e}")
        self.assertTrue(callable(barycor), "Failed: barycor is not a function")

        # Test readpar Class
        from tatpulsar.utils import readpar
        self.assertTrue(callable(readpar), "Failed: readpar is not a function")

        # Test Profile class
        from tatpulsar.data import Profile
        self.assertTrue(callable(Profile), "Failed: readpar is not a function")

if __name__ == '__main__':
    unittest.main()
