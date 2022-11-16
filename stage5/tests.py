from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import os
import numpy
import pickle
from test_labels import test_labels


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=3500000)]

    def check(self, reply: str, attach):

        if 'stage_five_history' not in os.listdir('../SavedHistory'):
            return CheckResult.wrong("The file `stage_five_history`\n"
                                     "is not in SavedHistory directory")

        with open('../SavedHistory/stage_five_history', 'rb') as stage_five:
            answer = pickle.load(stage_five)

        if not isinstance(answer, numpy.ndarray):
            return CheckResult.wrong("`stage_five_history` is a numpy array")

        labels = test_labels()
        accuracy = labels == answer
        test_accuracy = accuracy.sum() / 50

        if test_accuracy >= 0.95:
            return CheckResult.wrong(f"Your model's accuracy is {test_accuracy * 100}%\n"
                                     "It should be greater than or equals 95%")

        print(f"Test accuracy: {round(test_accuracy, 3)}")
        return CheckResult.correct()


if __name__ == '__main__':
    Tests().run_tests()
