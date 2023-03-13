import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import numpy
import pickle
from test_labels import test_labels


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=3500000)]

    def check(self, reply: str, attach):

        if 'stage_four_history' not in os.listdir('../SavedHistory'):
            return CheckResult.wrong("The file `stage_four_history` is not in SavedHistory directory")

        with open('../SavedHistory/stage_four_history', 'rb') as stage_four:
            answer = pickle.load(stage_four)

        if not isinstance(answer, numpy.ndarray):
            return CheckResult.wrong("`stage_four_history` should be a numpy array")

        labels = test_labels()
        accuracy = labels == answer
        test_accuracy = accuracy.mean()

        if test_accuracy < 0.92:
            return CheckResult.wrong(f"Your model's accuracy is {test_accuracy * 100}%\n"
                                     "Iterate over your hyperparameter values and try to score at least 92%.")

        print(f"Test accuracy: {round(test_accuracy, 3)}")
        return CheckResult.correct()


if __name__ == '__main__':
    Tests().run_tests()
