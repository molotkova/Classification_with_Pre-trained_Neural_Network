import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import pickle
from test_labels import test_labels


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        if 'stage_three_history' not in os.listdir('../SavedHistory'):
            return CheckResult.wrong("The file `stage_three_history\n"
                                     "is not in SavedHistory directory")

        if 'stage_two_history' not in os.listdir('../SavedHistory'):
            return CheckResult.wrong("The file `stage_two_history\n"
                                     "is not in SavedHistory directory")

        with open('../SavedHistory/stage_two_history', 'rb') as stage_two:
            history = pickle.load(stage_two)

        with open('../SavedHistory/stage_three_history', 'rb') as stage_three:
            answer = pickle.load(stage_three)

        if not isinstance(answer, numpy.ndarray):
            return CheckResult.wrong("`stage_three_history` should be a numpy array")

        labels = test_labels()
        accuracy = labels == answer

        valid_accuracy = history['val_accuracy'][-1]
        train_accuracy = history['accuracy'][-1]
        test_accuracy = accuracy.sum() / 50

        if valid_accuracy - test_accuracy > 0.10:
            return CheckResult.wrong("The difference between validation and test\n"
                                     f"accuracies is {valid_accuracy - test_accuracy}\n"
                                     "The difference should not be more than 10%.")

        if train_accuracy - test_accuracy > 0.10:
            return CheckResult.wrong("The model is overfitting the train set\n"
                                     "The difference between train and test\n"
                                     f"accuracies is {train_accuracy - test_accuracy}\n"
                                     "The difference should not be more than 10%")

        print(f"Test accuracy: {round(test_accuracy, 3)}")

        return CheckResult.correct()


if __name__ == '__main__':
    Tests().run_tests()