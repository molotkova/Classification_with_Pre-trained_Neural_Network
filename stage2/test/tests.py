import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle

class BaseModelTest(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        if 'stage_two_model.h5' not in os.listdir('../SavedModels'):
            return CheckResult.wrong("The file `stage_two_model.h5` is not in SavedModels directory")

        if 'stage_two_history' not in os.listdir('../SavedHistory'):
            return CheckResult.wrong("The file `stage_two_history` is not in SavedHistory directory")

        with open('../SavedHistory/stage_two_history', 'rb') as stage_two:
            history = pickle.load(stage_two)

        if (not isinstance(history, dict)
                or not isinstance(history['accuracy'], list)
                or not isinstance(history['val_accuracy'], list)
                or not isinstance(history['loss'], list)
                or not isinstance(history['val_loss'], list)):
            return CheckResult.wrong("`stage_two_history` should be a dictionary of four lists;\n"
                                     "Its keys should: \"accuracy\", \"val_accuracy\", \"loss\",and \" val_loss\".")

        # Make plot
        train_accuracy = history['accuracy']
        val_accuracy = history['val_accuracy']

        train_loss = history['loss']
        val_loss = history['val_loss']

        epochs = len(train_accuracy)
        epochs_range = range(1, epochs + 1)

        if ((train_accuracy[-1] - val_accuracy[-1]) > 0.10) and epochs != 5:
            return CheckResult.wrong("The model is overfitting the train set;\n"
                                     "The difference between train and val accuracies after the last epoch is more than 10%;\n"
                                     f"You've trained the model with {epochs} epochs, use 5 instead.")

        if (train_accuracy[-1] - val_accuracy[-1]) > 0.10:
            return CheckResult.wrong("The model is overfitting the train set;\n"
                                     "The difference between train and val accuracies after the last epoch is more than 10%;"
                                     "Make sure to follow the objectives to implement a correct solution.")

        mosaic = """
        AB
        AB
        """
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        axs = fig.subplot_mosaic(mosaic)

        for label, axes in axs.items():
            if label == 'A':
                axes.plot(epochs_range, train_accuracy, label='train_accuracy')
                axes.plot(epochs_range, val_accuracy, label='val_accuracy')
                axes.set_title("Accuracy Plot")
                axes.set_xlabel('Number of epoch')
                axes.set_ylabel('Accuracy Score')
                axes.legend()
            if label == 'B':
                axes.plot(epochs_range, train_loss, label='train_accuracy')
                axes.plot(epochs_range, val_loss, label='val_accuracy')
                axes.set_title("Loss Plot")
                axes.set_xlabel('Number of epoch')
                axes.set_ylabel('Loss Score')
                axes.legend()
        plt.show()

        return CheckResult.correct()


if __name__ == '__main__':
    BaseModelTest().run_tests()
