from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        if 'stage_two_model.h5' not in os.listdir('../SavedModels'):
            return CheckResult.wrong("The file `stage_two_model.h5`\n"
                                     "is not in SavedModels directory")

        model = load_model('../SavedModels/stage_two_model.h5')

        if 'stage_two_history' not in os.listdir('../SavedHistory'):
            return CheckResult.wrong("The file `stage_two_history`\n"
                                     "is not in SavedHistory directory")

        with open('../SavedHistory/stage_two_history', 'rb') as stage_two:
            history = pickle.load(stage_two)

        if (not isinstance(history, dict)
                or not isinstance(history['accuracy'], list)
                or not isinstance(history['val_accuracy'], list)):
            return CheckResult.wrong("`stage_two_history` should be a dictionary of lists")

        # Make plot

        train_accuracy = history['accuracy']
        val_accuracy = history['val_accuracy']

        train_loss = history['loss']
        val_loss = history['val_loss']

        epochs = len(train_accuracy)
        epochs_range = range(1, epochs + 1)

        if (train_accuracy[-1] - val_accuracy[-1]) > 0.10:
            return CheckResult.wrong("The model is overfitting the train set\n"
                                     "The difference between final train and val accuracies > 10%\n"
                                     f"You've trained the model with {epochs} epochs, use 5 instead")

        mosaic = """
        AB
        AB
        """
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        axs = fig.subplot_mosaic(mosaic)

        for label, axes in axs.items():
            if label == 'A':
                axes.plot(epochs_range, train_accuracy, label='train_accuracy')
                axes.plot(epochs_range, val_accuracy, label='test_accuracy')
                axes.set_title("Accuracy Plot")
                axes.set_xlabel('Number of epoch')
                axes.set_ylabel('Accuracy Score')
            if label == 'B':
                axes.plot(epochs_range, train_loss)
                axes.plot(epochs_range, val_loss)
                axes.set_title("Loss Plot")
                axes.set_xlabel('Number of epoch')
                axes.set_ylabel('Loss Score')
            axes.legend()
        plt.show()

        return CheckResult.correct()


if __name__ == '__main__':
    Tests().run_tests()
