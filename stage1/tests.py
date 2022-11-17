import sys
import subprocess
from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        result = subprocess.run([sys.executable, 'transfer.py'], capture_output=True, text=True)
        std_out = [txt for txt in result.stdout.split('\n') if txt]

        for idx, value in enumerate(std_out):
            if idx == 0:
                if value != "Found 500 images belonging to 2 classes.":
                    return CheckResult.wrong("`train_data_gen` comes first\n"
                                             "There should be 500 images in 2 classes")
            elif idx == 1:
                if value != "Found 200 images belonging to 2 classes.":
                    return CheckResult.wrong("`valid_data_gen` comes second\n"
                                             "There should be 200 images in 2 classes")
            elif idx == 2:
                if value != "Found 50 images belonging to 1 classes.":
                    return CheckResult.wrong("`test_data_gen` comes third\n"
                                             "There should be 50 images in 1 classes")
            elif idx == 3:
                if len(value.split()) != 4:
                    return CheckResult.wrong("Return height, width, batch_size, and shuffle values in this order")
                height, width, batch_size, shuffle = value.split()

                if height != width and height != "150":
                    return CheckResult.wrong("The image height and width value should be 150")

                if batch_size != "64":
                    return CheckResult.wrong("Incorrect batch size")

                if shuffle != "False":
                    return CheckResult.wrong("Do not shuffle `test_data_gen`")

        return CheckResult.correct()


if __name__ == '__main__':
    Tests().run_tests()
