"""Module providing job management for regression tests."""

import json
import os
import warnings
import shutil
import time
from nbconvert import PythonExporter
import nbformat
from . import checks
from . import test_slot


class TestConfiguration:
    """Data structure that holds the necessary info to define a test and its checks"""

    def __init__(self, file_dir: str,
                 filename: str,
                 outfileprefix: str,
                 num_procs: int,
                 checks_params: list,
                 message_prefix: str,
                 dependency: str,
                 args: list,
                 weight_class: str,
                 skip: str):
        """Constructor. Load checks into the data structure"""
        self.file_dir = file_dir
        self.filename = filename
        self.outfileprefix = outfileprefix
        self.num_procs = num_procs
        self.weight_class = weight_class  # default "short"
        self.checks = []
        self.ran = False
        self.submitted = False
        self.annotations = []
        self.dependency = dependency
        self.args = args
        self.skip = skip

        check_num = 0
        for check_params in checks_params:
            check_num += 1
            if not isinstance(check_params, dict):
                warnings.warn(message_prefix + f'Check number {check_num} ' + 'is not a dictionary')
                continue

            if "type" not in check_params:
                warnings.warn(
                    message_prefix + f'Check number {check_num} ' + 'requires "type" field')
                continue

            if check_params["type"] == "KeyValuePair":
                try:
                    prefix = message_prefix + f'Check number {check_num} '
                    new_check = checks.KeyValuePairCheck(check_params, prefix)
                    self.checks.append(new_check)
                except ValueError:
                    continue
            elif check_params["type"] == "StrCompare":
                try:
                    prefix = message_prefix + f'Check number {check_num} '
                    new_check = checks.StrCompareCheck(check_params, prefix)
                    self.checks.append(new_check)
                except ValueError:
                    continue
            elif check_params["type"] == "FloatCompare":
                try:
                    prefix = message_prefix + f'Check number {check_num} '
                    new_check = checks.FloatCompareCheck(check_params, prefix)
                    self.checks.append(new_check)
                except ValueError:
                    continue
            elif check_params["type"] == "IntCompareCheck":
                try:
                    prefix = message_prefix + f'Check number {check_num} '
                    new_check = checks.IntCompareCheck(check_params, prefix)
                    self.checks.append(new_check)
                except ValueError:
                    continue
            elif check_params["type"] == "ErrorCode":
                try:
                    prefix = message_prefix + f'Check number {check_num} '
                    new_check = checks.ErrorCodeCheck(check_params, prefix)
                    self.checks.append(new_check)
                except ValueError:
                    continue
            elif check_params["type"] == "GoldFile":
                try:
                    prefix = message_prefix + f'Check number {check_num} '
                    new_check = checks.GoldFileCheck(check_params, prefix)
                    self.checks.append(new_check)
                except ValueError:
                    continue
            else:
                warnings.warn("Unsupported check type: " + check_params["type"])
                raise ValueError

        if len(self.checks) == 0:
            warnings.warn(message_prefix + " has no valid checks")
            raise ValueError

    def GetTestPath(self):
        """Shorthand utility to get the relative path to a test"""
        return os.path.relpath(self.file_dir + self.filename)

    def GetOutFilenamePrefix(self) -> str:
        """Get the output filename prefix"""
        if self.outfileprefix == "":
            return self.filename
        return self.outfileprefix

    def __str__(self):
        """Converts the class to a readable format"""
        output = f'file_dir="{self.file_dir}" '
        output += f'filename="{self.filename}" '
        output += f'num_procs={self.num_procs} '

        check_num = 0
        for check in self.checks:
            check_num += 1
            output += f'\n    check_num={check_num} '
            output += check.__str__()

        return output

    def CheckDependencies(self, tests):
        """Loops through a test configuration and checks whether a dependency has executed"""
        if self.dependency is None:
            return True
        for test in tests:
            if test.filename == self.dependency:
                if test.ran:
                    return True

        return False


# Parse JSON configs
def ParseTestConfiguration(file_path: str):
    """Parses a JSON configuration at the path specified"""
    test_objects = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    err_read = "Error reading " + file_path + ": "

    if not isinstance(data, list):
        warnings.warn(err_read + "Main block is not a list")
        return []

    test_num = 0
    for test_block in data:
        test_num += 1
        message_prefix = err_read + f'Test {test_num} '

        if not isinstance(test_block, dict):
            warnings.warn(message_prefix + 'is not a dictionary')
            continue
        if "file" not in test_block:
            warnings.warn(message_prefix + 'does not have key "file"')
            continue
        if "num_procs" not in test_block:
            warnings.warn(message_prefix + 'does not have key "num_procs"')
            continue
        if "checks" not in test_block:
            warnings.warn(message_prefix + 'does not have key "checks"')
            continue

        if not isinstance(test_block["file"], str):
            warnings.warn(message_prefix + '"file" field must be a string')
            continue

        if not isinstance(test_block["checks"], list):
            warnings.warn(message_prefix + '"checks" field must be a list')
            continue

        args = []
        if "args" in test_block and not isinstance(test_block["args"], list):
            warnings.warn(message_prefix + '"args" field must be a list')
            continue
        if "args" in test_block:
            args = test_block["args"]

        dependency = None
        if "dependency" in test_block:
            dependency = test_block["dependency"]

        weight_class = "short"
        if "weight_class" in test_block:
            if isinstance(test_block["weight_class"], str):
                input_weight_class = test_block["weight_class"]
                allowable_list = ["short", "intermediate", "long"]
                if input_weight_class not in allowable_list:
                    warnings.warn(message_prefix + '"weight_class" field, with '
                                  + f'value "{input_weight_class}" must be in the '
                                  + 'list: ' + str(allowable_list))
                    continue
                weight_class = input_weight_class
            else:
                warnings.warn(message_prefix + '"weight_class" field must be a string')
                continue

        outfileprefix = ""
        if "outfileprefix" in test_block:
            if isinstance(test_block["outfileprefix"], str):
                outfileprefix = test_block["outfileprefix"]
            else:
                warnings.warn(message_prefix + '"outfileprefix" field must be a string')
                continue

        skip_reason = ""
        if "skip" in test_block:
            if isinstance(test_block["skip"], str):
                input_reason = test_block["skip"]
                if len(input_reason) == 0:
                    warnings.warn(message_prefix + '"skip" field must be a non-zero length string')
                    continue
                skip_reason = test_block["skip"]
            else:
                warnings.warn(message_prefix + '"skip" field must be a string')
                continue

        try:
            new_test = TestConfiguration(file_dir=os.path.dirname(file_path) + "/",
                                         filename=test_block["file"],
                                         outfileprefix=outfileprefix,
                                         num_procs=test_block["num_procs"],
                                         checks_params=test_block["checks"],
                                         message_prefix=message_prefix,
                                         dependency=dependency,
                                         args=args,
                                         weight_class=weight_class,
                                         skip=skip_reason)
            args_str = ''.join(map(str, new_test.args))
            test_objects[hash(new_test.filename + args_str)] = new_test
        except ValueError:
            continue

    return test_objects


def ListFilesInDir(folder: str, ext=None):
    """Lists the files in a directory, non-recursively. A file extension can be used as a filter"""
    files = []
    dirs_and_files = os.listdir(folder)
    for item in dirs_and_files:
        if not os.path.isdir(item):
            if ext is None:
                files.append(item)
            else:
                base_name, extension = os.path.splitext(item)
                if extension == ext:
                    files.append(item)
    return files


def ConvertNbToScript(notebook_path: str) -> str:
    """
    Converts a Jupyter notebook (.ipynb) to a Python script (.py) in the same directory.

    Parameters
    ----------
    notebook_path : str
        Path to the input .ipynb file.

    Returns
    -------
    str
        Path to the generated .py file.
    """
    # Return old file name if it is not a notebook
    if not notebook_path.endswith(".ipynb"):
        return notebook_path
    # Read notebook
    nb_node = nbformat.read(notebook_path, as_version=4)
    # Create exporter
    exporter = PythonExporter()
    source_code, _ = exporter.from_notebook_node(nb_node)
    # Determine output .py path
    base, _ = os.path.splitext(notebook_path)
    output_py_path = base + ".py"
    # Save the Python script
    with open(output_py_path, "w") as f:
        f.write(source_code)
    return output_py_path


def BuildSearchHierarchyForTests(argv):
    """Finds input files recursively and creates a map of directories to tests"""
    test_dir = argv.directory

    if not os.path.isdir(test_dir):
        raise Exception('"' + test_dir + '" directory does not exist')

    test_hierarchy = {}  # Map of directories to input files
    for dir_path, sub_dirs, files in os.walk(test_dir):
        for file_name in files:
            if argv.engine == "jupyter":
                file_name = ConvertNbToScript(os.path.join(dir_path, file_name))
            base_name, extension = os.path.splitext(file_name)
            if extension == ".py":
                abs_dir_path = os.path.abspath(dir_path) + "/"
                if abs_dir_path not in test_hierarchy:
                    test_hierarchy[abs_dir_path] = [file_name]
                else:
                    test_hierarchy[abs_dir_path].append(file_name)

    return test_hierarchy


def ConfigureTests(test_hierarchy: dict, argv):
    """Search through a map of dirs-to-input-file and looks for a .json file that will then be used
       to create a test object. Also preps the out and gold directories."""

    specific_test = argv.test
    if specific_test is not None:
        print("specific_test=" + specific_test)

    test_objects = []
    for testdir in test_hierarchy:
        for config_file in ListFilesInDir(testdir, ".json"):
            sub_test_objs = ParseTestConfiguration(testdir + config_file)
            specific_test_dependency = None
            for obj in sub_test_objs.values():
                if specific_test is None or obj.filename == specific_test:
                    test_objects.append(obj)
                    if specific_test is not None:
                        specific_test_dependency = obj.dependency
                else:
                    print("skipping " + obj.filename)

            # If a specific test has dependencies, also add them to the list of executed tests
            if specific_test_dependency is not None:
                if specific_test_dependency in sub_test_objs:
                    obj = sub_test_objs[specific_test_dependency]
                    test_objects.append(obj)
                else:
                    warnings.warn(
                        "Specified dependency '" + specific_test_dependency + "' does not exist.")

        # If the out directory exists then we clear it
        if os.path.isdir(testdir + "out/"):
            shutil.rmtree(testdir + "out/")

        # If the out directory does not exist then we create it
        if not os.path.isdir(testdir + "out/"):
            os.mkdir(testdir + "out/")

        # If the gold directory does not exist then we create it
        if not os.path.isdir(testdir + "gold/"):
            os.mkdir(testdir + "gold/")

    return test_objects


def EchoTests(tests: list):
    """For debugging, echos the string format of each test configuration"""
    test_num = 0
    for test in tests:
        print(f"test {test_num} " + str(test))


def RunTests(tests: list, argv):
    """Actually runs the tests. This routine dynamically checks the system load."""
    start_time = time.perf_counter()

    # os.cpu_count() may not be ideal in this case since it returns the number of logical cpus.
    capacity = os.cpu_count()
    if argv.jobs > 0:
        capacity = argv.jobs
    system_load = 0
    test_slots = []

    specific_test = ""
    if argv.test is not None:
        specific_test = argv.test

    weight_class_map = ["long", "intermediate", "short"]
    weight_classes_allowed = []
    if 0 <= argv.weights <= 7:
        binary_weights = '{0:03b}'.format(argv.weights)
        for k in range(0, 3):
            if binary_weights[k] == '1':
                weight_classes_allowed.append(weight_class_map[k])
    else:
        warnings.warn('Illegal value "' + str(argv.weights) + '" supplied '
                      + 'for argument -w, --weights')

    print("Executing tests with weights in: " + str(weight_classes_allowed))

    while True:
        done = True
        # Check for tests to run
        for test in tests:
            if test.ran or (test.weight_class not in weight_classes_allowed):
                continue
            done = False

            if not test.submitted and test.CheckDependencies(tests):
                if test.num_procs <= (capacity - system_load):
                    system_load += test.num_procs

                    new_slot = test_slot.TestSlot(test, argv)

                    # This will only run if a specific test has been specified
                    if new_slot.test.filename == specific_test:
                        print("Running " + new_slot.test.GetTestPath() + ":")

                    test_slots.append(new_slot)

        # Check test progress
        system_load = 0
        for slot in test_slots:
            if slot.Probe():
                system_load += slot.test.num_procs

        time.sleep(1.0)

        if done:
            break

    num_tests_failed = 0
    for slot in test_slots:
        if not slot.passed:
            num_tests_failed += 1

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("Done executing tests with weights in: " + str(weight_classes_allowed))

    num_skipped_tests = 0
    for test in tests:
        if not test.ran:
            num_skipped_tests += 1

    if num_skipped_tests > 0:
        print()
        print(f"\033[93mNumber of skipped tests : {num_skipped_tests}\033[0m")

        if num_skipped_tests > 0:
            print("\033[93mSkipped tests:")
            for test in tests:
                if not test.ran:
                    print(test.filename + f' class="{test.weight_class}"')
            print("\033[0m", end='')

    print()
    print("Elapsed time            : {:.2f} seconds".format(elapsed_time))
    print(f"Number of tests run     : {len(test_slots)}")
    print(f"Number of failed tests  : {num_tests_failed}")

    if num_tests_failed > 0:
        return 1
    return 0


def PrintCaughtWarnings(warning_manager, name: str):
    """Prints warnings"""
    if len(warning_manager) > 0:
        print(f"{name}:")
    for w in warning_manager:
        print("\033[93m" + str(w.category) + "\033[0m", end='')
        print(" ", w.message)
