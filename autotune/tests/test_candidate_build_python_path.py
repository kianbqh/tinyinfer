"""CPU-only regression for the candidate builder's interpreter selection."""
import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class CandidatePythonPathTests(unittest.TestCase):
    def test_configure_preserves_symlinked_venv_interpreter(self):
        source = Path(os.environ.get(
            'TINYC_CANDIDATE_BUILD_SOURCE',
            str(Path(__file__).resolve().parents[2] / 'bench/v2/kernels/candidate_build.py')))
        module = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
        main = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        assignments = [node for node in main.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'configure'
                               for target in node.targets)]
        self.assertEqual(len(assignments), 1, 'one actual configure command expected')
        # Evaluate the actual command expression, without importing ML modules
        # or invoking CMake, a compiler, a model, or any build subprocess.
        expression = compile(ast.Expression(assignments[0].value), str(source), 'eval')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            system = directory / 'system'
            system.mkdir()
            python_target = system / 'python3.12'
            cmake_target = system / 'cmake'
            python_target.write_text('fixture only; never executed', encoding='utf-8')
            cmake_target.write_text('fixture only; never executed', encoding='utf-8')
            venv_bin = directory / 'venv/bin'
            venv_bin.mkdir(parents=True)
            venv_python = venv_bin / 'python'
            venv_python.symlink_to(python_target)
            cmake_link = directory / 'cmake'
            cmake_link.symlink_to(cmake_target)
            self.assertTrue(venv_python.is_symlink())
            self.assertNotEqual(venv_python.absolute(), venv_python.resolve())
            args = SimpleNamespace(
                python=str(venv_python), cmake=str(cmake_link),
                ninja=str(directory / 'ninja'), cxx=str(directory / 'cxx'),
                python_include=str(directory / 'include/python3.12'),
                gtest_source=str(directory / 'googletest'))
            command = eval(expression, {'Path': Path}, {
                'args': args, 'root': directory / 'repo', 'build': directory / 'build',
                'identifier': 0, 'nvcc': directory / 'cuda/bin/nvcc'})
            interpreter_arguments = [arg for arg in command
                                     if arg.startswith('-DPython3_EXECUTABLE=')]
            self.assertEqual(interpreter_arguments,
                             ['-DPython3_EXECUTABLE=' + str(venv_python.absolute())])
            self.assertNotIn('-DPython3_EXECUTABLE=' + str(python_target.resolve()), command)
            self.assertEqual(command[0], str(cmake_target.resolve()),
                             'other tool paths must retain their existing resolution')


if __name__ == '__main__':
    unittest.main(verbosity=2)
