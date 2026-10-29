"""CPU-only build-entry and real setuptools sdist checks with mocked SDK imports.

Run from the repository root with:
    python -B -m unittest discover -s tests -p test_build_packaging.py

No compiler, Torch, CANN, network, or NPU is used. Shell scripts run only in
temporary fixture directories; the checkout's build artifacts are never removed.
"""

from contextlib import contextmanager, redirect_stdout
import io
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def find_bash():
    found = shutil.which('bash')
    if found:
        return found
    candidate = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
    return str(candidate) if candidate.is_file() else None


@unittest.skipUnless(find_bash(), 'Bash is needed for isolated shell-entry tests')
class BuildEntryTests(unittest.TestCase):
    def run_entry(self, name, python_body):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            assert root.is_relative_to(Path(tempfile.gettempdir()).resolve())
            script = root / name
            script.write_text((ROOT / name).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
            (root / 'deep_ep').mkdir()
            shell = 'export PATH="/usr/bin:/bin:$PATH"\npython() {\n' + python_body + '\n}\nsource "$1"'
            result = subprocess.run([find_bash(), '--noprofile', '--norc', '-c', shell,
                                     'build-entry-test', script.as_posix()],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            linked = list((root / 'deep_ep').iterdir())
            return result, [path.name for path in linked]

    def test_wheel_build_failure_is_not_masked_by_final_cd(self):
        result, _ = self.run_entry('build.sh', 'return 23')
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)

    def test_development_failure_does_not_link_partial_extension(self):
        result, linked = self.run_entry('develop.sh',
            'mkdir -p build/lib\nprintf partial > build/lib/_C.partial.so\nreturn 23')
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertEqual(linked, [])

    def test_successful_wheel_entry_returns_success(self):
        result, _ = self.run_entry('build.sh', 'test "$1" = setup.py && test "$2" = bdist_wheel')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_successful_development_entry_links_extension(self):
        result, linked = self.run_entry('develop.sh',
            'test "$1" = setup.py && test "$2" = build || return 29\n'
            'mkdir -p build/lib\nprintf fixture > build/lib/_C.fixture.so')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(linked, ['_C.fixture.so'])


class SourceDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import setuptools
            from setuptools.command.build_ext import build_ext
            from setuptools.dist import Distribution
        except ImportError as error:
            raise unittest.SkipTest('setuptools is required for the real sdist inventory test') from error

        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / 'source'
        source.mkdir()
        for name in ('setup.py', 'build_support.py', 'MANIFEST.in', 'README.md'):
            shutil.copy2(ROOT / name, source / name)
        for name in ('csrc', 'deep_ep'):
            shutil.copytree(ROOT / name, source / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.pyd'))
        (source / 'third-party').mkdir()
        for path in (ROOT / 'third-party').glob('*.LICENSE'):
            shutil.copy2(path, source / 'third-party' / path.name)

        # An initialized submodule fixture proves that sdist preserves dependency
        # headers. It is not the pinned real header and is never compiled/patched.
        dependency = source / 'third-party/deep_jit/include/deep_jit/backend/ascend/backend.hpp'
        dependency.parent.mkdir(parents=True)
        dependency.write_text('// packaging fixture only\n', encoding='utf-8')
        toolkit = root / 'toolkit'
        for relative in ('include', 'asc/include', 'lib64'):
            (toolkit / relative).mkdir(parents=True, exist_ok=True)

        class NoCompileBuildExtension(build_ext):
            def build_extensions(self):
                raise AssertionError('sdist inventory must not compile an extension')

        modules = {name: ModuleType(name) for name in
                   ('torch_npu', 'torch', 'torch.utils', 'torch.utils.cpp_extension')}
        modules['torch_npu'].__file__ = str(root / 'torch_npu/__init__.py')
        extension = modules['torch.utils.cpp_extension']
        extension.BuildExtension = NoCompileBuildExtension
        extension.include_paths = lambda: []
        extension.library_paths = lambda: []
        captured = {}

        def capture_setup(**kwargs):
            captured.update(kwargs)

        original_check_output = subprocess.check_output

        def git_output(args, **kwargs):
            if args[0] != 'git':
                return original_check_output(args, **kwargs)
            return b'' if args[1] == 'status' else b'fixture\n'

        with working_directory(source), patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {'ASCEND_HOME_PATH': str(toolkit)}), \
                patch.object(setuptools, 'setup', side_effect=capture_setup), \
                patch.object(subprocess, 'check_output', side_effect=git_output), \
                redirect_stdout(io.StringIO()):
            runpy.run_path(str(source / 'setup.py'), run_name='__main__')

        with working_directory(source), redirect_stdout(io.StringIO()):
            distribution = Distribution(captured)
            distribution.script_name = 'setup.py'
            distribution.script_args = ['sdist', '--formats=gztar', '--dist-dir', str(root / 'dist')]
            distribution.parse_command_line()
            distribution.run_commands()

        archives = list((root / 'dist').glob('*.tar.gz'))
        if len(archives) != 1:
            raise AssertionError(archives)
        with tarfile.open(archives[0], 'r:gz') as archive:
            cls.members = {name.partition('/')[2] for name in archive.getnames()}
        cls.required_csrc = {path.relative_to(source).as_posix()
                             for path in (source / 'csrc').rglob('*') if path.is_file()}
        cls.required_headers = {path.relative_to(source).as_posix()
                                for path in (source / 'deep_ep/include').rglob('*')
                                if path.suffix in ('.h', '.hpp')}
        cls.build_py_class = captured['cmdclass']['build_py']

    def test_sdist_contains_all_host_extension_sources_and_headers(self):
        self.assertTrue(self.required_csrc)
        self.assertEqual(self.required_csrc - self.members, set())

    def test_sdist_contains_initialized_deep_jit_headers(self):
        self.assertIn('third-party/deep_jit/include/deep_jit/backend/ascend/backend.hpp', self.members)

    def test_sdist_contains_device_headers_and_build_helper(self):
        self.assertTrue(self.required_headers)
        self.assertEqual(self.required_headers - self.members, set())
        self.assertIn('build_support.py', self.members)

    def test_sdist_preserves_declared_third_party_notices(self):
        for name in ('cann.LICENSE', 'ascend-deepep.LICENSE'):
            self.assertIn('third-party/' + name, self.members)

    def test_persistent_environment_values_round_trip_as_data(self):
        from setuptools.dist import Distribution
        value = "/tmp/user's cache\\nested\nsecond-line"
        with tempfile.TemporaryDirectory() as temporary:
            command = self.build_py_class(Distribution())
            command.build_lib = temporary
            with patch.dict(os.environ, {'EP_JIT_CACHE_DIR': value}):
                command.generate_default_envs()
            generated = (Path(temporary) / 'deep_ep/envs.py').read_text(encoding='utf-8')
            namespace = {}
            exec(compile(generated, 'generated-envs.py', 'exec'), namespace)
            self.assertEqual(namespace['persistent_envs']['EP_JIT_CACHE_DIR'], value)


if __name__ == '__main__':
    unittest.main()
