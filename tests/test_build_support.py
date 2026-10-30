"""Run with `python -m unittest discover -s tests -p test_build_support.py`."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_support as support


class BuildSupportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def toolkit(root):
        for relative in ('include', 'asc/include', 'lib64'):
            (root / relative).mkdir(parents=True, exist_ok=True)

    def test_native_architecture_and_aliases(self):
        self.toolkit(self.root)
        for host, sdk in (('x86_64', 'x86_64'), ('AMD64', 'x86_64'),
                          ('aarch64', 'aarch64'), ('arm64', 'aarch64')):
            native = self.root / f'{sdk}-linux'
            self.toolkit(native)
            with self.subTest(host=host), patch.object(support.platform, 'machine', return_value=host):
                includes, library = support.get_ascend_build_paths(self.root)
                self.assertEqual(includes, [str(native / name) for name in ('include', 'asc', 'asc/include')])
                self.assertEqual(library, str(native / 'lib64'))

    def test_canonical_root_fallback(self):
        self.toolkit(self.root)
        with patch.object(support.platform, 'machine', return_value='x86_64'):
            includes, library = support.get_ascend_build_paths(self.root)
        self.assertEqual(includes[0], str(self.root / 'include'))
        self.assertEqual(library, str(self.root / 'lib64'))

    def test_wrong_cpu_tree_is_rejected(self):
        self.toolkit(self.root / 'aarch64-linux')
        with patch.object(support.platform, 'machine', return_value='x86_64'):
            with self.assertRaisesRegex(RuntimeError, 'host CPU x86_64'):
                support.get_ascend_build_paths(self.root)

    def fixture(self, text=None):
        text = text if text is not None else '#pragma once\n' + support.DEEP_JIT_TOOLKIT_ANCHOR + '\n'
        source = self.root / 'source/include'
        backend = source / support.DEEP_JIT_BACKEND_PATH
        backend.parent.mkdir(parents=True)
        backend.write_text(text, encoding='utf-8')
        (backend.parent / 'device.hpp').write_text('// sibling include\n', encoding='utf-8')
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return source, self.root / 'build/include', text, digest

    def test_overlay_is_isolated_and_repeatable(self):
        source, output, original, digest = self.fixture()
        with patch.object(support, 'DEEP_JIT_BACKEND_SHA256', digest):
            support.prepare_deep_jit_headers(source, output)
            first = (output / support.DEEP_JIT_BACKEND_PATH).read_bytes()
            support.prepare_deep_jit_headers(source, output)
            self.assertEqual((output / support.DEEP_JIT_BACKEND_PATH).read_bytes(), first)
        self.assertEqual((source / support.DEEP_JIT_BACKEND_PATH).read_text(encoding='utf-8'), original)
        self.assertTrue((output / support.DEEP_JIT_BACKEND_PATH.parent / 'device.hpp').is_file())
        self.assertIn(b'home_path / "asc/include/adv_api"', first)
        self.assertIn(b'#if defined(__x86_64__)', first)
        self.assertIn(b'#elif defined(__aarch64__)', first)

    def test_unknown_header_version_is_rejected(self):
        source, output, original, digest = self.fixture()
        (source / support.DEEP_JIT_BACKEND_PATH).write_text(original + '// changed\n', encoding='utf-8')
        with patch.object(support, 'DEEP_JIT_BACKEND_SHA256', digest):
            with self.assertRaisesRegex(RuntimeError, 'differs from pinned'):
                support.prepare_deep_jit_headers(source, output)
        self.assertFalse(output.exists())

    def test_exact_anchor_is_required(self):
        source, output, _, digest = self.fixture('// different toolkit locator\n')
        with patch.object(support, 'DEEP_JIT_BACKEND_SHA256', digest):
            with self.assertRaisesRegex(RuntimeError, 'anchor did not match exactly once'):
                support.prepare_deep_jit_headers(source, output)

    def test_missing_submodule_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, 'initialize third-party/deep_jit'):
            support.prepare_deep_jit_headers(self.root / 'missing', self.root / 'build')


if __name__ == '__main__':
    unittest.main()
