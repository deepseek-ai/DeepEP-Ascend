"""CPU-only build helpers; importing this module does not load Torch or an NPU."""

import hashlib
from pathlib import Path
import platform
import shutil


DEEP_JIT_REVISION = '65f513952b8771408a19202f65e36398522db6db'
DEEP_JIT_BACKEND_SHA256 = '0b991603ef83d7d220c3cf18e7ce38d026ea2a40dbcf463512ba71e5ccdf349d'
DEEP_JIT_BACKEND_PATH = Path('deep_jit/backend/ascend/backend.hpp')
DEEP_JIT_TOOLKIT_ANCHOR = '''        const auto asc_devkit_include = home_path / "aarch64-linux/asc/include/adv_api";
        DJ_HOST_ASSERT(std::filesystem::is_directory(asc_devkit_include),
                       "Ascend adv_api include directory was not found: {}", asc_devkit_include.string());'''
DEEP_JIT_TOOLKIT_REPLACEMENT = '''        // Prefer CANN's architecture-neutral installation links, then the
        // native host SDK. This is a DeepEP build-time compatibility overlay.
        auto asc_devkit_include = home_path / "asc/include/adv_api";
        if (not std::filesystem::is_directory(asc_devkit_include)) {
#if defined(__x86_64__) || defined(_M_X64)
            asc_devkit_include = home_path / "x86_64-linux/asc/include/adv_api";
#elif defined(__aarch64__) || defined(_M_ARM64)
            asc_devkit_include = home_path / "aarch64-linux/asc/include/adv_api";
#endif
        }
        DJ_HOST_ASSERT(std::filesystem::is_directory(asc_devkit_include),
                       "Ascend adv_api include directory was not found for this host: {}", asc_devkit_include.string());'''


def get_ascend_build_paths(ascend_home):
    """Select existing native CANN directories, with standard root-link fallback."""
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine)
    root = Path(ascend_home)
    candidates = (root / f'{machine}-linux', root)

    def directory(relative):
        for candidate in candidates:
            path = candidate / relative
            if path.is_dir():
                return str(path)
        searched = ', '.join(str(candidate / relative) for candidate in candidates)
        raise RuntimeError(f'Cannot find CANN {relative} for host CPU {machine}; searched: {searched}')

    return [directory('include'), directory('asc'), directory('asc/include')], directory('lib64')


def prepare_deep_jit_headers(source_include, output_include):
    """Copy the pinned headers and patch only the build copy, failing on drift.

    The entire backend header is checked as well as the replacement anchor. A
    dependency update therefore requires an explicit review of this workaround.
    The source checkout and the installed CANN tree are never modified.
    """
    source_include = Path(source_include).resolve()
    output_include = Path(output_include).resolve()
    if (source_include == output_include or source_include in output_include.parents
            or output_include in source_include.parents):
        raise RuntimeError('DeepJIT source and compatibility output directories must be separate')
    source_header = source_include / DEEP_JIT_BACKEND_PATH
    if not source_header.is_file():
        raise RuntimeError('DeepJIT headers are missing; initialize third-party/deep_jit before building')
    text = source_header.read_text(encoding='utf-8')
    # read_text normalizes CRLF checkouts to the pinned repository's LF content.
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if digest != DEEP_JIT_BACKEND_SHA256:
        raise RuntimeError(
            f'DeepJIT backend header differs from pinned {DEEP_JIT_REVISION} '
            f'(SHA256 {digest}); review the CANN host compatibility overlay before building')
    if text.count(DEEP_JIT_TOOLKIT_ANCHOR) != 1:
        raise RuntimeError('DeepJIT CANN compatibility anchor did not match exactly once')
    patched = text.replace(DEEP_JIT_TOOLKIT_ANCHOR, DEEP_JIT_TOOLKIT_REPLACEMENT, 1)

    # Copy the whole include tree so quoted relative includes retain their usual
    # meaning. Always start from the source version; repeated builds are idempotent.
    shutil.copytree(source_include, output_include, dirs_exist_ok=True)
    with (output_include / DEEP_JIT_BACKEND_PATH).open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(patched)
    return str(output_include)
