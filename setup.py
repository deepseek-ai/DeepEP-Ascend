import ast
import re
import os
import shutil
import subprocess
import setuptools
import torch_npu

from pathlib import Path
from setuptools import Extension
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import include_paths as torch_include_paths
from torch.utils.cpp_extension import library_paths as torch_library_paths

from build_support import get_ascend_build_paths, prepare_deep_jit_headers

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_env_names = (
    'EP_JIT_CACHE_DIR',
    'EP_JIT_DEBUG',
    'EP_JIT_DUMP_ASM',
    'EP_JIT_KERNEL_DEBUG_INFO',
    'EP_JIT_LAUNCH_TIMEOUT',
    'EP_JIT_PRINT_COMPILER_COMMAND',
    'EP_JIT_PRINT_LOAD_TIME',
)


def get_package_version():
    version_file = Path(current_dir) / 'deep_ep' / '__init__.py'
    with open(version_file, 'r') as f:
        content = f.read()
    version_match = re.search(r'^__version__\s*=\s*(.*)$', content, re.MULTILINE)
    if version_match:
        public_version = ast.literal_eval(version_match.group(1))
    else:
        public_version = '0.1.0'

    # noinspection PyBroadException
    try:
        status_output = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
        if status_output:
            print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')
            assert False, 'Git working directory is not clean'

        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception:
        revision = '+local'
    return f'{public_version}{revision}'


def get_ascend_home():
    ascend_home = os.environ.get('ASCEND_HOME_PATH') or os.environ.get(
        'ASCEND_TOOLKIT_HOME', '/usr/local/Ascend/ascend-toolkit/latest')
    if not os.path.isdir(ascend_home):
        ascend_home = '/usr/local/Ascend/cann'
    if not os.path.isdir(ascend_home):
        raise RuntimeError(
            'Cannot find ASCEND_HOME_PATH. Please set the ASCEND_HOME_PATH environment variable '
            'to your CANN installation directory.'
        )
    return ascend_home


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        include_dir = prepare_deep_jit_headers(
            Path(current_dir) / 'third-party/deep_jit/include',
            Path(self.build_temp) / 'deep_jit_compat/include')
        for extension in self.extensions:
            extension.include_dirs = [include_dir] + [
                path for path in (extension.include_dirs or []) if path != include_dir]
        print(f' > DeepJIT host compatibility headers: {include_dir}')
        super().build_extensions()


class CustomBuildPy(build_py):
    def run(self):
        # Make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Then, copy csrc into the wheel for agent-side error lookup
        self.prepare_agent_files()

        # Finally, run the regular build
        build_py.run(self)

    def prepare_agent_files(self):
        # Copy csrc into the wheel for agent-side error lookup
        package_dir = os.path.join(self.build_lib, 'deep_ep')
        for name in ('csrc', ):
            dst = os.path.join(package_dir, name)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(os.path.join(current_dir, name), dst)

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in persistent_env_names:
            code += f'persistent_envs[{name!r}] = {os.environ[name]!r}\n' if name in os.environ else ''

        # Create temporary build directory
        build_include_dir = os.path.join(self.build_lib, 'deep_ep')
        os.makedirs(build_include_dir, exist_ok=True)
        with open(os.path.join(self.build_lib, 'deep_ep', 'envs.py'), 'w') as f:
            f.write(code)


if __name__ == '__main__':
    ascend_home = get_ascend_home()
    ascend_include_dirs, ascend_library_dir = get_ascend_build_paths(ascend_home)
    torch_npu_dir = os.path.dirname(torch_npu.__file__)
    package_dir = Path(current_dir) / 'deep_ep'

    sources = ['csrc/python_api.cpp']

    cxx_flags = [
        '-O3', '-g1', '-fPIC', '-std=c++20',
        '-Wno-deprecated-declarations',
        '-Wno-unused-variable',
        '-Wno-sign-compare',
        '-Wno-reorder',
        '-Wno-attributes',
    ]

    include_dirs = [
        f'{current_dir}/csrc',
        f'{current_dir}/deep_ep/include',
        f'{current_dir}/third-party/deep_jit/include',
        *ascend_include_dirs,
        f'{torch_npu_dir}/include',
        f'{torch_npu_dir}/include/third_party/acl/inc',
    ] + torch_include_paths()

    library_dirs = [
        ascend_library_dir,
        f'{torch_npu_dir}/lib',
    ] + torch_library_paths()

    libraries = ['ascendcl', 'hccl', 'hcomm', 'torch_npu']
    extra_link_args = [
        f'-Wl,-rpath,{ascend_library_dir}',
        '-Wl,-Bsymbolic-functions',
        '-Wl,--enable-new-dtags',
    ]

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Library dirs: {library_dirs}')
    print(f' > Libraries: {libraries}')
    print(f' > CXX flags: {cxx_flags}')
    print(f' > ASCEND_HOME_PATH: {ascend_home}')
    print(f' > TORCH_NPU_DIR: {torch_npu_dir}')
    # Print persistent env variables
    persistent_envs = []
    for name in persistent_env_names:
        if name in os.environ:
            persistent_envs.append((name, os.environ[name]))
    if len(persistent_envs) > 0:
        print(f' > Persistent envs:')
        for k, v in persistent_envs:
            print(f'   > {k}: {v}')
    print()

    ext_module = Extension(
        name='deep_ep._C',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=cxx_flags,
        extra_link_args=extra_link_args,
        language='c++',
    )

    # noinspection bad-argument-type
    setuptools.setup(
        name='deep_ep',
        version=get_package_version(),
        license_files=['third-party/cann.LICENSE', 'third-party/ascend-deepep.LICENSE'],
        packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),
        # Older setuptools versions do not expand recursive package-data globs.
        package_data={'deep_ep': [str(path.relative_to(package_dir)) for path in (package_dir / 'include').rglob('*')
                                 if path.suffix in ('.hpp', '.h')]},
        ext_modules=[ext_module],
        cmdclass={'build_ext': CustomBuildExtension, 'build_py': CustomBuildPy},
    )
