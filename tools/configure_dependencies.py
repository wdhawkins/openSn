#!/usr/bin/env python3
"""
OpenSn Dependency Installer

This script downloads and installs selected OpenSn dependencies.
It checks for required MPI compiler executables (mpicc, mpicxx, mpifort)
as well as curl and cmake (which are assumed to be installed).
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
import logging
from contextlib import contextmanager

@contextmanager
def pushd(new_dir):
    """Temporarily change directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)

def run_command(cmd, cwd=None, env=None, logger=None, verbose=False):
    """Run a shell command and raise an error if it fails."""
    if verbose:
        print(f"Running command: {cmd}")
    if logger:
        logger.info(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err_msg = result.stderr.decode("utf-8")
        if logger:
            logger.error(f"Command failed: {cmd}\nError: {err_msg}")
        raise RuntimeError(f"Command failed: {cmd}\nError: {err_msg}")
    return result.stdout.decode("utf-8")

def check_executable(name):
    """Verify that the given executable is available."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable '{name}' not found.")

def download_package(url, destination, logger, verbose=False):
    """Download a package tarball using curl."""
    cmd = f"curl -L --output {destination} {url}"
    try:
        run_command(cmd, logger=logger, verbose=verbose)
        logger.info(f"Downloaded to {destination}")
        return True
    except Exception as ex:
        logger.error(f"Download failed: {ex}")
        return False

def get_mpi_env():
    """Return an environment with MPI compilers set if not already defined."""
    env = os.environ.copy()
    env.setdefault("CC", shutil.which("mpicc"))
    env.setdefault("CXX", shutil.which("mpicxx"))
    env.setdefault("FC", shutil.which("mpifort"))
    return env

def prepare_source(pkg, version, install_dir, extracted_dir, logger, verbose):
    """
    Copy the tarball from the downloads directory to the src directory and extract it.
    Returns the absolute path to the extracted directory.
    """
    src_dir = os.path.join(install_dir, "src")
    downloads_dir = os.path.join(install_dir, "downloads")
    tarball = os.path.join(downloads_dir, f"{pkg}-{version}.tar.gz")
    dst_tarball = os.path.join(src_dir, f"{pkg}-{version}.tar.gz")
    shutil.copy(tarball, dst_tarball)
    with pushd(src_dir):
        run_command(f"tar -zxf {pkg}-{version}.tar.gz", logger=logger, verbose=verbose)
    extracted_path = os.path.join(src_dir, extracted_dir)
    if not os.path.isdir(extracted_path):
        raise RuntimeError(f"Expected source directory {extracted_path} not found.")
    return extracted_path

def delete_package(pkg, record, logger):
    """
    Delete the installed package.
    """
    install_path = record.get("install_path")
    if install_path and os.path.isdir(install_path):
        logger.info(f"Removing installation directory: {install_path}")
        shutil.rmtree(install_path)

def install_boost_package(pkg, version, install_dir, jobs, logger, verbose):
    """Install Boost headers."""
    pkg_install_dir = os.path.join(install_dir, f"{pkg}-{version}")
    os.makedirs(pkg_install_dir, exist_ok=True)
    extracted_path = prepare_source(pkg, version, install_dir, f"boost_{version}", logger, verbose)
    with pushd(extracted_path):
        include_dir = os.path.join(pkg_install_dir, "include")
        os.makedirs(include_dir, exist_ok=True)
        run_command(f"cp -r boost {include_dir}", logger=logger, verbose=verbose)
    return pkg_install_dir

def install_lua_package(pkg, version, install_dir, jobs, logger, verbose):
    """Install Lua."""
    pkg_install_dir = os.path.join(install_dir, f"{pkg}-{version}")
    os.makedirs(pkg_install_dir, exist_ok=True)
    extracted_path = prepare_source(pkg, version, install_dir, f"lua-{version}", logger, verbose)
    with pushd(extracted_path):
        env = get_mpi_env()
        os_tag = "linux" if sys.platform != "darwin" else "macosx"
        run_command(f"make {os_tag} MYCFLAGS=-fPIC MYLIBS=-lncurses -j{jobs}", env=env, logger=logger, verbose=verbose)
        run_command(f"make install INSTALL_TOP={pkg_install_dir}", env=env, logger=logger, verbose=verbose)
    return pkg_install_dir

def install_petsc_package(pkg, version, install_dir, jobs, logger, verbose):
    """Install PETSc."""
    pkg_install_dir = os.path.join(install_dir, f"{pkg}-{version}")
    os.makedirs(pkg_install_dir, exist_ok=True)
    extracted_path = prepare_source(pkg, version, install_dir, f"{pkg}-{version}", logger, verbose)
    with pushd(extracted_path):
        env = get_mpi_env()
        config_cmd = (
            f"./configure --prefix={pkg_install_dir} "
            "--download-hypre=1 "
            "--with-ssl=0 "
            "--with-debugging=0 "
            "--with-pic=1 "
            "--with-shared-libraries=1 "
            "--download-bison=1 "
            "--download-fblaslapack=1 "
            "--download-metis=1 "
            "--download-parmetis=1 "
            "--download-superlu_dist=1 "
            "--download-ptscotch=1 "
            "--with-cxx-dialect=C++11 "
            "--with-64-bit-indices "
            f"CC={env['CC']} CXX={env['CXX']} FC={env['FC']} "
            'FFLAGS="-fallow-argument-mismatch" '
            'FCFLAGS="-fallow-argument-mismatch" '
            'F90FLAGS="-fallow-argument-mismatch" '
            'F77FLAGS="-fallow-argument-mismatch" '
            'COPTFLAGS="-O3" '
            'CXXOPTFLAGS="-O3" '
            'FOPTFLAGS="-O3 -fallow-argument-mismatch" '
            f"PETSC_DIR={extracted_path}"
        )
        run_command(config_cmd, env=env, logger=logger, verbose=verbose)
        run_command(f"make all -j{jobs}", env=env, logger=logger, verbose=verbose)
        run_command("make install", env=env, logger=logger, verbose=verbose)
    return pkg_install_dir

def install_vtk_package(pkg, version, install_dir, jobs, logger, verbose):
    """Install VTK."""
    pkg_install_dir = os.path.join(install_dir, f"{pkg}-{version}")
    os.makedirs(pkg_install_dir, exist_ok=True)
    extracted_path = prepare_source(pkg, version, install_dir, f"VTK-{version}", logger, verbose)
    build_dir = os.path.join(extracted_path, "build")
    os.makedirs(build_dir, exist_ok=True)
    env = get_mpi_env()
    cmake_cmd = (
        f"cmake -DCMAKE_INSTALL_PREFIX={pkg_install_dir} "
        "-DBUILD_SHARED_LIBS=ON "
        "-DVTK_USE_MPI=ON "
        "-DVTK_GROUP_ENABLE_StandAlone=WANT "
        "-DVTK_GROUP_ENABLE_Rendering=DONT_WANT "
        "-DVTK_GROUP_ENABLE_Imaging=DONT_WANT "
        "-DVTK_GROUP_ENABLE_Web=DONT_WANT "
        "-DVTK_GROUP_ENABLE_Qt=DONT_WANT "
        "-DVTK_MODULE_USE_EXTERNAL_VTK_hdf5=ON "
        "-DCMAKE_BUILD_TYPE=Release ../"
    )
    with pushd(build_dir):
        run_command(cmake_cmd, env=env, logger=logger, verbose=verbose)
        run_command(f"make -j{jobs}", env=env, logger=logger, verbose=verbose)
        run_command("make install", env=env, logger=logger, verbose=verbose)
    return pkg_install_dir

def install_caliper_package(pkg, version, install_dir, jobs, logger, verbose):
    """Install Caliper."""
    pkg_install_dir = os.path.join(install_dir, f"{pkg}-{version}")
    os.makedirs(pkg_install_dir, exist_ok=True)
    extracted_path = prepare_source(pkg, version, install_dir, f"Caliper-{version}", logger, verbose)
    build_dir = os.path.join(extracted_path, "build")
    os.makedirs(build_dir, exist_ok=True)
    env = get_mpi_env()
    cmake_cmd = f"cmake -DCMAKE_INSTALL_PREFIX={pkg_install_dir} -DWITH_MPI=ON -DWITH_KOKKOS=OFF ../"
    with pushd(build_dir):
        run_command(cmake_cmd, env=env, logger=logger, verbose=verbose)
        run_command(f"make -j{jobs}", env=env, logger=logger, verbose=verbose)
        run_command("make install", env=env, logger=logger, verbose=verbose)
    return pkg_install_dir

def install_hdf5_package(pkg, version, install_dir, jobs, logger, verbose):
    """Install HDF5."""
    pkg_install_dir = os.path.join(install_dir, f"{pkg}-{version}")
    os.makedirs(pkg_install_dir, exist_ok=True)
    extracted_path = prepare_source(pkg, version, install_dir, f"{pkg}-{version}", logger, verbose)
    build_dir = os.path.join(extracted_path, "build")
    os.makedirs(build_dir, exist_ok=True)
    env = get_mpi_env()
    cmake_cmd = f"cmake -DCMAKE_INSTALL_PREFIX={pkg_install_dir} .."
    with pushd(build_dir):
        run_command(cmake_cmd, env=env, logger=logger, verbose=verbose)
        run_command(f"make -j{jobs}", env=env, logger=logger, verbose=verbose)
        run_command("make install", env=env, logger=logger, verbose=verbose)
    return pkg_install_dir

def main():
    parser = argparse.ArgumentParser(
        description="OpenSn Dependency Installer",
        epilog="Select packages using --packages (comma separated, or 'all')"
    )
    parser.add_argument("-d", "--directory", type=str, required=True,
                        help="Installation directory for dependencies")
    parser.add_argument("-j", "--jobs", type=int, default=4,
                        help="Number of compile jobs (default: 4)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download packages and exit")
    parser.add_argument("--packages", type=str, default="all",
                        help="Comma-separated list of packages to install (default: all)")
    parser.add_argument("--uninstall", action="store_true",
                        help="Uninstall specified packages (must be used with --packages listing one or more packages)")
    parser.add_argument("--upgrade", action="store_true",
                        help="Upgrade installed packages if versions differ")
    parser.add_argument("--list-installed", action="store_true",
                        help="List installed packages from the state file and exit")
    parser.add_argument("--list-available", action="store_true",
                        help="List available packages and their versions and exit")
    parser.add_argument("--clean", action="store_true",
                        help="Clean downloads and src directories and exit")
    args = parser.parse_args()

    print("Resolving installation directories")
    install_dir = os.path.abspath(args.directory)
    src_dir = os.path.join(install_dir, "src")
    downloads_dir = os.path.join(install_dir, "downloads")
    logs_dir = os.path.join(install_dir, "logs")
    bin_dir = os.path.join(install_dir, "bin")
    for d in [install_dir, src_dir, downloads_dir, logs_dir, bin_dir]:
        os.makedirs(d, exist_ok=True)

    # Set up logger
    log_file_path = os.path.join(logs_dir, "install.log")
    logger = logging.getLogger('opensn_installer')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Starting OpenSn dependency installation")

    # Package info
    packages_info = {
        "boost": {
            "version": "1_87_0",
            "url": "https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz",
            "installer": install_boost_package
        },
        "lua": {
            "version": "5.4.6",
            "url": "https://www.lua.org/ftp/lua-5.4.6.tar.gz",
            "installer": install_lua_package
        },
        "petsc": {
            "version": "3.23.0",
            "url": "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.23.0.tar.gz",
            "installer": install_petsc_package
        },
        "vtk": {
            "version": "9.4.2",
            "url": "https://www.vtk.org/files/release/9.4/VTK-9.4.2.tar.gz",
            "installer": install_vtk_package
        },
        "caliper": {
            "version": "2.10.0",
            "url": "https://github.com/LLNL/Caliper/archive/refs/tags/v2.10.0.tar.gz",
            "installer": install_caliper_package
        },
        "hdf5": {
            "version": "1.14.6",
            "url": "https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_6/downloads/hdf5-1.14.6.tar.gz",
            "installer": install_hdf5_package
        }
    }

    # Handle --clean option
    if args.clean:
        print("Cleaning downloads and src directories")
        try:
            shutil.rmtree(downloads_dir)
            os.makedirs(downloads_dir, exist_ok=True)
            shutil.rmtree(src_dir)
            os.makedirs(src_dir, exist_ok=True)
        except Exception as e:
            print(f"Error during cleaning: {e}")
        sys.exit(0)

    print("Checking required executables")
    try:
        check_executable("curl")
        check_executable("cmake")
        for exe in ["mpicc", "mpicxx", "mpifort"]:
            check_executable(exe)
    except RuntimeError as e:
        print(f"{e}")
        sys.exit(1)

    # Handle --list-available option
    if args.list_available:
        print("Listing available packages")
        try:
            for pkg, info in packages_info.items():
                print(f"  {pkg}: {info['version']}")
        except Exception as e:
            print(f"Error listing available packages: {e}")
        sys.exit(0)

    # Read state file to track installed packages
    state_file = os.path.join(install_dir, "installed_packages.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            installed_state = json.load(f)
    else:
        installed_state = {}

    # Handle --list-installed option
    if args.list_installed:
        if installed_state:
            print("Installed packages:")
            for pkg, record in installed_state.items():
                print(f"  {pkg}: {record['version']} -> {record['install_path']}")
        else:
            print("No packages installed yet.")
        sys.exit(0)

    # Handle uninstall
    if args.uninstall:
        if args.packages.lower() == "all":
            print("When uninstalling, please specify one or more packages using --packages (cannot be 'all').")
            sys.exit(1)
        uninstall_packages = [p.strip() for p in args.packages.split(",") if p.strip()]
        if not uninstall_packages:
            print("No packages specified for uninstall.")
            sys.exit(1)
        for pkg in uninstall_packages:
            if pkg not in installed_state:
                print(f"{pkg} is not installed.")
                continue
            record = installed_state[pkg]
            print(f"Removing {pkg} version {record['version']}")
            logger.info(f"Uninstalling {pkg} version {record['version']}")
            try:
                delete_package(pkg, record, logger)
                del installed_state[pkg]
                print(f"{pkg} uninstalled")
            except Exception as e:
                logger.error(f"Error uninstalling {pkg}: {e}")
                print(f"Failed to uninstall {pkg}: {e}. Please see the log file: {log_file_path}")
        with open(state_file, "w") as f:
            json.dump(installed_state, f, indent=2)
        logger.info("Updated installation state after uninstallation.")
        prefix_paths = [record["install_path"] for record in installed_state.values()]
        cmake_prefix = ":".join(prefix_paths)
        env_script = os.path.join(bin_dir, "set_opensn_env.sh")
        with open(env_script, "w") as f:
            f.write(f'export CMAKE_PREFIX_PATH="{cmake_prefix}"${{CMAKE_PREFIX_PATH:+:${{CMAKE_PREFIX_PATH}}}}\n')
        os.chmod(env_script, 0o755)
        sys.exit(0)

    # Determine selected packages
    if args.packages.lower() == "all":
        selected_packages = list(packages_info.keys())
    else:
        selected_packages = [p.strip() for p in args.packages.split(",") if p.strip() in packages_info]
        if not selected_packages:
            logger.error("No valid packages selected.")
            print("No valid packages selected. Please see the log file:", log_file_path)
            sys.exit(1)
    logger.info(f"Selected packages: {', '.join(selected_packages)}")

    # Download packages
    download_errors = False
    for pkg in selected_packages:
        info = packages_info[pkg]
        tarball_path = os.path.join(downloads_dir, f"{pkg}-{info['version']}.tar.gz")
        if pkg in installed_state and installed_state[pkg]["version"] == info["version"]:
            print(f"{pkg} is already installed, skipping download")
            continue
        if not os.path.exists(tarball_path):
            print(f"Downloading {pkg} version {info['version']}")
            logger.info(f"Downloading {pkg} version {info['version']}")
            if not download_package(info["url"], tarball_path, logger, args.verbose):
                print(f"Failed to download {pkg}. Please see the log file: {log_file_path}")
                logger.error(f"Failed to download {pkg}")
                download_errors = True
        else:
            logger.info(f"{pkg} tarball already exists. Skipping download.")
            print(f"{pkg} tarball already exists. Skipping download.")
    if download_errors:
        sys.exit(1)

    if args.download_only:
        logger.info("Download-only mode; exiting after downloads.")
        sys.exit(0)

    # Install
    install_errors = []
    for pkg in selected_packages:
        info = packages_info[pkg]
        if pkg in installed_state:
            record = installed_state[pkg]
            if record["version"] == info["version"]:
                logger.info(f"{pkg} version {record['version']} already installed. Skipping.")
                print(f"{pkg} version {record['version']} already installed.")
                continue
            else:
                if args.upgrade:
                    print(f"Upgrading {pkg} from {record['version']} to {info['version']}")
                    logger.info(f"Upgrading {pkg} from {record['version']} to {info['version']}")
                    try:
                        delete_package(pkg, record, logger)
                    except Exception as e:
                        print(f"Error during upgrade cleanup for {pkg}: {e}")
                        install_errors.append(pkg)
                        continue
                else:
                    logger.info(f"{pkg} version mismatch: installed {record['version']} but expected {info['version']}.")
                    print(f"{pkg} version mismatch: installed {record['version']} but expected {info['version']}. Please uninstall first if you want to reinstall.")
                    continue
        print(f"Installing {pkg} version {info['version']}")
        try:
            logger.info(f"Installing {pkg} version {info['version']}")
            pkg_install_dir = info["installer"](pkg, info["version"], install_dir, args.jobs, logger, args.verbose)
            installed_state[pkg] = {"version": info["version"], "install_path": pkg_install_dir}
        except Exception as e:
            logger.error(f"Error installing {pkg}: {e}")
            install_errors.append(pkg)
            print(f"{pkg} encountered an error: {e}. Please see the log file: {log_file_path}")

    with open(state_file, "w") as f:
        json.dump(installed_state, f, indent=2)
    logger.info("Updated installation state.")

    if install_errors:
        logger.error(f"Failed to install: {', '.join(install_errors)}")
        print(f"Installation failed for: {', '.join(install_errors)}. Please see the log file: {log_file_path}")
        sys.exit(1)

    print("Creating environment setup script")
    try:
        prefix_paths = [record["install_path"] for record in installed_state.values()]
        cmake_prefix = ":".join(prefix_paths)
        env_script = os.path.join(bin_dir, "set_opensn_env.sh")
        with open(env_script, "w") as f:
            f.write(f'export CMAKE_PREFIX_PATH="{cmake_prefix}"${{CMAKE_PREFIX_PATH:+:${{CMAKE_PREFIX_PATH}}}}\n')
        os.chmod(env_script, 0o755)
    except Exception as e:
        print(f"Error creating environment script: {e}")

    print("\nOpenSn dependency installation complete.")
    print("To update environment variables, run:")
    print(f"    $ source {env_script}\n")

if __name__ == '__main__':
    main()
