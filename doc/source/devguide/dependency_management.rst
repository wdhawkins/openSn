.. _dependency-maintenance:

Version and Dependency Maintenance
==================================

OpenSn keeps release and dependency policy in a small set of authoritative
files. This page describes how those files are used and how to change them
without leaving the CMake build, Python package, documentation, or dependency
bootstrap out of sync.

Dependency metadata
-------------------

``dependencies.json`` is the authoritative dependency policy manifest. Each
entry can contain:

``minimum``
   The oldest version supported by the OpenSn build. A null value means that
   OpenSn does not impose a version constraint.

``bootstrap``
   The exact tested release downloaded by ``tools/dependencies``, including its
   URL and SHA-256 checksum. The bootstrap version may be newer than the minimum
   supported version.

``documentation_order``
   The dependencies displayed in the generated installation table and their
   order.

``cmake/OpenSnDependencyVersions.cmake`` reads every entry in the manifest and
defines ``OPENSN_<NAME>_MIN_VERSION`` when a minimum is set and
``OPENSN_<NAME>_BOOTSTRAP_VERSION``, ``_URL``, and ``_SHA256`` when a bootstrap
block is present. ``<NAME>`` is the entry name in upper case with ``-``
replaced by ``_``, for example ``OPENSN_MPICPP_LITE_MIN_VERSION``. The main
build and the dependency bootstrap use these variables. ``setup.py`` reads the
Python minimum, and ``doc/source/conf.py`` generates documentation values and
the dependency table. The manifest therefore replaces version literals in those
consumers; it does not replace their package-discovery or build logic.

Updating a dependency
---------------------

To update an existing dependency:

#. Decide independently whether the supported minimum, tested bootstrap
   release, or both are changing.
#. Update the dependency entry in ``dependencies.json``. When changing a
   bootstrap archive, update its version, URL, and SHA-256 checksum together.
#. If OpenSn's Spack recipe supports the dependency, update the matching
   constraint in ``distribution/spack/packages/opensn/package.py``. The Spack
   recipe remains separate because it describes a complete Spack concretization,
   including variants and conflicts that do not belong in the bootstrap
   manifest.
#. If the package's CMake targets, components, options, or archive layout
   changed, update the corresponding ``find_package`` and
   ``ExternalProject_Add`` logic.
#. Build the dependency stack in a clean prefix, then configure and build
   OpenSn against that prefix and run the tests.

Do not raise the minimum merely because the bootstrap release changed. The
minimum documents the compatibility contract; the bootstrap selects the
version used for routine development and continuous integration.

Download a new bootstrap archive and calculate its checksum before editing the
manifest, for example:

.. code-block:: shell

   curl -L <archive-url> -o /tmp/dependency.tar.gz
   cmake -E sha256sum /tmp/dependency.tar.gz

Tool minimums need two additional consistency updates. When changing the CMake
minimum, update ``cmake_minimum_required`` in the root ``CMakeLists.txt`` and
the ``cmake`` build requirement in ``pyproject.toml``. These values are needed
before either build system can read the manifest. When changing a tool minimum
or constraint represented by Spack, update the matching Spack package entry as
well.

Adding a dependency
-------------------

Adding a new dependency requires each layer that uses it to be updated:

#. Add its metadata to ``dependencies.json``. Include a ``bootstrap`` block only
   when ``tools/dependencies`` will install it, and add it to
   ``documentation_order`` when it belongs in the installation table. CMake
   receives the corresponding ``OPENSN_<NAME>_*`` variables automatically.
#. Add ``find_package`` logic to the root ``CMakeLists.txt`` and link its
   imported target with the narrowest correct visibility.
#. If an exported OpenSn target exposes the dependency, add the matching
   ``find_dependency`` call to ``cmake/OpenSnConfig.cmake.in``. This is what
   allows downstream applications to reconstruct the exported target graph.
#. If it is part of the everyday dependency bootstrap, add an
   ``ExternalProject_Add`` recipe under ``tools/dependencies`` that installs it
   into ``CMAKE_INSTALL_PREFIX``. Do not add a package-specific environment
   variable when the installed CMake package can be found through
   ``CMAKE_PREFIX_PATH``.
#. Update ``distribution/spack/packages/opensn/package.py`` and any required
   variants. This remains necessary even when Spack is not used for routine
   development.
#. Add focused build or runtime coverage.

Updating the OpenSn version
---------------------------

``VERSION.txt`` is the authoritative OpenSn version and must contain three
numeric components, ``MAJOR.MINOR.PATCH``, such as ``1.2.3``. CMake, the Python
package, and Sphinx read it directly, and CMake and ``setup.py`` reject any
other format. The build reconfigures automatically when the file changes.

The shared-library ``SOVERSION`` is the major version. Increment ``MAJOR`` for
changes that break the C++ or Python interface, ``MINOR`` for
backward-compatible features, and ``PATCH`` for backward-compatible fixes.

Making a release
----------------

A release is an annotated ``vMAJOR.MINOR.PATCH`` tag on a commit in ``main``,
plus a GitHub release for that tag. Publishing a release requires a maintainer
with permission to push tags to ``Open-Sn/opensn``. No workflow runs on tags or
releases, and the documentation site is deployed from ``main``, so publishing
the release does not start any builds.

The steps below use the remote names from :doc:`workflow`, where ``upstream``
is ``Open-Sn/opensn``.

#. **Open a release pull request.** On a branch from the latest ``main``:

   - Set ``VERSION.txt`` to the new version.
   - In ``distribution/spack/packages/opensn/package.py``, add
     ``version("X.Y.Z", tag="vX.Y.Z")`` above the previous release and point
     ``url`` at ``.../archive/refs/tags/vX.Y.Z.tar.gz``. Spack resolves the tag
     only when it fetches, so the entry can be added before the tag exists.
   - Update the ``opensn@`` examples in ``distribution/spack/README.md``.

   Merge the pull request after CI passes.

#. **Check the merged commit.** Fetch ``upstream`` and confirm that CI passed on
   the merge commit in ``main``. From a clean checkout of that commit, confirm
   that ``python setup.py --version`` and the CMake configure report the new
   version.

#. **Tag the release.** Create an annotated tag on the merge commit and push it:

   .. code-block:: shell

      git fetch upstream
      git tag -a vX.Y.Z -m "OpenSn vX.Y.Z" <merge-commit>
      git push upstream vX.Y.Z

   Do not move or delete a tag after it has been pushed. Anything that was
   fetched from it, including Spack, keeps the old commit. Fix a bad release
   with a new patch release.

#. **Publish the GitHub release.** With the GitHub CLI:

   .. code-block:: shell

      gh release create vX.Y.Z --repo Open-Sn/opensn --verify-tag \
        --title vX.Y.Z --generate-notes

   In the web interface, use **Releases** > **Draft a new release**, choose the
   existing tag, and select **Generate release notes**. Edit the generated
   notes into a short *What's Changed* list of user-visible changes and keep the
   *Full Changelog* comparison link. Do not upload files. GitHub attaches the
   source archives that the Spack ``url`` points to.
