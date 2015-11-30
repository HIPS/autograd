

from subprocess import check_output
versionstr = check_output("git describe --tags").strip().decode('utf-8')
versionstr = versionstr.lstrip('v')

parts = versionstr.split('-')
version = parts[0]

# pad verson if necessary
version = version.split('.')
if len(version) != 3:
    version += ['0']*(3-len(version))

major, minor, micro = map(int, version)

if len(parts) == 1:
    __version__ = "{:d}.{:d}.{:d}".format(major, minor, micro)
else:
    revision, sha = parts[1:]
    __version__ = "{:d}.{:d}.{:d}.post{:03d}+{}".format(
        major, minor, micro, int(revision), sha,
    )
