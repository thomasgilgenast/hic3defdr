"""
Module for simplifying Docker image building, tagging, and pushing.

Our Dockerfile expects a ``VERSION`` build-arg, and by convention we tag our
Docker images using the output of ``git describe``. Running
::

    $ python _docker.py build

builds both the normal and slim Docker images, making sure that the correct
version information is passed to the image build process and that the resulting
image is tagged correctly.

Running
::

    $ python _docker.py promote

promotes the current images (normal and slim) to the "latest" and "slim" tags,
respectively.

Running
::

    $ python _docker.py push
    $ python _docker.py pushlatest

pushes the explicitly tagged (push) and "latest"/"slim" tagged (pushlatest)
images to Docker Hub. Pushing requires logging in with the correct Docker Hub
credentials. For example you can run
::

    $ docker login -u $DOCKER_HUB_USER -p $DOCKER_HUB_PASSWORD

To run all four "steps", simply run
::

    $ docker _build.py

"""

import sys
import subprocess

from hic3defdr._version import get_version


CMDS = {
    'build': [
        'docker build --build-arg VERSION=<version> '
        '-t creminslab/hic3defdr:<describe> .',
        'docker build --build-arg VERSION=<version> '
        '-t creminslab/hic3defdr:<describe>-slim -f docker-slim/Dockerfile .',
    ],
    'promote': [
        'docker tag creminslab/hic3defdr:<describe> creminslab/hic3defdr:latest',
        'docker tag creminslab/hic3defdr:<describe>-slim creminslab/hic3defdr:slim',
    ],
    'push': [
        'docker push creminslab/hic3defdr:<describe>',
        'docker push creminslab/hic3defdr:<describe>-slim',
    ],
    'pushlatest': [
        'docker push creminslab/hic3defdr:latest',
        'docker push creminslab/hic3defdr:slim',
    ]
}


def run_step(step):
    print('running step: %s' % step)

    # get version
    version = get_version()

    # get git describe
    describe = subprocess.check_output(
        'git describe --tags --dirty --always', shell=True)\
        .decode('utf-8').strip()

    # refuse to promote pre-release versions
    if step in ['promote', 'pushlatest'] and \
            any(x in describe for x in ['a', 'b', 'c', 'd', 'g']):
        print('prerelease or dev version detected; skipping %s' % step)
        return

    # refuse to push dirty/git versions
    if step in ['push'] and any(x in describe for x in ['d', 'g']):
        print('dev version detected; skipping %s' % step)
        return

    # run commands
    for c in CMDS[step]:
        cmd = c.replace('<version>', version).replace('<describe>', describe)
        print(cmd)
        subprocess.check_call(cmd, shell=True)
        print('done')


def main():
    if len(sys.argv) == 1:
        run_step('build')
        run_step('promote')
        run_step('push')
        run_step('pushlatest')
    else:
        run_step(sys.argv[1])


if __name__ == '__main__':
    main()
