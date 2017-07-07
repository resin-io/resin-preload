#!/usr/bin/python3

import json
import os

from contextlib import contextmanager
from distutils.version import LooseVersion
from logging import getLogger, INFO, StreamHandler
from math import ceil, floor
from requests import get
from sh import (
    btrfs,
    docker,
    dockerd,
    e2fsck,
    inotifywait,
    kill,
    losetup,
    mount,
    parted,
    resize2fs,
    umount,
    ErrorReturnCode,
)
from shutil import copyfile, rmtree
from sys import exit
from tempfile import mkdtemp

os.environ["LANG"] = "C"

API_TOKEN = os.environ["API_TOKEN"]
API_KEY = os.environ["API_KEY"]
APP_ID = os.environ["APP_ID"]

API_HOST = os.environ["API_HOST"] or "https://api.resin.io"
REGISTRY_HOST = os.environ["REGISTRY_HOST"] or "registry2.resin.io"

log = getLogger(__name__)
log.setLevel(INFO)
log.addHandler(StreamHandler())


def human_size(size, precision=2):
    suffixes = ['', 'Ki', 'Mi', 'Gi', 'Ti']
    idx = 0
    while idx < 4 and size >= 1024:
        idx += 1
        size /= 1024
    result = "{:.{}f}".format(size, precision).rstrip("0").rstrip(".")
    return "{} {}B".format(result, suffixes[idx])


def get_offset_and_size(image, partition):
    output = parted("-s", "-m", image, "unit", "B", "p").stdout.decode("utf8")
    lines = output.strip().split("\n")
    for line in lines:
        if line.startswith("{}:".format(partition)):
            data = line.split(":")
            offset = int(data[1][:-1])
            size = int(data[3][:-1])
            return offset, size


def mount_partition(image, partition, extra_options):
    offset, size = get_offset_and_size(image, partition)
    mountpoint = mkdtemp()
    options = "offset={},sizelimit={}".format(offset, size)
    if extra_options:
        options += "," + extra_options
    mount("-o", options, image, mountpoint)
    return mountpoint


@contextmanager
def mount_context_manager(image, partition, extra_options=""):
    mountpoint = mount_partition(image, partition, extra_options)
    yield mountpoint
    umount(mountpoint)
    os.rmdir(mountpoint)


def losetup_partition(image, partition):
    offset, size = get_offset_and_size(image, partition)
    device = losetup("-f").stdout.decode("utf8").strip()
    losetup("-o", offset, "--sizelimit", size, device, image)
    return device


@contextmanager
def losetup_context_manager(image, partition):
    device = losetup_partition(image, partition)
    yield device
    losetup("-d", device)


def api(endpoint, params=None, headers=None):
    url = "{}/{}".format(API_HOST, endpoint)
    params = params or {}
    headers = headers or {}
    if API_TOKEN:
        headers["Authorization"] = "Bearer {}".format(API_TOKEN)
    elif API_KEY:
        params["apikey"] = API_KEY
    response = get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


def get_registry_token(image_repo):
    params = {
        "service": REGISTRY_HOST,
        "scope": "repository:{}:pull".format(image_repo),
    }
    return api("auth/v1/token", params)["token"]


def registry(endpoint, registry_token, headers=None, decode_json=True):
    headers = headers or {}
    headers["Authorization"] = "Bearer {}".format(registry_token)
    url = "https://{}/{}".format(REGISTRY_HOST, endpoint)
    response = get(url, headers=headers)
    response.raise_for_status()
    return response.json() if decode_json else response.content


def get_app_data():
    """Fetches application metadata"""
    endpoint = "v2/application({})?$expand=environment_variable".format(APP_ID)
    data = api(endpoint)["d"][0]
    image_repo = "{app_name}/{commit}".format(**data).lower()
    image_id = "{}/{}".format(REGISTRY_HOST, image_repo).lower()
    env = {v["name"]: v["value"] for v in data.get("environment_variable", [])}
    return {
        "appId": data["id"],
        "name": data["app_name"],
        "commit": data["commit"],
        "imageRepo": image_repo,
        "imageId": image_id,
        "env": {k: v for k, v in env.items() if not k.startswith("RESIN_")},
        "config": {k: v for k, v in env.items() if k.startswith("RESIN_")},
    }


def get_container_size(image_repo):
    log.info("Fetching container size for {}".format(image_repo))
    token = get_registry_token(image_repo)
    data = registry("v2/{}/manifests/latest".format(image_repo), token)
    headers = {"Range": "bytes=-4"}
    size = 0
    # the last 4 bytes of each gzipped layer are the layer size % 32
    for layer in data["fsLayers"]:
        endpoint = "v2/{}/blobs/{}".format(image_repo, layer["blobSum"])
        layer_size_bytes = registry(
            endpoint,
            token,
            headers,
            decode_json=False
        )
        layer_size = int.from_bytes(layer_size_bytes, byteorder="little")
        size += layer_size
        log.info("{} {}".format(endpoint, size))
    log.info("Container size: {}".format(human_size(size)))
    return size


def get_resin_os_version(image):
    with mount_context_manager(image, 2) as mountpoint:
        with open(os.path.join(mountpoint, "etc/os-release")) as f:
            for line in f:
                key, value = line.split("=", 1)
                if key == "VERSION":
                    return value.strip('"\n"')


def expand_image(image, additional_space):
    log.info("Expanding image size by {}".format(human_size(additional_space)))
    # Add zero bytes to image to be able to resize partitions
    with open(image, "a") as f:
        size = f.tell()
        f.truncate(size + additional_space)


def expand_partitions(image):
    """Resizes partitions 4 & 6 to the end of the image"""
    log.info(
        "Expanding extended partition 4 and logical partition 6 to the end of "
        "the disk image."
    )
    parted("-s", image, "resizepart", 4, "100%", "resizepart", 6, "100%")


def expand_ext4(image):
    """Resizes the ext4 filesystem in the 6th partition of the image"""
    with losetup_context_manager(image, 6) as loop_device:
        log.info("Using {}".format(loop_device))
        log.info("Resizing filesystem")
        try:
            status = e2fsck("-p", "-f", loop_device, _ok_code=[0, 1, 2])
            status = status.exit_code
            if status == 0:
                log.info("e2fsck: File system OK")
            else:
                log.warning("e2fsck: File system errors corrected")
        except ErrorReturnCode:
            log.error("e2fsck: File system errors could not be corrected")
            exit(1)
        resize2fs("-f", loop_device)


def expand_btrfs(mountpoint):
    btrfs("filesystem", "resize", "max", mountpoint)


def fix_rce_docker(mountpoint):
    """
    Removes the /rce folder if a /docker folder exists.
    Returns "<mountpoint>/docker" if this folder exists, "<mountpoint>/rce"
    otherwise.
    """
    _docker_dir = mountpoint + "/docker"
    _rce_dir = mountpoint + "/rce"
    if os.path.isdir(_docker_dir):
        if os.path.isdir(_rce_dir):
            rmtree(_rce_dir)
        return _docker_dir
    else:
        return _rce_dir


def start_docker_daemon(filesystem_type, mountpoint, socket):
    """Starts the docker daemon and waits for it to be ready."""
    docker_dir = fix_rce_docker(mountpoint)
    dockerd(s=filesystem_type, g=docker_dir, H="unix://" + socket, _bg=True)
    log.info("Waiting for Docker to start...")
    while os.path.isfile(socket):
        inotifywait("-t", 1, "-e", "create", os.path.dirname(socket))
    ok = False
    while not ok:
        output = docker("-H", "unix://" + socket, "version", _ok_code=[0, 1])
        ok = output.exit_code == 0
    log.info("Docker started")


@contextmanager
def docker_context_manager(filesystem_type, mountpoint, socket):
    start_docker_daemon(filesystem_type, mountpoint, socket)
    yield
    with open("/var/run/docker.pid", "r") as f:
        kill(f.read().strip())


def write_apps_json(data, output):
    """Writes data dict to output as json"""
    data = data.copy()
    # NOTE: This replaces the registry host in the `imageId` to stop the newer
    # supervisors from re-downloading the app on first boot
    data["imageId"] = "registry2.resin.io/" + data["imageRepo"]
    # Keep only the fields we need from APPS_JSON
    del data["imageRepo"]
    with open(output, "w") as f:
        json.dump([data], f, indent=4, sort_keys=True)


def replace_splash_image(disk_image):
    """
    Replaces the resin-logo.png used on boot splash to allow a more branded
    experience.
    """
    splash_image = "/img/resin-logo.png"
    if os.path.isfile(splash_image):
        log.info("Replacing splash image")
        with mount_context_manager(disk_image, 1) as mountpoint:
            copyfile(splash_image, mountpoint + "/splash/resin-logo.png")
    else:
        log.info("Leaving splash image alone")


def docker_pull(docker_sock, image_id):
    log.info("Pulling image...")
    docker("-H", "unix://" + docker_sock, "pull", image_id, _fg=True)
    log.info("Docker images loaded:")
    docker("-H", "unix://" + docker_sock, "images", "--all", _fg=True)


def resize_fs_copy_splash_image_and_pull(image, app_data):
    # Use ext4 for 2.0.0+ versions, btrfs otherwise
    version = get_resin_os_version(image)
    version_ge_2 = LooseVersion(version) >= LooseVersion("2.0.0")
    extra_options = ""
    if version_ge_2:
        # For ext4, we'll have to keep it unmounted to resize
        log.info("Expanding ext filesystem")
        expand_ext4(image)
    else:
        extra_options = "nospace_cache,rw"
    docker_sock = mkdtemp() + "/docker.sock"
    with mount_context_manager(image, 6, extra_options) as mountpoint:
        if version_ge_2:
            driver = "aufs"
        else:
            # For btrfs we need to mount the fs for resizing.
            log.info("Expanding btrfs filesystem")
            expand_btrfs(mountpoint)
            driver = "btrfs"
        with docker_context_manager(driver, mountpoint, docker_sock):
            write_apps_json(app_data, mountpoint + "/apps.json")
            docker_pull(docker_sock, app_data["imageId"])


def round_to_sector_size(size, sector_size=512):
    sectors = size / sector_size
    if not sectors.is_integer():
        sectors = floor(sectors) + 1
    return sectors * sector_size


def check():
    assert API_TOKEN or API_KEY, "API_TOKEN or API_KEY must be set"
    assert APP_ID, "APP_ID must be set"


def main():
    check()
    IMAGE = "/img/resin.img"
    log.info("Fetching application data")
    log.info("Using API host {}".format(API_HOST))
    log.info("Using Registry host {}".format(REGISTRY_HOST))
    app_data = get_app_data()
    replace_splash_image(IMAGE)
    repo = app_data["imageRepo"]
    container_size = get_container_size(repo)
    # Size will be increased by 110% of the container size
    additional_space = round_to_sector_size(ceil(container_size * 1.1))
    expand_image(IMAGE, additional_space)
    expand_partitions(IMAGE)
    resize_fs_copy_splash_image_and_pull(IMAGE, app_data)
    log.info("Done.")


if __name__ == "__main__":
    main()
