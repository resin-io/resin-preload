#!/usr/bin/python3 -u
import json
import os
import re
import sys
import traceback

from contextlib import contextmanager
from functools import partial
from logging import getLogger, INFO, StreamHandler
from math import ceil, floor
from re import match, search
from retry.api import retry_call
from sh import (
    btrfs,
    dd,
    df,
    docker as _docker,
    dockerd,
    file,
    fsck,
    losetup,
    mount,
    parted,
    resize2fs,
    sfdisk,
    sgdisk,
    umount,
    update_ca_certificates,
    ErrorReturnCode,
)
from shutil import copyfile, rmtree
from tempfile import mkdtemp, NamedTemporaryFile

os.environ["LANG"] = "C"

IMAGE = "/img/balena.img"

# In bytes:
SECTOR_SIZE = 512
MBR_SIZE = 512
GPT_SIZE = SECTOR_SIZE * 34
MBR_BOOTSTRAP_CODE_SIZE = 446

SPLASH_IMAGE_FROM = "/img/balena-logo.png"

CONFIG_PARTITIONS = [
    "resin-boot",  # resinOS 1.26+
    "resin-conf",  # resinOS 1.8
    "flash-conf",  # resinOS 1.8 flash
    "flash-boot",  # flasher images
]

SUPERVISOR_REPOSITORY_RE = "^resin(playground)?/[a-z0-9]+-supervisor$"

# 'sh' module '_truncate_exc' option:
# http://amoffat.github.io/sh/sections/special_arguments.html#truncate-exc
# https://github.com/amoffat/sh/blob/1.12.14/sh.py#L1151
SH_OPTS = {"_truncate_exc": False}

DOCKER_HOST = "tcp://0.0.0.0:{}".format(os.environ.get("DOCKER_PORT") or 8000)
docker = partial(_docker, "--host", DOCKER_HOST, **SH_OPTS)

log = getLogger(__name__)
log.setLevel(INFO)
log.addHandler(StreamHandler())

PARTITIONS_CACHE = {}


class RetryCounter:
    """Counter and logger for the number of times that a function is retried.
    Usage:
        retry_counter = RetryCounter()
        hint = "Try reducing the system entropy"
        wrapped_func, wrap_key = retry_counter.wrap(my_func, hint, *args)
        retry_call(wrapped_func, fargs=args, ...)
        retry_counter.clear(wrap_key)
    """

    def __init__(self):
        self.counter = {}

    @staticmethod
    def key(func_name, *args, **kwargs):
        return " ".join(
            str(e) for e in (func_name,) + args + tuple(kwargs.values())
        )

    def clear(self, key):
        del self.counter[key]

    def inc(self, key):
        self.counter[key] = self.counter.setdefault(key, 0) + 1
        return self.counter[key]

    def wrap(self, func, hint, *args, **kwargs):
        """Return a function that wraps the given func, counting its usage"""
        key = self.key(func.__name__, *args, **kwargs)

        def wrapped(*args, **kwargs):
            count = self.inc(key)
            if count > 1:
                log.info(
                    "\nRetrying (count={}) {}\n{}".format(
                        count,
                        key,
                        hint,
                    ),
                )
            return func(*args, **kwargs)

        return (wrapped, key)


retry_counter = RetryCounter()


def get_partitions(image):
    return {p.label: p for p in PartitionTable(image).partitions if p.label}


def prepare_global_partitions():
    partitions = os.environ.get("PARTITIONS")
    if partitions is not None:
        partitions = json.loads(partitions)
        result = {}
        for label, data in partitions.items():
            result[label] = FilePartition(data["image"], label)
        return result
    return get_partitions(IMAGE)


@contextmanager
def losetup_context_manager(image, offset=None, size=None):
    args = ["-f", "--show"]
    if offset is not None:
        args.extend(["--offset", offset])
    if size is not None:
        args.extend(["--sizelimit", size])
    args.append(image)
    hint = """\
Hint: If using a Virtual Machine, consider increasing the number of processors.
If using Docker Desktop for Windows or macOS, it may require restarting."""
    lo_wrap, lo_wrap_key = retry_counter.wrap(losetup, hint, *args, **SH_OPTS)
    # In the case of slow hardware the kernel might be in the middle of
    # tearing down internal structure
    device = retry_call(
        lo_wrap,
        fargs=args,
        tries=10,
        delay=3,
        max_delay=30,
        backoff=2
    ).stdout.decode("utf8").strip()
    retry_counter.clear(lo_wrap_key)
    yield device
    losetup("-d", device, **SH_OPTS)


@contextmanager
def device_mount_context_manager(device):
    mountpoint = mkdtemp()
    mount(device, mountpoint, **SH_OPTS)
    yield mountpoint
    umount(mountpoint, **SH_OPTS)
    os.rmdir(mountpoint)


@contextmanager
def mount_context_manager(image, offset=None, size=None):
    with losetup_context_manager(image, offset, size) as device:
        with device_mount_context_manager(device) as mountpoint:
            yield mountpoint


class FilePartition(object):
    def __init__(self, image, label):
        self.image = image
        self.label = label

    def losetup_context_manager(self):
        return losetup_context_manager(self.image)

    def mount_context_manager(self):
        return mount_context_manager(self.image)

    def resize(self, additional_bytes):
        if additional_bytes > 0:
            expand_file(self.image, additional_bytes)
            expand_filesystem(self)

    def str(self):
        return self.image

    def free_space(self):
        with self.losetup_context_manager() as device:
            fs = get_filesystem(device)
            with mount_context_manager(device) as mountpoint:
                if fs == 'btrfs':
                    out = btrfs("fi", "usage", "--raw", mountpoint, **SH_OPTS)
                    for line in out:
                        line = line.strip()
                        if line.startswith("Free (estimated):"):
                            return int(line[line.rfind(" ") + 1:-1])
                else:
                    output = df("-B1", "--output=avail", mountpoint, **SH_OPTS)
                    return int(output.split("\n")[1].strip())


class Partition(FilePartition):
    def __init__(
        self,
        partition_table,
        number,
        node=None,
        start=None,
        size=None,
        type=None,
        uuid=None,
        name=None,
        bootable=False,
    ):
        self.partition_table = partition_table
        self.number = number
        self.parent = None
        self.node = node
        self.start = start
        self.size = size
        self.type = type
        self.uuid = uuid
        self.name = name
        self.bootable = bootable
        # label, not part of the sfdisk script
        self.label = self._get_label()

    def _get_label(self):
        with self.losetup_context_manager() as device:
            out = file("-s", device, **SH_OPTS).stdout.decode("utf8").strip()
            # "label:" is for fat partitions,
            # "volume name" is for ext partitions
            # "BTRFS Filesystem label" is for btrfs partitions
            match = search(
                '(BTRFS Filesystem label|label:|volume name) "(.*)"',
                out,
            )
            if match is not None:
                return match.groups()[1].strip()

    def set_parent(self, parent):
        # For logical partitions on MBR disks we store the parent extended
        # partition
        assert self.partition_table.label == "dos"
        self.parent = parent

    @property
    def image(self):
        return self.partition_table.image

    @property
    def end(self):
        # last sector (included)
        return self.start + self.size - 1

    @property
    def start_bytes(self):
        return self.start * SECTOR_SIZE

    @property
    def size_bytes(self):
        return self.size * SECTOR_SIZE

    @property
    def end_bytes(self):
        # last byte (included)
        return self.start_bytes + self.size_bytes - 1

    def is_included_in(self, other):
        return (
            other.start <= self.start <= other.end and
            other.start <= self.end <= other.end
        )

    def is_extended(self):
        return self.partition_table.label == "dos" and self.type == "f"

    def is_last(self):
        # returns True if this partition is the last on the disk
        return self == self.partition_table.get_partitions_in_disk_order()[-1]

    def get_sfdisk_line(self):
        result = "{} : start={}, size={}, type={}".format(
            self.node,
            self.start,
            self.size,
            self.type
        )
        if self.uuid is not None:
            result += ", uuid={}".format(self.uuid)
        if self.name is not None:
            result += ', name="{}"'.format(self.name)
        if self.bootable:
            result += ", bootable"
        return result

    def losetup_context_manager(self):
        return losetup_context_manager(
            self.image,
            self.start_bytes,
            self.size_bytes,
        )

    def mount_context_manager(self):
        return mount_context_manager(
            self.image,
            self.start_bytes,
            self.size_bytes,
        )

    def str(self):
        return "partition n°{} of {}".format(self.number, self.image)

    def _resize_last_partition_of_disk_image(self, additional_bytes):
        # This is the simple case: expand the partition and its parent extended
        # partition if it is a logical one.
        additional_sectors = additional_bytes // SECTOR_SIZE
        # Expand image size
        expand_file(self.image, additional_bytes)
        if self.partition_table.label == "gpt":
            # Move backup GPT data structures to the end of the disk.
            # This is required because we resized the image.
            sgdisk("-e", self.image, **SH_OPTS)
        parted_args = [self.image]
        if self.parent is not None:
            log.info("Expanding extended {}".format(self.parent.str()))
            # Resize the extended partition
            parted_args.extend(["resizepart", self.parent.number, "100%"])
            self.parent.size += additional_sectors
        # Resize the partition itself
        log.info(
            "Expanding{} {}".format(
                " logical" if self.parent is not None else "",
                self.str(),
            )
        )
        parted_args.extend(["resizepart", self.number, "100%"])
        parted(*parted_args, _in="fix\n", **SH_OPTS)
        self.size += additional_sectors

    def _resize_partition_on_disk_image(self, additional_bytes):
        # This function expects the partitions to be in disk order: it will
        # fail if there are primary partitions after an extended one containing
        # logical partitions.
        # Resizing logical partitions that are not the last on the disk is not
        # implemented
        assert self.parent is None
        partition_table = self.partition_table
        image = self.image
        # Create a new temporary file of the correct size
        tmp = NamedTemporaryFile(dir=os.path.dirname(image), delete=False)
        tmp.truncate(file_size(image) + additional_bytes)
        tmp.close()
        # Update the partition table
        additional_sectors = additional_bytes // SECTOR_SIZE
        # resize the partition
        self.size += additional_sectors
        # move the partitions after
        for part in partition_table.partitions[self.number:]:
            part.start += additional_sectors
        # update last lba
        if partition_table.lastlba is not None:
            partition_table.lastlba += additional_sectors
        sfdisk(tmp.name, _in=partition_table.get_sfdisk_script(), **SH_OPTS)
        # Now we copy the data from the image to the temporary file
        copy = partial(
            ddd,
            _if=image,
            of=tmp.name,
            bs=1024 ** 2,  # one MiB
            conv="notrunc",
            iflag="count_bytes,skip_bytes",  # count and skip in bytes
            oflag="seek_bytes",  # seek in bytes
        )
        # Preserve GRUB
        copy(count=MBR_BOOTSTRAP_CODE_SIZE)
        # Copy across any data that's located between the MBR and the first
        # partition (some devices rely on the bootloader being there, like the
        # Variscite DART-6UL)
        if self.partition_table.label == "dos":
            copy(
                skip=MBR_SIZE,
                seek=MBR_SIZE,
                count=partition_table.partitions[0].start_bytes - MBR_SIZE,
            )
        elif self.partition_table.label == "gpt":
            copy(
                skip=GPT_SIZE,
                seek=GPT_SIZE,
                count=partition_table.partitions[0].start_bytes - GPT_SIZE,
            )
        # TODO: if we copy an extended partition, there is no need to copy its
        # logical partitions.
        # Copy partitions before and the partition itself
        for part in partition_table.partitions[:self.number]:
            # No need to copy extended partitions, we'll copy their logical
            # partitions
            if not part.is_extended():
                copy(
                    skip=part.start_bytes,
                    seek=part.start_bytes,
                    count=part.size_bytes,
                )
        # Copy partitions after.
        for part in partition_table.partitions[self.number:]:
            if not part.is_extended():
                copy(
                    skip=part.start_bytes,
                    seek=part.start_bytes + additional_bytes,
                    count=part.size_bytes,
                )
        # Replace the original image contents.
        ddd(_if=tmp.name, of=image, bs=1024 ** 2)

    def resize(self, additional_bytes):
        if additional_bytes > 0:
            # Is it the last partition on the disk?
            if self.is_last():
                self._resize_last_partition_of_disk_image(additional_bytes)
            else:
                self._resize_partition_on_disk_image(additional_bytes)
            expand_filesystem(self)


class PartitionTable(object):
    def __init__(self, image):
        self.image = image
        data = json.loads(
            sfdisk("--dump", "--json", image, **SH_OPTS).stdout.decode("utf8")
        )["partitiontable"]
        self.label = data["label"]
        assert self.label in ("dos", "gpt")
        self.id = data["id"]
        self.device = data["device"]
        self.unit = data["unit"]
        self.firstlba = data.get("firstlba")
        self.lastlba = data.get("lastlba")
        self.partitions = []
        extended_partition = None
        for number, partition_data in enumerate(data["partitions"], 1):
            part = Partition(self, number, **partition_data)
            if part.is_extended():
                extended_partition = part
            if extended_partition and part.is_included_in(extended_partition):
                part.set_parent(extended_partition)
            self.partitions.append(part)

    def get_partitions_in_disk_order(self):
        # Returns the partitions in the same order that they are on the disk
        # This excludes extended partitions.
        partitions = (p for p in self.partitions if not p.is_extended())
        return sorted(partitions, key=lambda p: p.start)

    def get_sfdisk_script(self):
        result = (
            "label: {}\n"
            "label-id: {}\n"
            "device: {}\n"
            "unit: {}\n"
        ).format(self.label, self.id, self.device, self.unit)
        if self.firstlba is not None:
            result += "first-lba: {}\n".format(self.firstlba)
        if self.lastlba is not None:
            result += "last-lba: {}\n".format(self.lastlba)
        result += "\n"
        result += "\n".join(p.get_sfdisk_line() for p in self.partitions)
        return result


def get_filesystem(device):
    result = fsck("-N", device, **SH_OPTS)
    line = result.stdout.decode("utf8").strip().split("\n")[1]
    return line.rsplit(" ", 2)[-2].split(".")[1]


def expand_filesystem(partition):
    with partition.losetup_context_manager() as device:
        # Detects the partition filesystem (ext{2,3,4} or btrfs) and uses the
        # appropriate tool to expand the filesystem to all the available space.
        fs = get_filesystem(device)
        log.info(
            "Resizing {} filesystem of {} using {}".format(
                fs,
                partition.str(),
                device,
            )
        )
        if fs.startswith("ext"):
            try:
                kwargs = {"_ok_code": [0, 1, 2], **SH_OPTS}
                status = fsck("-p", "-f", device, **kwargs)
                if status.exit_code == 0:
                    log.info("File system OK")
                else:
                    log.warning("File system errors corrected")
            except ErrorReturnCode:
                raise Exception("File system errors could not be corrected")
            resize2fs("-f", device, **SH_OPTS)
        elif fs == "btrfs":
            # For btrfs we need to mount the fs for resizing.
            with mount_context_manager(device) as mountpoint:
                btrfs("filesystem", "resize", "max", mountpoint, **SH_OPTS)


def expand_file(path, additional_bytes):
    with open(path, "a") as f:
        size = f.tell()
        f.truncate(size + additional_bytes)


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


def start_docker_daemon(storage_driver, docker_dir):
    """Starts the docker daemon and waits for it to be ready."""
    running_dockerd = dockerd(
        storage_driver=storage_driver,
        data_root=docker_dir,
        host=DOCKER_HOST,
        _bg=True,
        **SH_OPTS,
    )
    log.info("Waiting for Docker to start...")
    ok = False
    while not ok:
        # dockerd should not exit, if it does, we'll throw an exception.
        if running_dockerd.process.exit_code is not None:
            # There is no reason for dockerd to exit with a 0 status now.
            assert running_dockerd.process.exit_code != 0
            # This will raise an sh.ErrorReturnCode_X exception.
            running_dockerd.wait()
        # Check that we can connect to dockerd.
        output = docker("version", _ok_code=[0, 1])
        ok = output.exit_code == 0
    log.info("Docker started")
    return running_dockerd


def read_file(name):
    with open(name, "rb") as f:
        return f.read()


def write_file(name, content):
    with open(name, "wb") as f:
        f.write(content)


@contextmanager
def docker_context_manager(storage_driver, mountpoint):
    docker_dir = fix_rce_docker(mountpoint)
    # If we don't remove <part6>/<docker|rce>/network/files/local-kv.db and the
    # preload container was started with bridged networking, the following
    # dockerd is not reachable from the host.
    local_kv_db_path = "{}/network/files/local-kv.db".format(docker_dir)
    kv_file_existed = (
        os.path.exists(local_kv_db_path) and os.path.isfile(local_kv_db_path)
    )
    if kv_file_existed:
        local_kv_db_content = read_file(local_kv_db_path)
        os.remove(local_kv_db_path)
    running_dockerd = start_docker_daemon(storage_driver, docker_dir)
    yield
    running_dockerd.terminate()
    running_dockerd.wait()
    if kv_file_existed:
        write_file(local_kv_db_path, local_kv_db_content)


def write_resin_device_pinning(app_data, output):
    """Create resin-device-pinnnig.json to hold pinning information"""
    if type(app_data) != dict:
        # app_data is a list when the supervisor version is < 7.0.0,
        # pinning is not suported on these.
        return
    if not app_data.get("pinDevice", False):
        return
    apps = app_data.get("apps", {})
    if len(apps) != 1:
        raise Exception("Malformed apps.json")

    with open(output, "w") as f:
        f.write(
            "RELEASE_ID={}".format(
                next(iter(apps.values())).get('releaseId'),
            ),
        )


def write_apps_json(data, output):
    """Updates or creates preloaded state"""
    if os.path.isfile(output):
        with open(output, "r") as f:
            current_state = json.load(f)
        for uuid in data.keys():
            if len(uuid) != len(current_state.keys()[0]):
                raise Exception("Current and target preloaded states not "
                                "compatible")
            if uuid not in current_state['apps']:
                current_state['apps'][uuid] = data[uuid]
        data = current_state
    with open(output, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def replace_splash_image(splash_image_path, image=None):
    """
    Replaces the balena logo used on boot splash to allow a more branded
    experience.
    """
    if os.path.isfile(SPLASH_IMAGE_FROM):
        boot = (
            get_partition("flash-boot") or
            get_partition("resin-boot", image)
        )
        with boot.mount_context_manager() as mpoint:
            path = mpoint + splash_image_path
            if os.path.isdir(os.path.dirname(path)):
                log.info("Replacing splash image")
                copyfile(SPLASH_IMAGE_FROM, path)
            else:
                log.info(
                    "No splash folder on the boot partition, the splash image "
                    "won't be inserted."
                )
    else:
        log.info("Leaving splash image alone")


def start_dockerd_and_wait_for_stdin(app_data, image=None):
    rootA_file_contents = get_rootA_file_contents(image)
    driver = get_docker_storage_driver(
        rootA_file_contents.get("docker_service", ""),
    )
    part = get_partition("resin-data", image)
    with part.mount_context_manager() as mpoint:
        write_apps_json(app_data, mpoint + "/apps.json")
        with docker_context_manager(driver, mpoint):
            # Signal that Docker is ready.
            print(json.dumps({"statusCode": 0}))
            sys.stdout.flush()
            # Wait for the js to finish its job.
            input()


def round_to_sector_size(size, sector_size=SECTOR_SIZE):
    sectors = size / sector_size
    if not sectors.is_integer():
        sectors = floor(sectors) + 1
    return int(sectors * sector_size)


def file_size(path):
    with open(path, "a") as f:
        return f.tell()


def ddd(**kwargs):
    # dd helper
    return dd(
        *("{}={}".format(k.lstrip("_"), v) for k, v in kwargs.items()),
        **SH_OPTS,
    )


def get_json(partition_name, path, image=None):
    part = get_partition(partition_name, image)
    if part:
        with part.mount_context_manager() as mountpoint:
            try:
                with open(os.path.join(mountpoint, path)) as f:
                    return json.load(f)
            except FileNotFoundError:
                pass


def get_device_type(image=None):
    result = get_json("resin-boot", "device-type.json", image=image)
    if result is None:
        result = get_json("flash-boot", "device-type.json", image=image)
    return result


def get_config(image=None):
    for partition_name in CONFIG_PARTITIONS:
        data = get_json(partition_name, "config.json", image=image)
        if data is not None:
            return data


def preload(additional_bytes, app_data, splash_image_path, image=None):
    replace_splash_image(splash_image_path, image)
    part = get_partition("resin-data", image)
    part.resize(additional_bytes)
    start_dockerd_and_wait_for_stdin(app_data, image)


def get_inner_image_path(root_mountpoint):
    opt = os.path.join(root_mountpoint, "opt")
    device_type = get_device_type()
    if device_type is not None:
        deploy_artifact = device_type["yocto"]["deployArtifact"]
        return os.path.join(opt, deploy_artifact.replace("flasher-", "", 1))
    else:
        # Old images don't have a device type file (resinOS 1.8),
        # return the first file in "/opt".
        return next(
            filter(
                os.path.isfile,
                map(partial(os.path.join, opt), os.listdir(opt))
            )
        )


def _get_balena_os_version(etc_issue_contents):
    """
    Return a balenaOS version string such as "2.53.0", given the contents
    of the "/etc/issue" file in the etc_issue_contents argument.
    """
    m = match('balenaOS (.+?) ', etc_issue_contents)
    return m[1] if m is not None else ""


def _get_images_and_supervisor_version(image=None):
    rootA_file_contents = get_rootA_file_contents(image)
    driver = get_docker_storage_driver(
        rootA_file_contents.get("docker_service", ""),
    )
    part = get_partition("resin-data", image)
    with part.mount_context_manager() as mountpoint:
        with docker_context_manager(driver, mountpoint):
            output = docker(
                "images",
                "--all",
                "--format",
                "{{.Repository}} {{.Tag}}"
            )
            images = set()
            supervisor_version = None
            for line in output:
                repository, version = line.strip().split()
                if match(SUPERVISOR_REPOSITORY_RE, repository):
                    if version != "latest":
                        version_search = re.search(
                            r"^v?(?P<semver>\d+\.\d+\.\d+).*",
                            version,
                        )
                        if version_search:
                            supervisor_version = version_search.group('semver')
                        else:
                            raise Exception(
                                "Could not extract supervisor version.",
                            )
                else:
                    images.add(repository)
            return (
                list(images),
                supervisor_version,
                _get_balena_os_version(
                    rootA_file_contents.get("/etc/issue", ""),
                ),
            )


def get_images_and_supervisor_version():
    flasher_root = get_partition("flash-rootA")
    if flasher_root:
        with flasher_root.mount_context_manager() as mountpoint:
            inner_image_path = get_inner_image_path(mountpoint)
            return _get_images_and_supervisor_version(inner_image_path)
    return _get_images_and_supervisor_version()


def free_space():
    flasher_root = get_partition("flash-rootA")
    if flasher_root:
        with flasher_root.mount_context_manager() as mountpoint:
            inner_image_path = get_inner_image_path(mountpoint)
            return get_partition("resin-data", inner_image_path).free_space()
    return get_partition("resin-data").free_space()


def is_non_empty_folder(folder):
    # True if the folder has at least one file not starting with a dot.
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return False
    return any(f for f in os.listdir(folder) if not f.startswith("."))


def find_non_empty_folder_in_path(path, child_dir=""):
    # If child_dir is not given, returns any non empty folder like <path>/...;
    # else, returns any non empty folder like <path>/.../<child_dir>
    # where ... can be any subfodler of <path>.
    if os.path.exists(path) and os.path.isdir(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder, child_dir)
            if is_non_empty_folder(folder_path):
                return folder_path


def find_docker_aufs_root(mountpoint):
    # We're looking for a /<docker|balena>/aufs/diff/<xxxxxxxxxxxxx>/ folder
    # with some files not starting with a '.'
    for name in ("docker", "balena"):
        path = os.path.join(mountpoint, name, "aufs", "diff")
        if os.path.isdir(path):
            return find_non_empty_folder_in_path(path)


def find_docker_overlay2_root(mountpoint):
    # We're looking for a /<docker|balena>/overlay2/<xxxxxxxxxxxxx>/diff
    # folder with some files not starting with a '.'
    for name in ("docker", "balena"):
        path = os.path.join(mountpoint, name, "overlay2")
        if os.path.isdir(path):
            return find_non_empty_folder_in_path(path, "diff")


def get_docker_service_file_path(folder):
    for name in ("docker", "balena"):
        fpath = os.path.join(
            folder,
            "lib",
            "systemd",
            "system",
            name + ".service",
        )
        if os.path.exists(fpath):
            return fpath


def get_rootA_file_contents(image=None):
    file_contents = {
        "docker_service": "",
        "/etc/issue": "",
    }
    part = get_partition("resin-rootA", image)
    with part.mount_context_manager() as mountpoint:
        docker_root = find_docker_aufs_root(mountpoint)
        if docker_root is None:
            docker_root = find_docker_overlay2_root(mountpoint)
        root_folder = docker_root if docker_root is not None else mountpoint
        docker_service_path = get_docker_service_file_path(root_folder) or ""
        etc_issue_path = os.path.join(root_folder, "etc", "issue")
        with open(docker_service_path) as f:
            file_contents["docker_service"] = f.read()
        try:
            with open(etc_issue_path) as f:
                file_contents["/etc/issue"] = f.read()
        except OSError:
            # If very old or custom images don't have an '/etc/issue' file,
            # simply return an empty string for it.
            pass

    return file_contents


def find_one_of(lst, *args):
    for elem in args:
        index = lst.index(elem)
        if index != -1:
            return index
    return -1


def get_docker_storage_driver(docker_service_file_contents):
    for line in docker_service_file_contents.strip().split("\n"):
        if line.startswith("ExecStart="):
            words = line.split()
            position = find_one_of(words, "-s", "--storage-driver")
            if position != -1 and position < len(words) - 1:
                return words[position + 1]
    assert False, "Docker storage driver could not be found"


def main_preload(app_data, additional_bytes, splash_image_path):
    additional_bytes = round_to_sector_size(ceil(additional_bytes))
    flasher_root = get_partition("flash-rootA")
    if flasher_root:
        flasher_root.resize(additional_bytes)
        with flasher_root.mount_context_manager() as mountpoint:
            write_resin_device_pinning(
                app_data,
                mountpoint + "/etc/resin-device-pinning.conf"
            )
            inner_image_path = get_inner_image_path(mountpoint)
            log.info(
                "This is a flasher image, preloading to /{} on {}".format(
                    inner_image_path.split("/", 2)[2],
                    flasher_root.str(),
                )
            )
            preload(
                additional_bytes,
                app_data,
                splash_image_path,
                inner_image_path,
            )
    else:
        preload(additional_bytes, app_data, splash_image_path)


def get_image_info():
    images, supervisor_version, balena_os_version = (
        get_images_and_supervisor_version()
    )
    return {
        "preloaded_builds": images,
        "supervisor_version": supervisor_version,
        "free_space": free_space(),
        "config": get_config(),
        # balena_os_version will be "" if "balenaOS" not found in /etc/issue
        "balena_os_version": balena_os_version,
    }


PARTITIONS_CACHE[None] = prepare_global_partitions()


def get_partition(name, image=None):
    partitions = PARTITIONS_CACHE.get(image)
    if partitions is None:
        partitions = get_partitions(image)
        PARTITIONS_CACHE[image] = partitions
    # In resinOS 1.8 the root partition is named "resin-root"
    if name == "resin-rootA":
        names = ["resin-rootA", "resin-root"]
    elif name == "flash-rootA":
        names = ["flash-rootA", "flash-root"]
    else:
        names = [name]
    for name in names:
        part = partitions.get(name)
        if part is not None:
            return part


methods = {
    "get_image_info": get_image_info,
    "preload": main_preload,
}


if __name__ == "__main__":
    update_ca_certificates(**SH_OPTS)
    for line in sys.stdin:
        try:
            data = json.loads(line)
            method = methods[data["command"]]
            result = method(**data.get("parameters", {}))
            response = {
                "result": result,
                "statusCode": 0,
            }
        except BaseException:
            response = {
                "error": traceback.format_exc(),
                "statusCode": 1,
            }
        print(json.dumps(response))
        sys.stdout.flush()
        if response["statusCode"]:
            break
