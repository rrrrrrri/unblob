import binascii
import io
import shutil
import struct
import tempfile
from pathlib import Path

from unblob.extractors import Command

from ...file_utils import StructParser, get_endian, iterate_file
from ...models import (
    Extractor,
    ExtractResult,
    File,
    HandlerDoc,
    HandlerType,
    HexString,
    Reference,
    StructHandler,
    ValidChunk,
)

CRAMFS_FLAG_FSID_VERSION_2 = 0x00000001
BIG_ENDIAN_MAGIC = 0x28_CD_3D_45
FSID_CRC_OFFSET = 32
FSID_CRC_SIZE = 4
CRAMFS_HEADER_STRUCT = "cramfs_header_t"
CRAMFS_C_DEFINITIONS = r"""
    typedef struct cramfs_header {
        uint32 magic;
        uint32 fs_size;
        uint32 flags;
        uint32 future;
        char signature[16];
        uint32 fsid_crc;
        uint32 fsid_edition;
        uint32 fsid_blocks;
        uint32 fsid_files;
        char name[16];
    } cramfs_header_t;
"""

_CRAMFS_HEADER_PARSER = StructParser(CRAMFS_C_DEFINITIONS)


def swap_int32(i):
    return struct.unpack("<I", struct.pack(">I", i))[0]


def _calculate_crc(file: File, start_offset: int, fs_size: int) -> int:
    file.seek(start_offset, io.SEEK_SET)
    header_bytes = bytearray(file.read(FSID_CRC_OFFSET + FSID_CRC_SIZE))
    header_bytes[FSID_CRC_OFFSET : FSID_CRC_OFFSET + FSID_CRC_SIZE] = (
        b"\x00\x00\x00\x00"
    )
    computed_crc = binascii.crc32(header_bytes)

    for chunk in iterate_file(
        file,
        start_offset + FSID_CRC_OFFSET + FSID_CRC_SIZE,
        fs_size - (FSID_CRC_OFFSET + FSID_CRC_SIZE),
    ):
        computed_crc = binascii.crc32(chunk, computed_crc)

    return computed_crc


class CramFSExtractor(Extractor):
    def __init__(self):
        self._command = Command("7z", "x", "-y", "{inpath}", "-o{outdir}")

    def extract(self, inpath: Path, outdir: Path) -> ExtractResult | None:
        with File.from_path(inpath) as file:
            endian = get_endian(file, BIG_ENDIAN_MAGIC)
            header = _CRAMFS_HEADER_PARSER.parse(CRAMFS_HEADER_STRUCT, file, endian)

            if not (header.flags & CRAMFS_FLAG_FSID_VERSION_2):
                return self._command.extract(inpath, outdir)

            computed_crc = _calculate_crc(file, 0, header.fs_size)

        if header.fsid_crc != swap_int32(computed_crc):
            return self._command.extract(inpath, outdir)

        # Some Cisco images store the CRC bytes in the opposite order. 7-Zip
        # rejects those images, so normalize only a temporary extraction input
        # and leave both the source firmware and the carved CramFS untouched.
        normalized_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=".unblob-cramfs-",
                suffix=".cramfs",
                dir=outdir.parent,
                delete=False,
            ) as normalized_file:
                normalized_path = Path(normalized_file.name)
                with inpath.open("rb") as source:
                    shutil.copyfileobj(source, normalized_file)
                normalized_file.seek(FSID_CRC_OFFSET, io.SEEK_SET)
                normalized_file.write(struct.pack(f"{endian.value}I", computed_crc))

            return self._command.extract(normalized_path, outdir)
        finally:
            if normalized_path is not None:
                normalized_path.unlink(missing_ok=True)

    def get_dependencies(self) -> list[str]:
        return self._command.get_dependencies()


class CramFSHandler(StructHandler):
    NAME = "cramfs"

    PATTERNS = [
        HexString("28 CD 3D 45"),  # big endian
        HexString("45 3D CD 28"),  # little endian
    ]

    C_DEFINITIONS = CRAMFS_C_DEFINITIONS
    HEADER_STRUCT = CRAMFS_HEADER_STRUCT

    EXTRACTOR = CramFSExtractor()

    DOC = HandlerDoc(
        name="CramFS",
        description="CramFS is a lightweight, read-only file system format designed for simplicity and efficiency in embedded systems. It uses zlib compression for file data and stores metadata in a compact, contiguous structure.",
        handler_type=HandlerType.FILESYSTEM,
        vendor=None,
        references=[
            Reference(
                title="CramFS Documentation",
                url="https://web.archive.org/web/20160304053532/http://sourceforge.net/projects/cramfs/",
            ),
            Reference(
                title="CramFS Wikipedia",
                url="https://en.wikipedia.org/wiki/Cramfs",
            ),
        ],
        limitations=[],
    )

    def calculate_chunk(self, file: File, start_offset: int) -> ValidChunk | None:
        endian = get_endian(file, BIG_ENDIAN_MAGIC)
        header = self.parse_header(file, endian)
        valid_signature = header.signature == b"Compressed ROMFS"

        if valid_signature and self._is_crc_valid(file, start_offset, header):
            return ValidChunk(
                start_offset=start_offset,
                end_offset=start_offset + header.fs_size,
            )
        return None

    def _is_crc_valid(
        self,
        file: File,
        start_offset: int,
        header,
    ) -> bool:
        # old cramfs format do not support crc
        if not (header.flags & CRAMFS_FLAG_FSID_VERSION_2):
            return True

        computed_crc = _calculate_crc(file, start_offset, header.fs_size)

        # some vendors like their CRC's swapped, don't ask why
        return header.fsid_crc == computed_crc or header.fsid_crc == swap_int32(
            computed_crc
        )
