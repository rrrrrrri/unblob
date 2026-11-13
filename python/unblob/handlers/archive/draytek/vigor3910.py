from __future__ import annotations

import io
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from unblob.file_utils import (
    Endian,
    File,
    FileSystem,
    InvalidInputFormat,
    StructParser,
    iterate_file,
)
from unblob.models import (
    Extractor,
    ExtractResult,
    HandlerDoc,
    HandlerType,
    HexString,
    Reference,
    StructHandler,
    ValidChunk,
)

MODEL_HEADER_OFFSET = 0x40
DESCRIPTOR_LENGTH = 0x40
NONCE_LENGTH = 12
ARM64_IMAGE_HEADER_LENGTH = 64
ARM64_IMAGE_MAGIC_OFFSET = 56
ARM64_IMAGE_MAGIC = b"ARM\x64"
MAX_ENTRY_NAME_LENGTH = 0xFF
UINT32_MAX = (1 << 32) - 1

MODEL = b"v3910"
NONCE_ENTRY_NAME = "nonce"
IMAGE_ENTRY_NAME = "enc_Image"
ENCRYPTED_ENTRY_PREFIX = "enc_"

# Recovered from the Vigor 3910 firmware update implementation.
# https://wzt.ac.cn/2024/02/19/vigor_3910/
CHACHA20_KEY = b"0DraytekKd5Eason3DraytekKd5Eason"

SAFE_ENTRY_NAME = re.compile(rb"[A-Za-z0-9._-]+")

VIGOR3910_C_DEFINITIONS = r"""
    typedef struct vigor3910_header {
        uint8  opaque[64];
        uint32 model_length;
    } vigor3910_header_t;

    typedef struct vigor3910_entry_descriptor {
        uint32 flags;
        uint32 entry_length;
        uint32 checksum;
        uint8  reserved[52];
    } vigor3910_entry_descriptor_t;

    typedef struct vigor3910_name_header {
        uint32 name_length;
    } vigor3910_name_header_t;

    typedef struct vigor3910_data_header {
        uint32 data_length;
    } vigor3910_data_header_t;
"""


def _read_exact(file: File, length: int) -> bytes:
    data = file.read(length)
    if len(data) != length:
        raise EOFError
    return data


def _read_at(file: File, offset: int, length: int) -> bytes:
    file.seek(offset, io.SEEK_SET)
    return _read_exact(file, length)


def _new_chacha20_decryptor(
    key: bytes,
    nonce: bytes,
    counter: int = 0,
):
    if len(nonce) != NONCE_LENGTH:
        raise InvalidInputFormat("Invalid Vigor 3910 ChaCha20 nonce length")
    if not 0 <= counter <= UINT32_MAX:
        raise InvalidInputFormat("Invalid Vigor 3910 ChaCha20 counter")

    # cryptography exposes the original ChaCha20 64-bit counter/64-bit nonce
    # layout. Packing the IETF counter followed by the 96-bit nonce produces
    # the same four state words while the 32-bit counter does not wrap.
    full_nonce = struct.pack("<I", counter) + nonce
    return Cipher(algorithms.ChaCha20(key, full_nonce), mode=None).decryptor()


@dataclass(frozen=True)
class Vigor3910Entry:
    name: str
    data_offset: int
    data_length: int
    end_offset: int


@dataclass(frozen=True)
class Vigor3910Layout:
    start_offset: int
    end_offset: int
    nonce: bytes
    entries: tuple[Vigor3910Entry, ...]


class Vigor3910Parser:
    _struct_parser = StructParser(VIGOR3910_C_DEFINITIONS)

    @classmethod
    def _parse_descriptor(cls, source: File | bytes):
        descriptor = cls._struct_parser.parse(
            "vigor3910_entry_descriptor_t",
            source,
            Endian.LITTLE,
        )
        if int(descriptor.flags) != 0 or any(bytes(descriptor.reserved)):
            raise InvalidInputFormat("Invalid Vigor 3910 entry descriptor")
        if int(descriptor.entry_length) < DESCRIPTOR_LENGTH + 9:
            raise InvalidInputFormat("Invalid Vigor 3910 entry length")
        return descriptor

    @classmethod
    def _parse_entry(cls, file: File, descriptor) -> Vigor3910Entry:
        name_header = cls._struct_parser.parse(
            "vigor3910_name_header_t",
            file,
            Endian.LITTLE,
        )
        name_length = int(name_header.name_length)
        if not 0 < name_length <= MAX_ENTRY_NAME_LENGTH:
            raise InvalidInputFormat("Invalid Vigor 3910 entry name length")

        raw_name = _read_exact(file, name_length)
        if SAFE_ENTRY_NAME.fullmatch(raw_name) is None:
            raise InvalidInputFormat("Unsafe Vigor 3910 entry name")
        name = raw_name.decode("ascii")

        data_header = cls._struct_parser.parse(
            "vigor3910_data_header_t",
            file,
            Endian.LITTLE,
        )
        data_length = int(data_header.data_length)
        expected_length = (
            DESCRIPTOR_LENGTH
            + len(name_header)
            + name_length
            + len(data_header)
            + data_length
        )
        if int(descriptor.entry_length) != expected_length:
            raise InvalidInputFormat("Vigor 3910 entry lengths do not match")

        data_offset = file.tell()
        end_offset = data_offset + data_length
        file.seek(end_offset, io.SEEK_SET)
        return Vigor3910Entry(name, data_offset, data_length, end_offset)

    @classmethod
    def _probe_entry(cls, file: File, descriptor_offset: int) -> Vigor3910Entry | None:
        file.seek(descriptor_offset, io.SEEK_SET)
        descriptor_bytes = file.read(DESCRIPTOR_LENGTH)
        if descriptor_bytes == b"":
            return None
        if len(descriptor_bytes) != DESCRIPTOR_LENGTH:
            if descriptor_bytes.startswith(bytes(4)):
                raise EOFError
            return None

        try:
            descriptor = cls._parse_descriptor(descriptor_bytes)
        except InvalidInputFormat:
            return None
        return cls._parse_entry(file, descriptor)

    @classmethod
    def parse(cls, file: File, start_offset: int) -> Vigor3910Layout:
        file.seek(start_offset, io.SEEK_SET)
        header = cls._struct_parser.parse(
            "vigor3910_header_t",
            file,
            Endian.LITTLE,
        )
        model_length = int(header.model_length)
        if model_length != len(MODEL) or _read_exact(file, model_length) != MODEL:
            raise InvalidInputFormat("Unsupported Vigor firmware model")

        first_entry = cls._parse_entry(file, cls._parse_descriptor(file))
        second_entry = cls._parse_entry(file, cls._parse_descriptor(file))
        if (
            first_entry.name != NONCE_ENTRY_NAME
            or first_entry.data_length != NONCE_LENGTH
        ):
            raise InvalidInputFormat("Invalid Vigor 3910 nonce entry")
        if (
            second_entry.name != IMAGE_ENTRY_NAME
            or second_entry.data_length < ARM64_IMAGE_HEADER_LENGTH
        ):
            raise InvalidInputFormat("Invalid Vigor 3910 encrypted image entry")

        entries = [first_entry, second_entry]
        end_offset = second_entry.end_offset
        while entry := cls._probe_entry(file, end_offset):
            if entry.name.startswith(ENCRYPTED_ENTRY_PREFIX) and entry.data_length == 0:
                raise InvalidInputFormat("Empty Vigor 3910 encrypted entry")
            entries.append(entry)
            end_offset = entry.end_offset

        nonce = _read_at(file, first_entry.data_offset, first_entry.data_length)
        return Vigor3910Layout(start_offset, end_offset, nonce, tuple(entries))


class Vigor3910Extractor(Extractor):
    @staticmethod
    def _decrypt_entry(
        file: File,
        filesystem: FileSystem,
        nonce: bytes,
        entry: Vigor3910Entry,
    ) -> None:
        decryptor = _new_chacha20_decryptor(CHACHA20_KEY, nonce)
        read_length = 0
        output_path = Path(f"{entry.name}.decrypted")
        with filesystem.open(output_path, "wb+") as output:
            for encrypted in iterate_file(file, entry.data_offset, entry.data_length):
                read_length += len(encrypted)
                output.write(decryptor.update(encrypted))
            if read_length != entry.data_length:
                raise EOFError
            output.write(decryptor.finalize())

    def extract(self, inpath: Path, outdir: Path) -> ExtractResult:
        filesystem = FileSystem(outdir)
        with File.from_path(inpath) as file:
            layout = Vigor3910Parser.parse(file, 0)
            for entry in layout.entries:
                if entry.name.startswith(ENCRYPTED_ENTRY_PREFIX):
                    self._decrypt_entry(file, filesystem, layout.nonce, entry)

        return ExtractResult(reports=filesystem.problems)


class DraytekVigor3910Handler(StructHandler):
    NAME = "draytek_vigor3910"
    PATTERNS = [
        HexString(
            """
            05 00 00 00 76 33 39 31 30
            [64]
            05 00 00 00 6E 6F 6E 63 65 0C 00 00 00 [12]
            [64]
            09 00 00 00 65 6E 63 5F 49 6D 61 67 65
            """
        )
    ]
    PATTERN_MATCH_OFFSET = -MODEL_HEADER_OFFSET
    C_DEFINITIONS = VIGOR3910_C_DEFINITIONS
    HEADER_STRUCT = "vigor3910_header_t"
    EXTRACTOR = Vigor3910Extractor()
    DOC = HandlerDoc(
        name="DrayTek Vigor 3910 encrypted firmware",
        description=(
            "Encrypted firmware containers used by DrayTek Vigor 3910 systems. "
            "The handler decrypts every encrypted entry and emits the results "
            "for recursive extraction."
        ),
        handler_type=HandlerType.ARCHIVE,
        vendor="DrayTek",
        references=[
            Reference(
                title="DrayTek Vigor 3910 firmware analysis",
                url="https://wzt.ac.cn/2024/02/19/vigor_3910/",
            ),
            Reference(
                title="RFC 8439: ChaCha20 and Poly1305 for IETF Protocols",
                url="https://www.rfc-editor.org/rfc/rfc8439",
            ),
            Reference(
                title="Booting AArch64 Linux",
                url="https://www.kernel.org/doc/html/v6.12/arch/arm64/booting.html",
            ),
        ],
        limitations=[
            "Only Vigor 3910 containers using a 12-byte nonce and the known fixed key are supported.",
            "The proprietary entry checksums are not verified.",
        ],
    )

    def calculate_chunk(self, file: File, start_offset: int) -> ValidChunk:
        layout = Vigor3910Parser.parse(file, start_offset)
        image_entry = layout.entries[1]
        encrypted_header = _read_at(
            file,
            image_entry.data_offset,
            ARM64_IMAGE_HEADER_LENGTH,
        )
        decryptor = _new_chacha20_decryptor(CHACHA20_KEY, layout.nonce)
        decrypted_header = decryptor.update(encrypted_header) + decryptor.finalize()
        if (
            decrypted_header[
                ARM64_IMAGE_MAGIC_OFFSET : ARM64_IMAGE_MAGIC_OFFSET
                + len(ARM64_IMAGE_MAGIC)
            ]
            != ARM64_IMAGE_MAGIC
        ):
            raise InvalidInputFormat("Invalid decrypted Vigor 3910 ARM64 image header")

        return ValidChunk(
            start_offset=layout.start_offset,
            end_offset=layout.end_offset,
        )
