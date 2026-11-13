from __future__ import annotations

import binascii
import hashlib
import io
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from unblob.file_utils import (
    Endian,
    File,
    FileSystem,
    InvalidInputFormat,
    StructParser,
    iterate_file,
    iterate_patterns,
)
from unblob.handlers.compression.gzip import GZIPHandler
from unblob.models import (
    Extractor,
    ExtractResult,
    Handler,
    HandlerDoc,
    HandlerType,
    HexString,
    Reference,
    StructHandler,
    ValidChunk,
)
from unblob.report import ExtractionProblem

PERIOD = 0x800
ROUNDS = 3
UINT32_MAX = (1 << 32) - 1

SIGNATURE_LENGTH = 0x100
V3_FOOTER_LENGTH = 20
V3_TRAILER_LENGTH = V3_FOOTER_LENGTH + SIGNATURE_LENGTH

IKMF_MANIFEST_LENGTH = 0x80
IKMF_SIGNED_OVERHEAD = V3_FOOTER_LENGTH + IKMF_MANIFEST_LENGTH + SIGNATURE_LENGTH

XZ_STREAM_HEADER_LENGTH = 12
XZ_MAGIC = b"\xfd7zXZ\x00"
XZ_CRC32_FLAGS = b"\x00\x01"

V3_GZIP_PREFIX = b"\x1f\x8b\x08\x00"
V3_GZIP_HEADER_LENGTH = 10

# Static domain separator and labels reproduced from the IKMF v3 kernel KDF.
IKMF_KDF_DOMAIN = bytes.fromhex(
    "42712a9fd3556c1888e41d0b63af72c6195eb134cd078a90ef2c446da813f75b"
)
IKMF_MANIFEST_GATE_LABEL = b"ikuai-rootfs-manifest-gate-v1"
IKMF_TEXT_GATE_LABEL = b"ikuai-rootfs-text-gate-v1"
IKMF_STATE_MAC_LABEL = b"ikuai-rootfs-state-mac-v1"
IKMF_CONTEXT_INIT_LABEL = b"ikuai-rootfs-ctx-init-v1"
IKMF_CONTEXT_KDF_LABEL = b"ikuai-rootfs-ctx-kdf-v1"
IKMF_ROOTFS_KEY_LABEL = b"ikuai-rootfs-ctx-rootfs-key-v1"

# https://github.com/iKuaiPatch/rootfs/blob/6fe50a5f7ad1810b08ce6f3fe295dd55e49a304f/tools/ikmf.md
IKMF_C_DEFINITIONS = r"""
    typedef struct ikuai_ikmf_manifest {
        char   magic[4];
        uint32 version;
        uint32 header_length;
        uint32 flags;
        uint64 payload_length;
        uint64 payload_length_copy;
        uint8  payload_hash[32];
        uint8  unknown_digest[32];
        uint8  kernel_text_hash[32];
    } ikuai_ikmf_manifest_t;
"""

# Public key used to verify the legacy V3 rootfs signature.
# https://github.com/iKuaiPatch/rootfs/blob/6fe50a5f7ad1810b08ce6f3fe295dd55e49a304f/key/firmware_v3_public_key.pem
IKUAI_V3_PUBLIC_KEY = b"""-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAycYDpnZPG6asMUyErAId
QtKNvekwitsvuNvM7RzNvCDGqB7hwxD6AJjVFenRLwWP7r2qHC3ofD/Iqkx8rw7z
w2nsuSPWfobkmKyWLQP3sDcA/JEzXWU5h1SHjJMjR1cT5Ry+/TBbuMjeM2pdF1U0
DSnX8E4IcEjl1NM8CsIUBD3ca8e2qylBFxWbLioMJAd5bZ+mJUH4CSuwtvnSHFb5
FAX3s4XmhHpQrIFS1p9AP3TVZJz7fbigU2/x5RG7K9Iz774QlOybtZP/nEzRkbP3
gjO+G7ol4+QxeSMc5kaXIu5AZpNILVIhediuQL7jobddY/033grfUEREFhJX8kBY
MQIDAQAB
-----END PUBLIC KEY-----
"""

# The legacy V3 format uses the V2 hash table, not the standard CRC32 table.
# https://github.com/iKuaiPatch/rootfs/blob/6fe50a5f7ad1810b08ce6f3fe295dd55e49a304f/tools/rootfs_tools/src/codecs/V2Codec.cpp
V2_HASH_WORDS = struct.unpack(
    "<16I",
    bytes.fromhex(
        "00000000 6410b71d c8206e3b ac30d626"
        "9041de76 f4516b6b 5661b24d 3c710550"
        "2063b8ed 43930ff0 e8a3d6d6 8cb361cb"
        "b0c2649b d4d2d486 78e20aa0 1cf2ddbd"
    ),
)


def _read_exact(file: File, length: int) -> bytes:
    data = file.read(length)
    if len(data) != length:
        raise EOFError
    return data


def _read_at(file: File, offset: int, length: int) -> bytes:
    file.seek(offset, io.SEEK_SET)
    return _read_exact(file, length)


def _rol8(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (8 - shift))) & 0xFF


def _ror8(value: int, shift: int) -> int:
    return ((value >> shift) | (value << (8 - shift))) & 0xFF


def _rol32(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (32 - shift))) & UINT32_MAX


def _u32le(value: int) -> bytes:
    if not 0 <= value <= UINT32_MAX:
        raise InvalidInputFormat("Value does not fit in the IKMF u32 KDF input")
    return struct.pack("<I", value)


def _is_xz_stream_header(data: bytes) -> bool:
    if len(data) != XZ_STREAM_HEADER_LENGTH or not data.startswith(XZ_MAGIC):
        return False

    flags = data[6:8]
    stored_crc = struct.unpack("<I", data[8:12])[0]
    return flags == XZ_CRC32_FLAGS and stored_crc == binascii.crc32(flags)


@dataclass(frozen=True)
class IkuaiV3Cipher:
    table: bytes
    inverse_sbox: bytes
    position_maps: tuple[tuple[bytes, ...], ...]

    @classmethod
    def from_seed(cls, seed: bytes, payload_length: int) -> IkuaiV3Cipher:
        if len(seed) != 16:
            raise InvalidInputFormat("Invalid iKuai V3 seed length")
        if not 0 < payload_length <= UINT32_MAX:
            raise InvalidInputFormat("Invalid iKuai V3 payload length")

        table, inverse_sbox = cls._make_tables(seed)
        position_maps = cls._make_position_maps(table, inverse_sbox, payload_length)
        return cls(bytes(table), bytes(inverse_sbox), position_maps)

    @staticmethod
    def _make_tables(seed: bytes) -> tuple[bytearray, bytearray]:
        table = bytearray(PERIOD)
        for index in range(PERIOD):
            product = (index + 1) * 0x12FE500
            table[index] = (seed[index & 0xF] + product // 0x83) & 0xFF

        for index in range(PERIOD):
            table[index] ^= (index + table[(7 * index + 13) % PERIOD]) & 0xFF

        for index in range(PERIOD):
            current = table[index]
            neighbours = (
                current
                + table[(PERIOD + index - 1) % PERIOD]
                + table[(index + 1) % PERIOD]
            ) & 0xFF
            table[index] = _rol8(current, 1) ^ neighbours

        sbox = bytearray(range(256))
        accumulator = 0
        for index in range(256):
            current = sbox[index]
            accumulator = (current + seed[index & 0xF] + accumulator) & 0xFF
            sbox[index] = sbox[accumulator]
            sbox[accumulator] = current

        inverse_sbox = bytearray(256)
        for index, value in enumerate(sbox):
            inverse_sbox[value] = index

        for index in range(PERIOD):
            table[index] = sbox[table[index]] ^ ((0x10 >> (8 * (index & 3))) & 0xFF)

        return table, inverse_sbox

    @staticmethod
    def _make_position_maps(
        table: bytearray,
        inverse_sbox: bytearray,
        payload_length: int,
    ) -> tuple[tuple[bytes, ...], ...]:
        key_maps = []
        for round_index in range(ROUNDS):
            maps_for_round = []
            for key_byte in range(256):
                mixed = (payload_length + round_index + key_byte) & UINT32_MAX
                carry = (mixed >> 8) & 0xFF
                shift = (mixed & 7) + 1
                subtract_after_rotate = (mixed + round_index) & 0xFF
                maps_for_round.append(
                    bytes(
                        inverse_sbox[
                            (
                                _ror8(
                                    ((ciphertext - round_index - carry) & 0xFF) ^ carry,
                                    shift,
                                )
                                - subtract_after_rotate
                            )
                            & 0xFF
                        ]
                        for ciphertext in range(256)
                    )
                )
            key_maps.append(maps_for_round)

        return tuple(
            tuple(
                key_maps[round_index][
                    table[(17 * round_index + position) & (PERIOD - 1)]
                ]
                for position in range(PERIOD)
            )
            for round_index in range(ROUNDS)
        )

    def decrypt(self, data: bytes, offset: int = 0) -> bytes:
        output = bytearray(data)
        for round_index in range(ROUNDS - 1, -1, -1):
            maps_for_round = self.position_maps[round_index]
            for index, encrypted_value in enumerate(output):
                output[index] = maps_for_round[(offset + index) & (PERIOD - 1)][
                    encrypted_value
                ]

        return bytes(output)


class IkuaiV2Hash:
    def __init__(self) -> None:
        self._state = UINT32_MAX

    @staticmethod
    def _update(state: int, data: bytes) -> int:
        for value in data:
            intermediate = V2_HASH_WORDS[(value ^ state) & 0xF] ^ (state >> 4)
            state = V2_HASH_WORDS[(intermediate ^ (value >> 4)) & 0xF] ^ (
                intermediate >> 4
            )
        return state & UINT32_MAX

    def update(self, data: bytes) -> None:
        self._state = self._update(self._state, data)

    def digest(self) -> bytes:
        first = (~self._state) & UINT32_MAX
        second_state = self._update(UINT32_MAX, struct.pack(">I", first))
        second = (~second_state) & UINT32_MAX
        return struct.pack(">I", second)


class Hasher(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def digest(self) -> bytes: ...


def _ikmf_gate_value(
    source: bytes,
    digest1: bytes,
    digest2: bytes,
    payload_length: int,
    signed_length: int,
    magic: int,
    step: int,
) -> int:
    state = (payload_length ^ signed_length ^ magic) & UINT32_MAX
    accumulator = 0
    left = right = 0
    for source_byte, digest1_byte, digest2_byte in zip(
        source, digest1, digest2, strict=True
    ):
        left = _rol32(_rol32(state, 5) ^ accumulator ^ source_byte, 7)
        right = digest1_byte ^ digest2_byte
        state = left ^ right
        accumulator = (accumulator + step) & UINT32_MAX

    return magic if left == right else state


def derive_ikmf_seed(
    manifest: bytes,
    inner_signature: bytes,
    footer: bytes,
    payload_length: int,
    signed_length: int,
) -> bytes:
    manifest_hash = hashlib.sha256(manifest).digest()
    signature_hash = hashlib.sha256(inner_signature).digest()

    manifest_gate_key = hashlib.sha256(
        IKMF_KDF_DOMAIN
        + IKMF_MANIFEST_GATE_LABEL
        + manifest_hash
        + signature_hash
        + _u32le(payload_length)
        + _u32le(signed_length)
    ).digest()
    manifest_gate = _ikmf_gate_value(
        manifest_gate_key,
        manifest_hash,
        signature_hash,
        payload_length,
        signed_length,
        0x696B6D67,
        0x27D4EB2D,
    )

    kernel_text_hash = manifest[96:128]
    text_gate_key = hashlib.sha256(
        IKMF_KDF_DOMAIN
        + IKMF_TEXT_GATE_LABEL
        + kernel_text_hash
        + kernel_text_hash
        + manifest_hash
        + signature_hash
        + _u32le(payload_length)
        + _u32le(signed_length)
    ).digest()
    text_gate = _ikmf_gate_value(
        text_gate_key,
        manifest_hash,
        signature_hash,
        payload_length,
        signed_length,
        0x696B7467,
        0x045D9F3B,
    )

    state_mac = hashlib.sha256(
        IKMF_KDF_DOMAIN
        + IKMF_STATE_MAC_LABEL
        + manifest_hash
        + signature_hash
        + manifest_gate_key
        + _u32le(manifest_gate)
        + text_gate_key
        + _u32le(text_gate)
        + manifest[32:64]
        + manifest[64:96]
        + manifest[96:128]
        + _u32le(payload_length)
        + _u32le(signed_length)
    ).digest()

    context_hash = hashlib.sha256(
        IKMF_KDF_DOMAIN
        + IKMF_CONTEXT_INIT_LABEL
        + manifest_hash
        + signature_hash
        + manifest_gate_key
        + _u32le(manifest_gate)
        + text_gate_key
        + _u32le(text_gate)
        + state_mac
        + manifest[96:128]
        + _u32le(payload_length)
        + _u32le(signed_length)
        + manifest[64:96]
    ).digest()
    context_hash = hashlib.sha256(
        IKMF_KDF_DOMAIN + context_hash + IKMF_CONTEXT_KDF_LABEL + manifest[32:64]
    ).digest()

    return hashlib.sha256(
        IKMF_KDF_DOMAIN
        + context_hash
        + footer[:16]
        + signature_hash
        + text_gate_key
        + _u32le(text_gate)
        + _u32le(manifest_gate)
        + state_mac
        + IKMF_ROOTFS_KEY_LABEL
    ).digest()[:16]


def _decrypt_payload(
    file: File,
    filesystem: FileSystem,
    seed: bytes,
    payload_length: int,
    expected_hash: bytes,
    hasher: Hasher,
) -> None:
    cipher = IkuaiV3Cipher.from_seed(seed, payload_length)
    written = 0
    with filesystem.open(Path("rootfs.decrypted"), "wb+") as output:
        for encrypted in iterate_file(file, 0, payload_length):
            plaintext = cipher.decrypt(encrypted, written)
            output.write(plaintext)
            hasher.update(plaintext)
            written += len(encrypted)

    if written != payload_length:
        raise EOFError

    if hasher.digest() != expected_hash:
        filesystem.record_problem(
            ExtractionProblem(
                problem="iKuai rootfs payload hash does not match",
                resolution="Kept the decrypted payload for analysis.",
            )
        )


@lru_cache
def _load_rsa_public_key(public_key_pem: bytes) -> rsa.RSAPublicKey:
    public_key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(public_key, rsa.RSAPublicKey):
        raise TypeError("iKuai V3 public key is not an RSA key")
    return public_key


def _verify_v3_signature(
    file: File,
    signed_length: int,
    signature: bytes,
    public_key_pem: bytes,
) -> None:
    # Reject ordinary gzip candidates before hashing their preceding data.
    try:
        signed_digest = _load_rsa_public_key(
            public_key_pem
        ).recover_data_from_signature(
            signature,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    except InvalidSignature as exc:
        raise InvalidInputFormat("Invalid iKuai V3 rootfs signature") from exc

    digest = hashlib.sha256()
    read_length = 0
    for data in iterate_file(file, 0, signed_length):
        digest.update(data)
        read_length += len(data)
    if read_length != signed_length:
        raise EOFError

    if digest.digest() != signed_digest:
        raise InvalidInputFormat("Invalid iKuai V3 rootfs signature")


@dataclass(frozen=True)
class V3Layout:
    payload_length: int
    gzip_offset: int
    seed: bytes
    stored_hash: bytes


def _parse_v3_layout(file: File, gzip_offset: int) -> V3Layout:
    payload_length = gzip_offset - V3_TRAILER_LENGTH
    if payload_length <= 0 or payload_length > UINT32_MAX:
        raise InvalidInputFormat("Invalid iKuai V3 payload length")

    gzip_header = _read_at(file, gzip_offset, V3_GZIP_HEADER_LENGTH)
    # The handler pattern guarantees these bytes; the extractor repeats the
    # check while rediscovering the layout in the carved bundle.
    if gzip_header[:4] != V3_GZIP_PREFIX or gzip_header[8] != 0:
        raise InvalidInputFormat("Invalid iKuai V3 trailing gzip header")

    footer = _read_at(file, payload_length, V3_FOOTER_LENGTH)
    seed = footer[:16]
    stored_hash = footer[16:20]

    return V3Layout(payload_length, gzip_offset, seed, stored_hash)


def _validate_v3_header(file: File, layout: V3Layout) -> None:
    encrypted_header = _read_at(file, 0, XZ_STREAM_HEADER_LENGTH)
    decrypted_header = IkuaiV3Cipher.from_seed(
        layout.seed, layout.payload_length
    ).decrypt(encrypted_header)
    if not _is_xz_stream_header(decrypted_header):
        raise InvalidInputFormat("Invalid decrypted iKuai V3 XZ header")


class IkuaiIkmfRootfsExtractor(Extractor):
    def __init__(self) -> None:
        self._struct_parser = StructParser(IKMF_C_DEFINITIONS)

    def extract(self, inpath: Path, outdir: Path) -> ExtractResult:
        filesystem = FileSystem(outdir)
        with File.from_path(inpath) as file:
            manifest_offset = file.size() - IKMF_MANIFEST_LENGTH - 2 * SIGNATURE_LENGTH
            manifest = _read_at(file, manifest_offset, IKMF_MANIFEST_LENGTH)
            header = self._struct_parser.parse(
                "ikuai_ikmf_manifest_t", manifest, Endian.LITTLE
            )
            payload_length = int(header.payload_length)
            if manifest_offset != payload_length + V3_FOOTER_LENGTH:
                raise InvalidInputFormat("Invalid carved IKMF layout")

            footer = _read_at(file, payload_length, V3_FOOTER_LENGTH)
            inner_signature = _read_at(
                file,
                manifest_offset + IKMF_MANIFEST_LENGTH,
                SIGNATURE_LENGTH,
            )
            signed_length = payload_length + IKMF_SIGNED_OVERHEAD
            seed = derive_ikmf_seed(
                manifest,
                inner_signature,
                footer,
                payload_length,
                signed_length,
            )
            _decrypt_payload(
                file,
                filesystem,
                seed,
                payload_length,
                bytes(header.payload_hash),
                hashlib.sha256(),
            )

        return ExtractResult(reports=filesystem.problems)


class IkuaiV3RootfsExtractor(Extractor):
    def _find_layout(self, file: File) -> V3Layout:
        layouts = []
        for gzip_offset in iterate_patterns(file, V3_GZIP_PREFIX):
            try:
                layout = _parse_v3_layout(file, gzip_offset)
                _validate_v3_header(file, layout)
                layouts.append(layout)
            except (EOFError, InvalidInputFormat):
                continue

        if len(layouts) != 1:
            raise InvalidInputFormat("Unable to identify a unique iKuai V3 layout")
        return layouts[0]

    def extract(self, inpath: Path, outdir: Path) -> ExtractResult:
        filesystem = FileSystem(outdir)
        with File.from_path(inpath) as file:
            layout = self._find_layout(file)
            _decrypt_payload(
                file,
                filesystem,
                layout.seed,
                layout.payload_length,
                layout.stored_hash,
                IkuaiV2Hash(),
            )
            filesystem.carve(
                Path("trailing.gzip"),
                file,
                layout.gzip_offset,
                file.size() - layout.gzip_offset,
            )

        return ExtractResult(reports=filesystem.problems)


class IkuaiIkmfRootfsHandler(StructHandler):
    NAME = "ikuai_ikmf_rootfs"
    PATTERNS = [HexString("49 4B 4D 46 03 00 00 00 80 00 00 00 00 00 00 00")]
    C_DEFINITIONS = IKMF_C_DEFINITIONS
    HEADER_STRUCT = "ikuai_ikmf_manifest_t"
    EXTRACTOR = IkuaiIkmfRootfsExtractor()
    DOC = HandlerDoc(
        name="iKuai IKMF rootfs",
        description=(
            "Encrypted rootfs images used by iKuai 4.0.306 and later x64 "
            "systems. The handler derives the IKMF v3 payload key and emits "
            "the decrypted XZ stream for recursive extraction."
        ),
        handler_type=HandlerType.ARCHIVE,
        vendor="iKuai",
        references=[
            Reference(
                title="iKuai 4.0.306+ x64 rootfs (IKMF)",
                url="https://github.com/iKuaiPatch/rootfs/blob/6fe50a5f7ad1810b08ce6f3fe295dd55e49a304f/tools/ikmf.md",
            )
        ],
        limitations=["Only version 3 of the x64 IKMF format is supported."],
    )

    def calculate_chunk(self, file: File, start_offset: int) -> ValidChunk:
        header = self.parse_header(file, Endian.LITTLE)
        payload_length = int(header.payload_length)
        if payload_length == 0:
            raise InvalidInputFormat("IKMF payload is empty")
        if payload_length != int(header.payload_length_copy):
            raise InvalidInputFormat("IKMF payload lengths do not match")
        if payload_length > UINT32_MAX - IKMF_SIGNED_OVERHEAD:
            raise InvalidInputFormat("IKMF payload is too large")
        if not any(bytes(header.kernel_text_hash)):
            raise InvalidInputFormat("IKMF kernel text hash is empty")

        chunk_start = start_offset - V3_FOOTER_LENGTH - payload_length
        if chunk_start < 0:
            raise InvalidInputFormat("IKMF manifest precedes its payload")

        manifest = _read_at(file, start_offset, IKMF_MANIFEST_LENGTH)
        footer = _read_at(file, chunk_start + payload_length, V3_FOOTER_LENGTH)
        inner_signature = _read_at(
            file,
            start_offset + IKMF_MANIFEST_LENGTH,
            SIGNATURE_LENGTH,
        )
        signed_length = payload_length + IKMF_SIGNED_OVERHEAD
        seed = derive_ikmf_seed(
            manifest,
            inner_signature,
            footer,
            payload_length,
            signed_length,
        )
        encrypted_header = _read_at(file, chunk_start, XZ_STREAM_HEADER_LENGTH)
        if not _is_xz_stream_header(
            IkuaiV3Cipher.from_seed(seed, payload_length).decrypt(encrypted_header)
        ):
            raise InvalidInputFormat("Invalid decrypted IKMF XZ header")

        return ValidChunk(
            start_offset=chunk_start,
            end_offset=start_offset + IKMF_MANIFEST_LENGTH + 2 * SIGNATURE_LENGTH,
        )


class IkuaiV3RootfsHandler(Handler):
    NAME = "ikuai_v3_rootfs"
    PATTERNS = [HexString("1F 8B 08 00 [4] 00")]
    EXTRACTOR = IkuaiV3RootfsExtractor()
    PUBLIC_KEY_PEM = IKUAI_V3_PUBLIC_KEY
    DOC = HandlerDoc(
        name="iKuai V3 rootfs",
        description=(
            "RSA-signed V3 rootfs bundles used by iKuai systems. The handler "
            "verifies the official signature, decrypts the XZ rootfs, and "
            "preserves the trailing gzip member for recursive extraction."
        ),
        handler_type=HandlerType.ARCHIVE,
        vendor="iKuai",
        references=[
            Reference(
                title="iKuai rootfs V3 codec",
                url="https://github.com/iKuaiPatch/rootfs/blob/6fe50a5f7ad1810b08ce6f3fe295dd55e49a304f/tools/rootfs_tools/src/codecs/V3Codec.cpp",
            ),
            Reference(
                title="iKuai V3 firmware public key",
                url="https://github.com/iKuaiPatch/rootfs/blob/6fe50a5f7ad1810b08ce6f3fe295dd55e49a304f/key/firmware_v3_public_key.pem",
            ),
        ],
        limitations=[
            "The bundle must start at offset zero, use the official RSA key, and contain a trailing Linux gzip stream.",
            "V1, V2, and V3 images signed with custom keys are not supported.",
        ],
    )

    def calculate_chunk(self, file: File, start_offset: int) -> ValidChunk:
        layout = _parse_v3_layout(file, start_offset)
        signed_length = layout.payload_length + V3_FOOTER_LENGTH
        signature = _read_at(file, signed_length, SIGNATURE_LENGTH)
        _verify_v3_signature(
            file,
            signed_length,
            signature,
            self.PUBLIC_KEY_PEM,
        )
        _validate_v3_header(file, layout)

        file.seek(start_offset, io.SEEK_SET)
        gzip_chunk = GZIPHandler().calculate_chunk(file, start_offset)
        if gzip_chunk is None:
            raise InvalidInputFormat("Invalid iKuai V3 trailing gzip stream")

        return ValidChunk(start_offset=0, end_offset=gzip_chunk.end_offset)
