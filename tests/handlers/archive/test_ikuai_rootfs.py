import gzip
import hashlib
import io
import lzma
import struct
from pathlib import Path
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from unblob.file_utils import File, InvalidInputFormat, iterate_file
from unblob.finder import search_chunks
from unblob.handlers.archive.ikuai import rootfs
from unblob.handlers.archive.ikuai.rootfs import (
    IKMF_MANIFEST_LENGTH,
    PERIOD,
    ROUNDS,
    SIGNATURE_LENGTH,
    UINT32_MAX,
    V3_FOOTER_LENGTH,
    IkuaiIkmfRootfsHandler,
    IkuaiV2Hash,
    IkuaiV3Cipher,
    IkuaiV3RootfsHandler,
    derive_ikmf_seed,
)
from unblob.handlers.compression.gzip import GZIPHandler
from unblob.models import Task, TaskResult
from unblob.report import ExtractionProblem
from unblob.testing import find_chunks


def _rol8(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (8 - shift))) & 0xFF


def _encrypt_payload(payload: bytes, seed: bytes) -> bytes:
    context = IkuaiV3Cipher.from_seed(seed, len(payload))
    sbox = bytearray(256)
    for value, index in enumerate(context.inverse_sbox):
        sbox[index] = value

    output = bytearray(payload)
    for round_index in range(ROUNDS):
        base = 17 * round_index
        for index, plaintext in enumerate(output):
            key_byte = context.table[(base + index) & (PERIOD - 1)]
            mixed = (len(payload) + round_index + key_byte) & UINT32_MAX
            carry = (mixed >> 8) & 0xFF
            shift = (mixed & 7) + 1

            encrypted = sbox[plaintext]
            encrypted = (encrypted + ((mixed + round_index) & 0xFF)) & 0xFF
            encrypted = _rol8(encrypted, shift)
            encrypted ^= carry
            output[index] = (encrypted + round_index + carry) & 0xFF

    return bytes(output)


def _vendor_hash(payload: bytes) -> bytes:
    hasher = IkuaiV2Hash()
    hasher.update(payload)
    return hasher.digest()


def _xz_payload(content: bytes = b"synthetic iKuai rootfs\n") -> bytes:
    return lzma.compress(content, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC32)


def _trailing_gzip() -> bytes:
    data = bytearray(
        gzip.compress(b"synthetic trailing data\n", compresslevel=6, mtime=0)
    )
    data[9] = 3
    assert data[:4] == b"\x1f\x8b\x08\x00"
    assert data[8:10] == b"\x00\x03"
    return bytes(data)


def _public_key_pem(private_key: rsa.RSAPrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _v3_handler(private_key: rsa.RSAPrivateKey) -> type[IkuaiV3RootfsHandler]:
    class TestIkuaiV3RootfsHandler(IkuaiV3RootfsHandler):
        PUBLIC_KEY_PEM = _public_key_pem(private_key)

    return TestIkuaiV3RootfsHandler


def _build_v3(
    private_key: rsa.RSAPrivateKey,
    *,
    payload: bytes | None = None,
    stored_hash: bytes | None = None,
) -> tuple[bytes, bytes, bytes]:
    payload = payload or _xz_payload()
    seed = bytes(range(16))
    encrypted = _encrypt_payload(payload, seed)
    stored_hash = stored_hash or _vendor_hash(payload)
    signed = encrypted + seed + stored_hash
    signature = private_key.sign(signed, padding.PKCS1v15(), hashes.SHA256())
    trailing = _trailing_gzip()
    return signed + signature + trailing, payload, trailing


def _build_ikmf(
    *,
    payload: bytes | None = None,
    stored_hash: bytes | None = None,
    payload_length_copy: int | None = None,
    kernel_text_hash: bytes = b"K" * 32,
) -> tuple[bytes, bytes]:
    payload = payload or _xz_payload()
    payload_length = len(payload)
    stored_hash = stored_hash or hashlib.sha256(payload).digest()
    payload_length_copy = payload_length_copy or payload_length
    manifest = (
        struct.pack(
            "<4sIIIQQ",
            b"IKMF",
            3,
            IKMF_MANIFEST_LENGTH,
            0,
            payload_length,
            payload_length_copy,
        )
        + stored_hash
        + b"D" * 32
        + kernel_text_hash
    )
    footer = bytes(range(16, 32)) + b"TAG!"
    inner_signature = b"I" * SIGNATURE_LENGTH
    outer_signature = b"O" * SIGNATURE_LENGTH
    signed_length = (
        payload_length + V3_FOOTER_LENGTH + IKMF_MANIFEST_LENGTH + SIGNATURE_LENGTH
    )
    seed = derive_ikmf_seed(
        manifest,
        inner_signature,
        footer,
        payload_length,
        signed_length,
    )
    encrypted = _encrypt_payload(payload, seed)
    return (
        encrypted + footer + manifest + inner_signature + outer_signature,
        payload,
    )


def _search(content: bytes, handlers):
    task_result = TaskResult(task=Task(path=Path("/nonexistent"), depth=0, blob_id=""))
    with File.from_bytes(content) as file:
        chunks = search_chunks(file, len(content), handlers, task_result)
    assert task_result.reports == []
    return chunks


@pytest.fixture(scope="module")
def private_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def test_v3_cipher_decrypts_across_chunk_boundaries():
    payload = bytes(range(256)) * 17
    seed = bytes(range(16))
    encrypted = _encrypt_payload(payload, seed)
    cipher = IkuaiV3Cipher.from_seed(seed, len(payload))

    decrypted = b"".join(
        (
            cipher.decrypt(encrypted[:17], 0),
            cipher.decrypt(encrypted[17:2065], 17),
            cipher.decrypt(encrypted[2065:], 2065),
        )
    )

    assert decrypted == payload


def test_v3_cipher_reference_vector():
    cipher = IkuaiV3Cipher.from_seed(bytes(range(16)), 0x12345678)

    assert cipher.decrypt(bytes(range(32))) == bytes.fromhex(
        "c855cd72ac9c1262e069ff77fc788d7233addd6a1037fdcee5c34012aa52a312"
    )


def test_v2_hash_vector():
    assert _vendor_hash(b"iKuai rootfs") == bytes.fromhex("d78a2ad0")


def test_ikmf_kdf_reference_vector():
    payload_length = 0x123456
    manifest = (
        struct.pack(
            "<4sIIIQQ",
            b"IKMF",
            3,
            IKMF_MANIFEST_LENGTH,
            0,
            payload_length,
            payload_length,
        )
        + bytes(range(32))
        + bytes(range(32, 64))
        + bytes(range(64, 96))
    )
    inner_signature = bytes(range(256))
    footer = bytes(range(16, 32)) + b"TAG!"
    signed_length = (
        payload_length + V3_FOOTER_LENGTH + IKMF_MANIFEST_LENGTH + SIGNATURE_LENGTH
    )

    assert derive_ikmf_seed(
        manifest,
        inner_signature,
        footer,
        payload_length,
        signed_length,
    ) == bytes.fromhex("018c030f8e9a75abacd14f212aa7c77e")


def test_ikmf_chunk_embedded_at_nonzero_offset():
    image, _payload = _build_ikmf()
    prefix = b"unrelated-prefix"
    content = prefix + image + b"unrelated-suffix"

    assert find_chunks(IkuaiIkmfRootfsHandler, content) == [
        (len(prefix), len(prefix) + len(image))
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda image, offset: image[: offset + 4] + b"\x04" + image[offset + 5 :],
            id="version",
        ),
        pytest.param(
            lambda image, offset: image[: offset + 8] + b"\x81" + image[offset + 9 :],
            id="header-length",
        ),
        pytest.param(
            lambda image, offset: image[: offset + 12] + b"\x01" + image[offset + 13 :],
            id="flags",
        ),
        pytest.param(
            lambda image, offset: (
                image[: offset + 24] + struct.pack("<Q", 1) + image[offset + 32 :]
            ),
            id="payload-length-copy",
        ),
        pytest.param(
            lambda image, offset: (
                image[: offset + 96] + bytes(32) + image[offset + 128 :]
            ),
            id="kernel-text-hash",
        ),
        pytest.param(lambda image, _offset: image[:-1], id="truncated-outer-signature"),
    ],
)
def test_ikmf_rejects_invalid_layout(mutation):
    image, payload = _build_ikmf()
    manifest_offset = len(payload) + V3_FOOTER_LENGTH

    assert find_chunks(IkuaiIkmfRootfsHandler, mutation(image, manifest_offset)) == []


def test_ikmf_rejects_invalid_decrypted_xz_header():
    image, _payload = _build_ikmf(payload=b"not an XZ stream")

    assert find_chunks(IkuaiIkmfRootfsHandler, image) == []


def test_ikmf_extracts_and_reports_hash_mismatch(tmp_path: Path):
    image, payload = _build_ikmf(stored_hash=bytes(32))
    inpath = tmp_path / "ikmf.bin"
    outdir = tmp_path / "output"
    inpath.write_bytes(image)

    result = IkuaiIkmfRootfsHandler.EXTRACTOR.extract(inpath, outdir)

    assert (outdir / "rootfs.decrypted").read_bytes() == payload
    assert len(result.reports) == 1
    assert isinstance(result.reports[0], ExtractionProblem)


def test_v3_short_pattern_wins_before_gzip(
    private_key: rsa.RSAPrivateKey,
):
    image, _payload, _trailing = _build_v3(private_key)
    handler = _v3_handler(private_key)

    chunks = _search(image, (GZIPHandler, handler))

    assert [
        (chunk.handler.NAME, chunk.start_offset, chunk.end_offset) for chunk in chunks
    ] == [(IkuaiV3RootfsHandler.NAME, 0, len(image))]


def test_v3_accepts_non_unix_gzip_os(private_key: rsa.RSAPrivateKey):
    image, payload, _trailing = _build_v3(private_key)
    gzip_offset = len(payload) + V3_FOOTER_LENGTH + SIGNATURE_LENGTH
    image = image[: gzip_offset + 9] + b"\x00" + image[gzip_offset + 10 :]
    handler = _v3_handler(private_key)

    assert find_chunks(handler, image) == [(0, len(image))]


def test_v3_rejects_bad_signature_and_leaves_gzip_detectable(
    private_key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch
):
    image, payload, _trailing = _build_v3(private_key)
    signature_offset = len(payload) + V3_FOOTER_LENGTH
    corrupted = bytearray(image)
    corrupted[signature_offset] ^= 1
    handler = _v3_handler(private_key)

    body_reader = Mock(wraps=iterate_file)
    cipher_factory = Mock(wraps=IkuaiV3Cipher.from_seed)
    monkeypatch.setattr(rootfs, "iterate_file", body_reader)
    monkeypatch.setattr(IkuaiV3Cipher, "from_seed", cipher_factory)

    chunks = _search(bytes(corrupted), (GZIPHandler, handler))

    assert [chunk.handler.NAME for chunk in chunks] == [GZIPHandler.NAME]
    body_reader.assert_not_called()
    cipher_factory.assert_not_called()


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(_trailing_gzip(), id="whole-gzip"),
        pytest.param(b"random-prefix" * 32 + _trailing_gzip(), id="embedded-gzip"),
    ],
)
def test_v3_does_not_claim_ordinary_gzip(
    content: bytes, private_key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch
):
    handler = _v3_handler(private_key)
    body_reader = Mock(wraps=iterate_file)
    cipher_factory = Mock(wraps=IkuaiV3Cipher.from_seed)
    monkeypatch.setattr(rootfs, "iterate_file", body_reader)
    monkeypatch.setattr(IkuaiV3Cipher, "from_seed", cipher_factory)

    chunks = _search(content, (GZIPHandler, handler))

    assert [chunk.handler.NAME for chunk in chunks] == [GZIPHandler.NAME]
    body_reader.assert_not_called()
    cipher_factory.assert_not_called()


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda payload: payload.__setitem__(7, 2), id="flags"),
        pytest.param(lambda payload: payload.__setitem__(8, payload[8] ^ 1), id="crc"),
    ],
)
def test_v3_rejects_invalid_decrypted_xz_header(
    mutation, private_key: rsa.RSAPrivateKey
):
    payload = bytearray(_xz_payload())
    mutation(payload)
    image, _payload, _trailing = _build_v3(private_key, payload=bytes(payload))
    handler = _v3_handler(private_key)

    assert find_chunks(handler, image) == []


@pytest.mark.parametrize(
    "removed_offset",
    [
        pytest.param(0, id="seed"),
        pytest.param(16, id="hash"),
        pytest.param(V3_FOOTER_LENGTH, id="signature"),
    ],
)
def test_v3_rejects_truncated_trailer_fields(
    removed_offset: int, private_key: rsa.RSAPrivateKey
):
    image, payload, _trailing = _build_v3(private_key)
    removed_offset += len(payload)
    truncated = image[:removed_offset] + image[removed_offset + 1 :]
    handler = _v3_handler(private_key)

    assert find_chunks(handler, truncated) == []


def test_v3_rejects_truncated_gzip(private_key: rsa.RSAPrivateKey):
    image, _payload, _trailing = _build_v3(private_key)
    handler = _v3_handler(private_key)

    assert find_chunks(handler, image[:-1]) == []


def test_v3_rejects_nonzero_embedding(private_key: rsa.RSAPrivateKey):
    image, _payload, _trailing = _build_v3(private_key)
    handler = _v3_handler(private_key)

    assert find_chunks(handler, b"prefix" + image) == []


def test_v3_extracts_both_children_and_reports_hash_mismatch(
    private_key: rsa.RSAPrivateKey, tmp_path: Path
):
    image, payload, trailing = _build_v3(private_key, stored_hash=bytes(4))
    inpath = tmp_path / "v3.bin"
    outdir = tmp_path / "output"
    inpath.write_bytes(image)

    result = IkuaiV3RootfsHandler.EXTRACTOR.extract(inpath, outdir)

    assert (outdir / "rootfs.decrypted").read_bytes() == payload
    assert (outdir / "trailing.gzip").read_bytes() == trailing
    assert len(result.reports) == 1
    assert isinstance(result.reports[0], ExtractionProblem)


def test_v3_rejects_modified_body_before_decrypting(
    private_key: rsa.RSAPrivateKey, monkeypatch: pytest.MonkeyPatch
):
    image, payload, trailing = _build_v3(private_key)
    corrupted = bytearray(image)
    corrupted[len(payload) - 1] ^= 1
    handler = _v3_handler(private_key)
    cipher_factory = Mock(wraps=IkuaiV3Cipher.from_seed)
    monkeypatch.setattr(IkuaiV3Cipher, "from_seed", cipher_factory)

    with (
        File.from_bytes(bytes(corrupted)) as file,
        pytest.raises(InvalidInputFormat, match="signature"),
    ):
        handler().calculate_chunk(file, len(image) - len(trailing))

    cipher_factory.assert_not_called()


@pytest.mark.parametrize(
    "field, expected_error",
    [
        pytest.param("manifest", EOFError, id="manifest"),
        pytest.param("footer", InvalidInputFormat, id="footer"),
    ],
)
def test_ikmf_parser_rejects_truncated_structures(field, expected_error):
    image, payload = _build_ikmf()
    manifest_offset = len(payload) + V3_FOOTER_LENGTH
    if field == "manifest":
        truncated = image[: manifest_offset + IKMF_MANIFEST_LENGTH - 1]
    else:
        truncated = image[: manifest_offset - 1] + image[manifest_offset:]
        manifest_offset -= 1

    with File.from_bytes(truncated) as file:
        file.seek(manifest_offset, io.SEEK_SET)
        with pytest.raises(expected_error):
            IkuaiIkmfRootfsHandler().calculate_chunk(file, manifest_offset)
