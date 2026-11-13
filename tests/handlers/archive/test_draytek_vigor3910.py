import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from unblob.file_utils import File, InvalidInputFormat
from unblob.handlers.archive.draytek.vigor3910 import (
    ARM64_IMAGE_MAGIC,
    ARM64_IMAGE_MAGIC_OFFSET,
    CHACHA20_KEY,
    DESCRIPTOR_LENGTH,
    MODEL,
    MODEL_HEADER_OFFSET,
    DraytekVigor3910Handler,
    Vigor3910Extractor,
    Vigor3910Parser,
    _new_chacha20_decryptor,
)
from unblob.testing import find_chunks

HEADER_LENGTH = 64 + 4 + len(MODEL)
NONCE_DESCRIPTOR_OFFSET = HEADER_LENGTH


def _crypt(data: bytes, nonce: bytes, *, key: bytes = CHACHA20_KEY) -> bytes:
    cipher = Cipher(algorithms.ChaCha20(key, bytes(4) + nonce), mode=None)
    context = cipher.encryptor()
    return context.update(data) + context.finalize()


def _entry(
    name: bytes,
    data: bytes,
    *,
    flags: int = 0,
    reserved: bytes = bytes(52),
    entry_length: int | None = None,
) -> bytes:
    body = struct.pack("<I", len(name)) + name + struct.pack("<I", len(data)) + data
    entry_length = entry_length or DESCRIPTOR_LENGTH + len(body)
    descriptor = struct.pack("<III", flags, entry_length, 0xA5A5A5A5) + reserved
    assert len(descriptor) == DESCRIPTOR_LENGTH
    return descriptor + body


def _arm64_image(payload: bytes = b"synthetic payload") -> bytes:
    image = bytearray(64)
    image[ARM64_IMAGE_MAGIC_OFFSET : ARM64_IMAGE_MAGIC_OFFSET + 4] = ARM64_IMAGE_MAGIC
    return bytes(image) + payload


def _build_image(
    *,
    nonce: bytes = b"abcdefghijkl",
    image_plaintext: bytes | None = None,
    boot_plaintext: bytes = b"synthetic boot image",
    metadata_name: bytes = b"fw_ver",
) -> tuple[bytes, dict[str, bytes]]:
    if image_plaintext is None:
        image_plaintext = _arm64_image()
    entries = (
        _entry(b"nonce", nonce),
        _entry(b"enc_Image", _crypt(image_plaintext, nonce)),
        _entry(b"enc_boot.img", _crypt(boot_plaintext, nonce)),
        _entry(metadata_name, b"4.3.2.5"),
    )
    header = bytes(64) + struct.pack("<I", len(MODEL)) + MODEL
    return header + b"".join(entries), {
        "enc_Image": image_plaintext,
        "enc_boot.img": boot_plaintext,
    }


def test_chacha20_ietf_reference_vector_across_chunks():
    key = bytes(range(32))
    nonce = bytes.fromhex("000000000000004a00000000")
    plaintext = (
        b"Ladies and Gentlemen of the class of '99: If I could offer you only "
        b"one tip for the future, sunscreen would be it."
    )
    ciphertext = bytes.fromhex(
        "6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0b"
        "f91b65c5524733ab8f593dabcd62b3571639d624e65152ab8f530c359f0861d8"
        "07ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab7793736"
        "5af90bbf74a35be6b40b8eedf2785e42874d"
    )
    decryptor = _new_chacha20_decryptor(key, nonce, counter=1)

    decrypted = b"".join(
        (
            decryptor.update(ciphertext[:17]),
            decryptor.update(ciphertext[17:73]),
            decryptor.update(ciphertext[73:]),
            decryptor.finalize(),
        )
    )

    assert decrypted == plaintext


@pytest.mark.parametrize(
    "nonce, counter",
    [
        pytest.param(b"short", 0, id="nonce"),
        pytest.param(bytes(12), -1, id="negative-counter"),
        pytest.param(bytes(12), 1 << 32, id="large-counter"),
    ],
)
def test_chacha20_rejects_invalid_parameters(nonce: bytes, counter: int):
    with pytest.raises(InvalidInputFormat):
        _new_chacha20_decryptor(CHACHA20_KEY, nonce, counter)


def test_chunk_embedded_at_nonzero_offset_stops_before_suffix():
    image, _plaintexts = _build_image()
    prefix = b"unrelated-prefix"
    suffix = b"unrelated-suffix"

    assert find_chunks(
        DraytekVigor3910Handler,
        prefix + image + suffix,
    ) == [(len(prefix), len(prefix) + len(image))]


def test_pattern_before_header_does_not_seek_negative():
    image, _plaintexts = _build_image()

    assert find_chunks(DraytekVigor3910Handler, image[MODEL_HEADER_OFFSET:]) == []


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda image: image[:64] + struct.pack("<I", 4) + image[68:],
            id="model-length",
        ),
        pytest.param(
            lambda image: image[:68] + b"v3911" + image[73:],
            id="model",
        ),
        pytest.param(
            lambda image: (
                image[:NONCE_DESCRIPTOR_OFFSET]
                + struct.pack("<I", 1)
                + image[NONCE_DESCRIPTOR_OFFSET + 4 :]
            ),
            id="descriptor-flags",
        ),
        pytest.param(
            lambda image: (
                image[: NONCE_DESCRIPTOR_OFFSET + 12]
                + b"\x01"
                + image[NONCE_DESCRIPTOR_OFFSET + 13 :]
            ),
            id="descriptor-reserved",
        ),
        pytest.param(
            lambda image: (
                image[: NONCE_DESCRIPTOR_OFFSET + 4]
                + struct.pack("<I", DESCRIPTOR_LENGTH + 9)
                + image[NONCE_DESCRIPTOR_OFFSET + 8 :]
            ),
            id="entry-length-mismatch",
        ),
        pytest.param(
            lambda image: image.replace(b"nonce", b"noncf", 1),
            id="nonce-name",
        ),
        pytest.param(
            lambda image: image.replace(
                struct.pack("<I", 12), struct.pack("<I", 11), 1
            ),
            id="nonce-length",
        ),
        pytest.param(
            lambda image: image.replace(b"enc_Image", b"enc_Imagd", 1),
            id="image-name",
        ),
    ],
)
def test_rejects_invalid_required_layout(mutation):
    image, _plaintexts = _build_image()

    assert find_chunks(DraytekVigor3910Handler, mutation(image)) == []


def test_rejects_unsafe_optional_entry_name():
    image, _plaintexts = _build_image(metadata_name=b"../bad")

    assert find_chunks(DraytekVigor3910Handler, image) == []


def test_rejects_invalid_optional_entry_name_length():
    image, _plaintexts = _build_image()
    name_offset = image.rfind(b"fw_ver")
    malformed = image[: name_offset - 4] + bytes(4) + image[name_offset:]

    assert find_chunks(DraytekVigor3910Handler, malformed) == []


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(_build_image(image_plaintext=b"")[0], id="required"),
        pytest.param(_build_image(boot_plaintext=b"")[0], id="optional"),
    ],
)
def test_rejects_empty_encrypted_entry(image: bytes):
    assert find_chunks(DraytekVigor3910Handler, image) == []


def test_rejects_invalid_decrypted_arm64_header():
    image, _plaintexts = _build_image(image_plaintext=bytes(80))

    assert find_chunks(DraytekVigor3910Handler, image) == []


@pytest.mark.parametrize(
    "truncate_at",
    [
        pytest.param(250, id="encrypted-image"),
        pytest.param(-20, id="optional-entry"),
    ],
)
def test_rejects_truncated_entry(truncate_at: int):
    image, _plaintexts = _build_image()
    if truncate_at < 0:
        truncate_at += len(image)

    assert find_chunks(DraytekVigor3910Handler, image[:truncate_at]) == []


def test_rejects_truncated_next_descriptor():
    image, _plaintexts = _build_image()

    assert find_chunks(DraytekVigor3910Handler, image + bytes(12)) == []


def test_extractor_decrypts_every_encrypted_entry(tmp_path: Path):
    image, plaintexts = _build_image()
    inpath = tmp_path / "firmware.all"
    outdir = tmp_path / "output"
    inpath.write_bytes(image)

    result = Vigor3910Extractor().extract(inpath, outdir)

    assert result.reports == []
    assert (outdir / "enc_Image.decrypted").read_bytes() == plaintexts["enc_Image"]
    assert (outdir / "enc_boot.img.decrypted").read_bytes() == plaintexts[
        "enc_boot.img"
    ]
    assert not (outdir / "fw_ver.decrypted").exists()


def test_parser_rejects_invalid_model_when_called_directly():
    image, _plaintexts = _build_image()
    malformed = image[:68] + b"v3911" + image[73:]

    with (
        File.from_bytes(malformed) as file,
        pytest.raises(InvalidInputFormat, match="Unsupported"),
    ):
        DraytekVigor3910Handler().calculate_chunk(file, 0)


@pytest.mark.parametrize("image_length", [60, 63])
def test_rejects_image_header_crossing_entry_boundary(image_length: int):
    image, _plaintexts = _build_image(image_plaintext=_arm64_image()[:image_length])

    assert find_chunks(DraytekVigor3910Handler, image) == []


def test_accepts_header_sized_image_and_short_other_encrypted_entry(tmp_path: Path):
    image, plaintexts = _build_image(
        image_plaintext=_arm64_image()[:64],
        boot_plaintext=b"x",
    )
    assert find_chunks(DraytekVigor3910Handler, image) == [(0, len(image))]

    inpath = tmp_path / "firmware.all"
    outdir = tmp_path / "output"
    inpath.write_bytes(image)
    result = Vigor3910Extractor().extract(inpath, outdir)

    assert result.reports == []
    assert (outdir / "enc_Image.decrypted").read_bytes() == plaintexts["enc_Image"]
    assert (outdir / "enc_boot.img.decrypted").read_bytes() == b"x"


@pytest.mark.parametrize("name", [b"nonce", b"enc_Image"])
def test_parser_rejects_truncated_required_descriptor(name: bytes):
    image, _plaintexts = _build_image()
    descriptor_offset = image.index(name) - 4 - DESCRIPTOR_LENGTH

    with (
        File.from_bytes(image[: descriptor_offset + DESCRIPTOR_LENGTH - 1]) as file,
        pytest.raises(EOFError),
    ):
        Vigor3910Parser.parse(file, 0)


@pytest.mark.parametrize(
    "descriptor_fields",
    [
        pytest.param({"flags": 1}, id="flags"),
        pytest.param({"reserved": b"\x01" + bytes(51)}, id="reserved"),
        pytest.param({"entry_length": DESCRIPTOR_LENGTH}, id="length"),
    ],
)
def test_invalid_optional_descriptor_ends_container(descriptor_fields):
    image, _plaintexts = _build_image()
    suffix = _entry(b"extra", b"data", **descriptor_fields)

    assert find_chunks(DraytekVigor3910Handler, image + suffix) == [(0, len(image))]
