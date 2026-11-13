from pathlib import Path

import pytest

from unblob.handlers.filesystem.cramfs import (
    FSID_CRC_OFFSET,
    FSID_CRC_SIZE,
    CramFSHandler,
)
from unblob.testing import find_chunks

CRAMFS_TEST_DATA = Path(__file__).parents[2] / "integration/filesystem/cramfs"


def swap_stored_crc(path: Path) -> bytes:
    content = bytearray(path.read_bytes())
    crc_end = FSID_CRC_OFFSET + FSID_CRC_SIZE
    content[FSID_CRC_OFFSET:crc_end] = reversed(content[FSID_CRC_OFFSET:crc_end])
    return bytes(content)


@pytest.mark.parametrize(
    "fixture_path",
    [
        pytest.param(
            CRAMFS_TEST_DATA / "big_endian/__input__/fruits.cramfs_be",
            id="big_endian",
        ),
        pytest.param(
            CRAMFS_TEST_DATA / "little_endian/__input__/fruits.cramfs_le",
            id="little_endian",
        ),
    ],
)
def test_extracts_swapped_crc_without_modifying_input(
    tmp_path: Path, fixture_path: Path
):
    content = swap_stored_crc(fixture_path)
    assert find_chunks(CramFSHandler, content) == [(0, len(content))]

    inpath = tmp_path / fixture_path.name
    inpath.write_bytes(content)
    outdir = tmp_path / "extract"
    outdir.mkdir()

    CramFSHandler.EXTRACTOR.extract(inpath, outdir)

    assert inpath.read_bytes() == content
    assert (outdir / "apple.txt").read_text() == "apple\n"
    assert (outdir / "cherry.txt").read_text() == "cherry\n"
    assert not list(tmp_path.glob(".unblob-cramfs-*"))
