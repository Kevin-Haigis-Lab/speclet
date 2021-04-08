#!/usr/bin/env python3

from pathlib import Path

from src.io import data_io


def test_root_dir():
    assert data_io.project_root_dir().exists()
    assert data_io.project_root_dir().is_dir()


def test_modeling_data_dir():
    assert data_io.modeling_data_dir().exists()
    assert data_io.modeling_data_dir().is_dir()


def test_files_exist():
    p = data_io.data_path(to=data_io.DataFile.crc_subsample)
    assert isinstance(p, Path)
    assert p.exists()
    assert p.is_file()
