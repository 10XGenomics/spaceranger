# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.

"""Functions to write to parquet file."""
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

BATCH_SIZE_DEFAULT = 10000  # Default batch size for Parquet batch writing


class ParquetBatchWriter:
    """A batch writer for Parquet files using PyArrow."""

    def __init__(
        self,
        file_path: str,
        schema: pa.Schema,
        batch_size: int = BATCH_SIZE_DEFAULT,
    ):
        """Initialize a new ParquetBatchWriter.

        Args:
            batch_size (int): Number of records to buffer before writing to disk.
            file_path (str): Path to the output Parquet file.
            schema (pa.Schema, optional): PyArrow schema defining the file structure.
                                          If None, it will be inferred from the first record.
            batch_size (int, optional): Size of the batches to write. Defaults to BATCH_SIZE_DEFAULT.
        """
        self.batch_size = batch_size
        self.file_path = file_path
        self.schema = schema
        self.buffer: list[dict[str, Any]] = []  # Temporary in-memory storage
        self.writer = None  # Will be initialized lazily
        self.closed = False  # Flag to track if the writer is closed

        if self.schema is not None:  # if a schema is provided we can initialize the writer
            self._initialize_writer()

    def _initialize_writer(self):
        """Lazy initialization of the Parquet writer."""
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.file_path, self.schema)

    def add_record(self, record: dict[str, Any]):
        """Add a record to the buffer and write a batch if buffer size reaches batch_size.

        Args:
            record (Dict[str, Any]): A dictionary representing a single row.
        """
        if self.schema is None:
            # Infer schema from the first record and initialize the writer
            table = pa.Table.from_pylist([record])
            self.schema = table.schema
            self._initialize_writer()

        # Ensure schema consistency
        if set(record.keys()) != set(self.schema.names):
            raise ValueError(
                f"Schema mismatch! Expected: {self.schema.names}, Got: {record.keys()}"
            )

        self.buffer.append(record)

        # Write batch if buffer is full
        if len(self.buffer) >= self.batch_size:
            self._write_batch()

    def _write_batch(self):
        """Writes the current buffer to the Parquet file and clears the buffer."""
        if not self.buffer:
            return

        self._initialize_writer()  # Ensure writer is initialized
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        self.writer.write_table(table)
        self.buffer.clear()

    def close(self):
        """Flushes remaining data and closes the Parquet writer."""
        if not self.closed:
            self._write_batch()
            if self.writer:
                self.writer.close()
            self.closed = True

    def __enter__(self):
        """Support for 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup when exiting the context."""
        self.close()
