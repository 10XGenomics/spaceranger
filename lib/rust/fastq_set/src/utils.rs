// Copyright (c) 2018 10x Genomics, Inc. All rights reserved.
#![deny(missing_docs)]

//! Utility methods.

use anyhow::Error;
use flate2::write::GzEncoder;
use std::boxed::Box;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const GZ_BUF_SIZE: usize = 1 << 22;

/// Open a (possibly gzipped) file into a BufReader.
pub(crate) fn write_with_gz(p: &Path) -> Result<Box<dyn Write>, Error> {
    let w = File::create(p)?;

    let ext = p.extension().unwrap();

    if ext == "gz" {
        let gz = GzEncoder::new(w, flate2::Compression::fast());
        let buf_writer = BufWriter::with_capacity(GZ_BUF_SIZE, gz);
        Ok(Box::new(buf_writer))
    // disabling lz4 for now -- need to check on how to ensure all reads are flushed on drop.
    // } else if ext == "lz4" {
    //    let lz = lz4::Encoder::new(w)?;
    //    let buf_writer = BufWriter::with_capacity(GZ_BUF_SIZE, lz);
    //    Ok(Box::new(buf_writer))
    } else {
        let buf_writer = BufWriter::with_capacity(32 * 1024, w);
        Ok(Box::new(buf_writer))
    }
}
