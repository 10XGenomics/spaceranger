//!
//! Reader for DZI (Deep Zoom Image) format as described in
//! - <https://openseadragon.github.io/examples/tilesource-dzi/>
//! - <https://learn.microsoft.com/en-us/previous-versions/windows/silverlight/dotnet-windows-silverlight/cc645077(v=vs.95)>
//!
#![allow(missing_docs)]

use anyhow::{bail, Context, Result};
use image::{Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::ops::Range;
use std::path::{Path, PathBuf};
use zip::ZipArchive;

/// Struct to hold the `dzi_info.json`
#[derive(Debug, Serialize, Deserialize)]
pub struct DziInfo {
    #[serde(rename = "Image")]
    image: ImageInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageInfo {
    xmlns: String,
    #[serde(rename = "Format")]
    format: String,
    #[serde(rename = "Overlap")]
    overlap: String,
    #[serde(rename = "TileSize")]
    tile_size: String,
    #[serde(rename = "Size")]
    size: Size,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Size {
    #[serde(rename = "Height")]
    height: String,
    #[serde(rename = "Width")]
    width: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum BaseDziPath {
    Folder(PathBuf),
    ZipFile(PathBuf),
}

impl BaseDziPath {
    fn from_pipeline_out(folder: PathBuf) -> Result<Self> {
        let full_folder_path = folder.join("dzi_tiles_paths");
        let full_zip_path = folder.join("dzi_tiles_zip.zip");
        if full_folder_path.exists() && full_folder_path.is_dir() {
            Ok(Self::Folder(full_folder_path))
        } else if full_zip_path.exists() && full_zip_path.is_file() {
            Ok(Self::ZipFile(full_zip_path))
        } else {
            bail!("Could not find DZI folder or zip file!")
        }
    }
}
/// Reader for a specific tile in the DZI image
pub struct TileReader {
    w: usize,
    h: usize,
    tile_size: usize,
    overlap: usize,
    image: RgbImage,
    raw_bytes: Vec<u8>,
}

impl TileReader {
    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw_bytes
    }
    pub fn non_overlapping_image(&self) -> RgbImage {
        let start_w = if self.w == 0 { 0 } else { self.overlap };
        let start_h = if self.h == 0 { 0 } else { self.overlap };

        let end_w = (start_w + self.tile_size).min(self.image.width() as usize);
        let end_h = (start_h + self.tile_size).min(self.image.height() as usize);

        let mut image = RgbImage::new((end_w - start_w) as u32, (end_h - start_h) as u32);

        for y in start_h..end_h {
            let yt = y - start_h;
            for x in start_w..end_w {
                let xt = x - start_w;
                image.put_pixel(
                    xt as u32,
                    yt as u32,
                    *self.image.get_pixel(x as u32, y as u32),
                );
            }
        }
        image
    }
    /// Iterate over the non overlapping pixels in this tile.
    /// Returns an iterator of (x, y, pixel) where (x, y) are the coordinates of the pixel in the
    /// fully tiled image at this level.
    fn iter_non_overlaping(
        &self,
        pixel_range: &PixelRange,
    ) -> impl Iterator<Item = (usize, usize, &Rgb<u8>)> {
        let (offset_w, offset_h) = (self.w * self.tile_size, self.h * self.tile_size);

        // No overlap for tiles that start at the top/left edges
        let skip_overlap_w = if self.w == 0 { 0 } else { self.overlap };
        let skip_overlap_h = if self.h == 0 { 0 } else { self.overlap };

        let start_w = (offset_w.max(pixel_range.x.start) - offset_w) + skip_overlap_w;
        let start_h = (offset_h.max(pixel_range.y.start) - offset_h) + skip_overlap_h;

        let end_w = ((offset_w + self.tile_size)
            .min(pixel_range.x.end)
            .saturating_sub(offset_w)
            + skip_overlap_w)
            .min(self.image.width() as usize);
        let end_h = ((offset_h + self.tile_size)
            .min(pixel_range.y.end)
            .saturating_sub(offset_h)
            + skip_overlap_h)
            .min(self.image.height() as usize);

        (start_h..end_h).flat_map(move |y| {
            (start_w..end_w).map(move |x| {
                (
                    offset_w + x - skip_overlap_w,
                    offset_h + y - skip_overlap_h,
                    self.image.get_pixel(x as u32, y as u32),
                )
            })
        })
    }
}

#[derive(Debug)]
pub struct DziReader {
    tile_size: usize,
    width: usize,
    height: usize,
    overlap: usize,
    tiles_path: BaseDziPath,
    relative_path: PathBuf,
    max_level: usize,
}

pub enum Level {
    Highest,
}

// A rectangular pixel range
#[derive(Debug, Clone)]
pub struct PixelRange {
    pub x: Range<usize>,
    pub y: Range<usize>,
}

impl PixelRange {
    pub fn nrows(&self) -> usize {
        self.height()
    }
    pub fn ncols(&self) -> usize {
        self.width()
    }
    pub fn width(&self) -> usize {
        self.x.end.saturating_sub(self.x.start)
    }
    pub fn height(&self) -> usize {
        self.y.end.saturating_sub(self.y.start)
    }
    pub fn square(self) -> Self {
        let width = self.width();
        let height = self.height();
        let max_dim = width.max(height);

        let x_start = self.x.start.saturating_sub((max_dim - width) / 2);
        let y_start = self.y.start.saturating_sub((max_dim - height) / 2);

        PixelRange {
            x: x_start..x_start + max_dim,
            y: y_start..y_start + max_dim,
        }
    }
    pub fn clamp(&self, width: usize, height: usize) -> Self {
        Self {
            x: self.x.start..self.x.end.min(width),
            y: self.y.start..self.y.end.min(height),
        }
    }
    fn relative_index(&self, x: usize, y: usize) -> (usize, usize) {
        assert!(self.x.contains(&x));
        assert!(self.y.contains(&y));
        (x - self.x.start, y - self.y.start)
    }
    fn buffer(&self) -> RgbImage {
        RgbImage::new(self.width() as u32, self.height() as u32)
    }
}

impl DziReader {
    // Create a reader from a folder that contains dzi_info.json and dzi_tiles_paths/
    pub fn new(folder: impl AsRef<Path>, channel: Option<u32>) -> Result<Self> {
        let folder = folder.as_ref();
        let info_json = std::fs::read_to_string(folder.join("dzi_info.json"))
            .with_context(|| format!("While reading dzi_info.json inside {folder:?}"))?;
        let info: DziInfo = serde_json::from_str(&info_json)?;
        let tile_size: usize = info.image.tile_size.parse()?;

        let width: usize = info.image.size.width.parse()?;
        let height: usize = info.image.size.height.parse()?;

        let tiles_path = BaseDziPath::from_pipeline_out(folder.to_path_buf())?;
        let relative_path = PathBuf::from(channel.unwrap_or(0).to_string());

        let max_level = (width.max(height) as f64).log2().ceil() as usize;

        Ok(DziReader {
            overlap: info.image.overlap.parse()?,
            tile_size,
            tiles_path,
            relative_path,
            max_level,
            width,
            height,
        })
    }
    fn tile_num(&self, pixel: usize) -> usize {
        pixel / self.tile_size
    }

    pub fn max_level(&self) -> usize {
        self.max_level
    }

    pub fn min_level(&self) -> usize {
        (self.tile_size as f64).log2().ceil() as usize
    }

    pub fn tile_size(&self) -> usize {
        self.tile_size
    }

    pub fn overlap(&self) -> usize {
        self.overlap
    }

    fn tile_range(&self, Range { start, end }: Range<usize>) -> Range<usize> {
        self.tile_num(start)..(self.tile_num(end) + 1)
    }

    pub fn read_tile(
        &self,
        level: usize,
        w_tile: usize,
        h_tile: usize,
    ) -> Result<Option<TileReader>> {
        let full_relative_path = self
            .relative_path
            .join(level.to_string())
            .join(format!("{w_tile}_{h_tile}.png"));
        let raw_bytes = match &self.tiles_path {
            BaseDziPath::Folder(tiles_path) => {
                let path = tiles_path.join(full_relative_path);
                if !path.exists() {
                    None
                } else {
                    Some(std::fs::read(path)?)
                }
            }
            BaseDziPath::ZipFile(zip_path) => {
                let mut zip_archive = ZipArchive::new(BufReader::new(File::open(zip_path)?))?;
                let zip_file = zip_archive.by_name(full_relative_path.to_str().unwrap());
                if let Ok(mut zip_file) = zip_file {
                    let mut tile_bytes = vec![];
                    let _ = zip_file.read_to_end(&mut tile_bytes)?;
                    Some(tile_bytes)
                } else {
                    None
                }
            }
        };

        match raw_bytes {
            Some(raw_bytes) => Ok(Some(TileReader {
                w: w_tile,
                h: h_tile,
                tile_size: self.tile_size,
                overlap: self.overlap,
                image: image::ImageReader::new(Cursor::new(&raw_bytes))
                    .with_guessed_format()?
                    .decode()?
                    .into_rgb8(),
                raw_bytes,
            })),
            None => Ok(None),
        }
    }

    /// Read a range of pixels from the full resolution image
    pub fn read_full_res_image(&self, range: &PixelRange) -> Result<RgbImage> {
        let range = range.clamp(self.width, self.height);
        let mut buffer = range.buffer();
        let w_tile_range = self.tile_range(range.x.clone());
        let h_tile_range = self.tile_range(range.y.clone());

        for w_tile in w_tile_range {
            for h_tile in h_tile_range.clone() {
                if let Some(tile) = self.read_tile(self.max_level, w_tile, h_tile)? {
                    for (x, y, pixel) in tile.iter_non_overlaping(&range) {
                        let (x, y) = range.relative_index(x, y);
                        buffer.put_pixel(x as u32, y as u32, *pixel);
                    }
                } else {
                    unreachable!("{w_tile} {h_tile}");
                }
            }
        }

        Ok(buffer)
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
}
