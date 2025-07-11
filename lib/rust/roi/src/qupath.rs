#![allow(missing_docs)]
use anyhow::{Context, Result};
use geo::{coord, Area, BoundingRect, MapCoords};
use geo_types::{Coord, Rect};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::fs::{read_to_string, write};
use std::hash::{Hash, Hasher};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct NamedPolygon {
    #[serde(
        deserialize_with = "geojson::de::deserialize_geometry",
        serialize_with = "geojson::ser::serialize_geometry"
    )]
    pub geometry: geo_types::Polygon<f64>,
    pub name: String,
}

impl Hash for NamedPolygon {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.name.hash(state);
        self.to_ordered_float_polygon().hash(state);
    }
}

impl PartialEq for NamedPolygon {
    fn eq(&self, other: &Self) -> bool {
        self.name.eq(&other.name)
            && self
                .to_ordered_float_polygon()
                .eq(&other.to_ordered_float_polygon())
    }
}

impl Eq for NamedPolygon {}

fn _sanitise_name(name: &str) -> String {
    let parts = name.split('_').collect::<Vec<_>>();
    if parts.len() < 2 {
        String::from(name)
    } else {
        let prefix = parts[..parts.len() - 1].join("_");
        let suffix = parts[parts.len() - 1].to_lowercase();
        format!("{prefix}_{suffix}")
    }
}

const MIN_AREA: f64 = 16.0;

/// Writes polygons with sequential names starting from 1
pub fn write_geojson_collection_sequential_names(
    polygons: &[NamedPolygon],
    fname: &Path,
) -> Result<()> {
    #[derive(Serialize)]
    struct NamedCollection {
        #[serde(serialize_with = "geojson::ser::serialize_geometry")]
        geometry: geo_types::Polygon<f64>,
        cell_id: usize,
    }
    let polygon_str = geojson::ser::to_feature_collection_string(
        &polygons
            .iter()
            .enumerate()
            .map(|(ind, polygon)| NamedCollection {
                geometry: polygon.geometry.clone(),
                cell_id: (ind + 1),
            })
            .collect::<Vec<_>>(),
    )?;

    write(fname, polygon_str)?;

    Ok(())
}

impl NamedPolygon {
    pub fn to_ordered_float_polygon(&self) -> geo::Polygon<OrderedFloat<f64>> {
        geo::Polygon::new(
            self.geometry
                .exterior()
                .coords()
                .map(|&coord| coord! {x:OrderedFloat(coord.x), y:OrderedFloat(coord.y)})
                .collect(),
            self.geometry
                .interiors()
                .iter()
                .map(|line_string| {
                    line_string
                        .coords()
                        .map(|&coord| coord! {x:OrderedFloat(coord.x), y:OrderedFloat(coord.y)})
                        .collect()
                })
                .collect(),
        )
    }

    pub fn load_geojson_collection(fname: &Path) -> Result<Vec<Self>> {
        #[derive(Deserialize)]
        struct NamedCollection {
            #[serde(deserialize_with = "geojson::de::deserialize_geometry")]
            geometry: geo_types::Geometry<f64>,
            name: Option<String>,
        }

        let collection: Vec<NamedCollection> =
            geojson::de::deserialize_feature_collection_str_to_vec(
                &read_to_string(fname).with_context(|| format!("While reading {fname:?}"))?,
            )?;

        Ok(collection
            .into_iter()
            .enumerate()
            .map(|(ind, c)| NamedPolygon {
                name: c.name.map_or_else(
                    || format!("CellID{ind:0>9}"),
                    |x| _sanitise_name(x.as_str()),
                ),
                geometry: match c.geometry {
                    geo_types::Geometry::Polygon(p) => p,

                    geo_types::Geometry::MultiPolygon(m) => m
                        .into_iter()
                        // These are mostly a results of manual errors while segmenting in QuPath
                        .filter(|p| p.unsigned_area() > MIN_AREA)
                        .exactly_one()
                        .unwrap(),
                    c => panic!("Unexpected geometry {c:?}"),
                },
            })
            .collect())
    }

    pub fn bounding_box(&self) -> Rect {
        self.geometry
            .bounding_rect()
            .expect("Failed to compute bounding box")
    }

    pub fn map_coords(&self, f: impl Fn(Coord) -> Coord + Copy) -> Self {
        NamedPolygon {
            geometry: self.geometry.map_coords(f),
            name: self.name.clone(),
        }
    }

    /// Computes area of the named polygon
    pub fn area(&self) -> f64 {
        self.geometry.unsigned_area()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::Coord;
    use tempfile::tempdir;

    #[test]
    fn test_sanitise_names() {
        assert_eq!(
            _sanitise_name("ROI_SouRce_sourCe123"),
            "ROI_SouRce_source123"
        );
        assert_eq!(_sanitise_name("SourCe123"), "SourCe123");
        assert_eq!(_sanitise_name("_SourCe123"), "_source123");
    }

    #[test]
    fn test_load_geojson() {
        let polygons =
            NamedPolygon::load_geojson_collection(Path::new("test/test.geojson")).unwrap();
        assert_eq!(polygons.len(), 2);
        assert_eq!(polygons[0].name, "ROI1_source");
        assert!(polygons[0].geometry.interiors().is_empty());
        assert_eq!(polygons[0].geometry.exterior().0.len(), 38);
        assert_eq!(
            polygons[0].geometry.exterior().0[0],
            Coord {
                x: 11398.0,
                y: 3015.0
            }
        );
        assert_eq!(polygons[1].name, "ROI1_mask");
        assert_eq!(polygons[1].geometry.exterior().0.len(), 40);
        assert_eq!(
            polygons[1].geometry.exterior().0[10],
            Coord {
                x: 9833.0,
                y: 3644.0
            }
        );
    }

    #[test]
    fn test_load_geojson2() -> Result<()> {
        // This file has amultipolygon source. One of the polygons is tiny, likely a user-click error.
        let polygons =
            NamedPolygon::load_geojson_collection(Path::new("test/test2.geojson")).unwrap();
        assert_eq!(polygons.len(), 2);
        Ok(())
    }

    #[test]
    fn test_load_geojson3() -> Result<()> {
        let fpath = Path::new("test/qpath_segmentations.geojson");
        let polygons = NamedPolygon::load_geojson_collection(fpath)?;
        assert_eq!(polygons.len(), 53);
        Ok(())
    }

    #[test]
    fn test_load_and_save_geojson() -> Result<()> {
        let fpath = Path::new("test/qpath_segmentations.geojson");
        let polygons = NamedPolygon::load_geojson_collection(fpath)?;
        let dir = tempdir()?;
        let file_path = dir.path().join("my-temporary-polygons.geojson");
        write_geojson_collection_sequential_names(&polygons, &file_path)?;

        #[derive(Deserialize)]
        struct NamedCollection {
            #[serde(deserialize_with = "geojson::de::deserialize_geometry")]
            geometry: geo_types::Polygon<f64>,
            cell_id: usize,
        }

        let polygons_reloaded: Vec<NamedCollection> =
            geojson::de::deserialize_feature_collection_str_to_vec(
                &read_to_string(&file_path)
                    .with_context(|| format!("While reading temporary {file_path:?}"))?,
            )?;

        for (ind, (polygon, polygon_reloaded)) in
            polygons.iter().zip(polygons_reloaded.iter()).enumerate()
        {
            assert_eq!(polygon.geometry, polygon_reloaded.geometry);
            assert_eq!((ind + 1), polygon_reloaded.cell_id);
        }

        dir.close()?;
        Ok(())
    }
}
