//! Utilities useful for writing PC applications that use this engine + UI. Not directly related to
//! the 3D engine or GUI integration. Feature-gated.

use std::{
    fs::File,
    io::{self, ErrorKind, Read, Write},
    path::Path,
};

use bincode::{Decode, Encode};

/// Save to file, using Bincode. We currently use this for preference files.
pub fn save<T: Encode>(path: &Path, data: &T) -> io::Result<()> {
    let config = bincode::config::standard();

    let encoded: Vec<u8> = bincode::encode_to_vec(data, config).unwrap();

    let mut file = File::create(path)?;
    file.write_all(&encoded)?;
    Ok(())
}

/// Load from file, using Bincode. We currently use this for preference files.
pub fn load<T: Decode<()>>(path: &Path) -> io::Result<T> {
    let config = bincode::config::standard();

    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let (decoded, _len) = match bincode::decode_from_slice(&buffer, config) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Error loading from file. Did the format change?");
            return Err(io::Error::new(ErrorKind::Other, "error loading"));
        }
    };
    Ok(decoded)
}
