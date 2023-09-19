#![feature(iterator_try_collect)]
use serde::{Serialize, Deserialize};
use std::fmt::Display;
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::str::FromStr;
use std::io;
use std::path::Path;
use std::string::String;
use serde::ser::StdError;

#[derive(Debug, PartialEq)]
enum StreamElementType {
    Int,
    Float,
}

#[derive(Debug, PartialEq)]
enum ParseStreamError {
    DtypeParseError,
    NoStreamDimension,
    InvalidStreamDimension,
}

impl Display for ParseStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DtypeParseError => write!(f, "Failed to read data type. Data type has to be 'int' or 'float"),
            Self::NoStreamDimension => write!(f, "No dimension of the data specified"),
            Self::InvalidStreamDimension => write!(f, "Having a dynamic inner dimension is not allowed"),
        }
    }
}

impl StdError for ParseStreamError {}

impl From<ParseStreamError> for io::Error {
    fn from(value: ParseStreamError) -> Self {
        io::Error::new(io::ErrorKind::Other, value)
    }
}

impl FromStr for StreamElementType {
    type Err = ParseStreamError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "int" {
            return Ok(Self::Int);
        } else if value == "float" {
            return Ok(Self::Float);
        } else {
            return Err(ParseStreamError::DtypeParseError);
        }
    }
}

#[derive(Debug, PartialEq)]
enum StreamDimensions {
    Static(Vec<u16>),
    Dynamic(Vec<u16>),
}

impl TryFrom<Vec<i16>> for StreamDimensions {
    type Error = ParseStreamError;

    fn try_from(value: Vec<i16>) -> Result<Self, Self::Error> {
        if value.len() < 1 {
            return Err(ParseStreamError::NoStreamDimension);
        }
        let mut converted_vec: Vec<u16> = Vec::with_capacity(value.len());
        for &elem in value[..value.len() - 1].into_iter() {
            if elem < 0 {
                return Err(ParseStreamError::InvalidStreamDimension);
            }
            converted_vec.push(elem as u16);
        }
        if value[value.len() - 1] > -1 {
            converted_vec.push(value[value.len() - 1] as u16);
            return Ok(Self::Static(converted_vec));
        } else {
            return Ok(Self::Dynamic(converted_vec));
        }
    }
}

struct Stream {
    name: Option<String>,
    shape: StreamDimensions,
    dtype: StreamElementType,
}

impl TryFrom<&RawStreamMetaData> for Stream {
    type Error = ParseStreamError;

    fn try_from(value: &RawStreamMetaData) -> Result<Self, Self::Error> {
        let name = &value.name;
        let shape = match &value.shape {
            Some(dim) => StreamDimensions::try_from(dim.clone())?,
            None => StreamDimensions::Dynamic(vec![1]),
        };
        let dtype = value.dtype.parse::<StreamElementType>()?;
        Ok(Stream {
            name: name.clone(), shape, dtype
        })
    }
}

impl Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct RawStreamMetaData {
    name: Option<String>,
    shape: Option<Vec<i16>>,
    dtype: String,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct RawStreamsMD {
    Metadata: Vec<RawStreamMetaData>,
}

struct StreamReader
{
    input: Box<dyn Iterator<Item= String>>,
    high_water_mark: u32,
    metadata: Vec<Stream>,
}

impl StreamReader {
    fn new(fname: &Path, high_water_mark: u32) -> Result<StreamReader, io::Error> {
        let file = File::open(fname)?;
        let freader = BufReader::new(file);
        let mut filtered_lines = read_lines_without_comments(freader)?;
        let metadata_str = read_until_data_section(&mut filtered_lines, 100)?;
        let raw_metadata: RawStreamsMD = match serde_yaml::from_str(&metadata_str){
            Ok(val) => val,
            Err(e) => return Err(io::Error::new(io::ErrorKind::Other, e)),
        };
        // TODO! We still need a way to return th
        let metadata: Vec<Stream> = raw_metadata.Metadata.iter().map(Stream::try_from).try_collect()?;
        Ok(StreamReader {
            input: Box::new(filtered_lines),
            high_water_mark,
            metadata,
        })
    }
}

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
pub fn read_lines_without_comments(file: BufReader<File>) -> Result<impl Iterator<Item = String>, io::Error> {
    Ok(file.lines().map(|line| line.unwrap_or("".to_string())).filter(|line| !line.starts_with("#")))
}

// read the metadata  section and return it as a string ready to be parsed into the Stream Meta
// Data structure
pub fn read_until_data_section(input: &mut impl Iterator<Item = String>, metadata_size: u32) -> Result<std::string::String, std::io::Error> {
    let mut metadata_string = String::new();
    let mut line_count = 0;
    for line in input {
        if line.starts_with("Data:") {
            metadata_string.push_str(&line);
            metadata_string.push('\n');
            return Ok(metadata_string);
        }
        line_count += 1;
        if line_count > metadata_size {
            return Err(std::io::Error::new(io::ErrorKind::Other, "Metadata line count exceeded")); 
        }
    }
    Err(std::io::Error::new(io::ErrorKind::Other, "No Data section found in the file")) 
}

#[cfg(test)]
mod tests {
    use crate::*;
    use std::fs::read_to_string;
    #[test]
    fn test_meta_with_name() {
        let input = 
"name: time
dtype: int
shape: [-1]
".to_string();
        let deser_meta_info: RawStreamMetaData = serde_yaml::from_str(&input).unwrap();
        let expected = RawStreamMetaData {name: Some("time".to_string()), shape: Some(vec![-1]), dtype: "int".to_string()};
        assert_eq!(deser_meta_info, expected);
    }

    #[test]
    fn test_meta_without_name() {
        let input = 
"dtype: float
shape: [3, 3]
".to_string();
        let deser_meta_info: RawStreamMetaData = serde_yaml::from_str(&input).unwrap();
        let expected = RawStreamMetaData {name: None, shape: Some(vec![3, 3]), dtype: "float".to_string()};
        assert_eq!(deser_meta_info, expected);
    }

    #[test]
    fn test_streams_meta_data() {
        let serialized_data = read_to_string("./tests/test_metadata.yaml").unwrap();
        let deser_data: RawStreamsMD = serde_yaml::from_str(&serialized_data).unwrap();
        let expected = RawStreamsMD{ 
            Metadata: vec![RawStreamMetaData {name: Some("time".to_string()), shape: Some(vec![1]), dtype: "float".to_string()}
                           ,RawStreamMetaData {name: Some("velocity".to_string()), shape: Some(vec![3]), dtype: "float".to_string()}
                           ,RawStreamMetaData {name: Some("probes".to_string()), shape: Some(vec![-1]), dtype: "int".to_string()}] };
        assert_eq!(deser_data, expected);
    }

    #[test]
    fn test_comment_filter() {
        let file = File::open("./tests/test_metadata_with_comments.yaml").unwrap();
        let file = BufReader::new(file);
        let lines_with_comments_filtered_out: String = read_lines_without_comments(file).unwrap().map(|line| line + "\n").collect();
        let expected = read_to_string("./tests/test_metadata.yaml").unwrap();
        assert_eq!(lines_with_comments_filtered_out, expected);
    }

    #[test]
    fn test_stream_dtype_parsing() {
        assert_eq!(Ok(StreamElementType::Int), "int".parse::<StreamElementType>()); 
        assert_eq!(Ok(StreamElementType::Float), "float".parse::<StreamElementType>()); 
    }
    #[test]
    #[should_panic]
    fn test_stream_dtype_parsing_fail() {
        assert_eq!(Ok(StreamElementType::Int), "this".parse::<StreamElementType>()); 
    }

    #[test]
    fn test_stream_dimension_parsing() {
        assert_eq!(StreamDimensions::Static(vec![1,1,2]), vec![1,1,2].try_into().unwrap());
        assert_eq!(StreamDimensions::Dynamic(vec![1,1,2]), vec![1,1,2, -1].try_into().unwrap());
    }

    #[test]
    #[should_panic]
    fn test_stream_dimension_parsing_fails() {
        let _stream_dim: StreamDimensions = vec![1,-1,4].try_into().unwrap();
    }

    #[test]
    fn test_raw_stream_metadata() {
    }
}
