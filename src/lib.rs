#![feature(iterator_try_collect)]
use std::io::BufRead;
use serde::{Serialize, Deserialize};
use std::fmt::Display;
use std::fs::File;
use std::str::FromStr;
use std::io;
use std::path::Path;
use std::string::String;
use std::num::ParseFloatError;
use serde::ser::StdError;
use ndarray::prelude::*;

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
    WrongElementCount,
    ElementParseError,
}

impl Display for ParseStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DtypeParseError => write!(f, "Failed to read data type. Data type has to be 'int' or 'float"),
            Self::NoStreamDimension => write!(f, "No dimension of the data specified"),
            Self::InvalidStreamDimension => write!(f, "Having a dynamic inner dimension is not allowed"),
            Self::WrongElementCount => write!(f, "The stream content did not have the same amount of elements as where specified by the shape of the array"),
            Self::ElementParseError => write!(f, "It was not possible to parse an element in the stream data into a number"),
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

impl From<ParseFloatError> for ParseStreamError {
    fn from(value: ParseFloatError) -> Self {
        ParseStreamError::ElementParseError
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

pub struct StreamReader
{
    input: Box<dyn Iterator<Item= io::Result<String>>>,
    high_water_mark: u32,
    metadata: Vec<Stream>,
    buffers: Vec<Box<[ArrayD<f32>]>>,
}

impl StreamReader {
    pub fn new(fname: &Path, high_water_mark: u32) -> Result<StreamReader, io::Error> {
        let lines = build_line_iterator(fname)?;
        let mut filtered_lines = filter_out_comment_lines('#', lines);
        let metadata_str = read_until_line_starts_with("Data:", 92, &mut filtered_lines)?;
        let raw_metadata: RawStreamsMD = match serde_yaml::from_str(&metadata_str){
            Ok(val) => val,
            Err(e) => return Err(io::Error::new(io::ErrorKind::Other, e)),
        };
        let metadata: Vec<Stream> = raw_metadata.Metadata.iter().map(Stream::try_from).try_collect()?;
        let stream_buffers: Vec<Box<[ArrayD<f32>]> = metadata.iter().map(|m| Box::new([ArrayD; m.shape.iter().product()])); 
        Ok(StreamReader {
            input: Box::new(filtered_lines),
            high_water_mark,
            metadata,
            buffers: Box::new([Box::new([Box::new([metadata.shape.iter().product()]); high_water_mark]); metadata.len()])
        })
    }
}


/// Given a path to a file, generate an iterator that returns the lines of a file.
pub fn build_line_iterator<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where 
    P: AsRef<Path>
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

/// filter out lines that start with the `comment_char`
pub fn filter_out_comment_lines(comment_char: char, lines: impl Iterator<Item = io::Result<String>>) -> impl Iterator<Item = io::Result<String>> {
    lines.filter(
        move |l| l.as_ref().is_ok_and(|line| !line.starts_with(comment_char)) || l.is_err()
    )
}

/// Accumulate the lines of a file until the start of a line matches `matchstr`.
/// This method borrows the line iterator, so that when the function terminates the iterator is
/// left at the first line after the line that matched.
pub fn read_until_line_starts_with<'a>(matchstr: &'a str, max_lines: i32, lines: &'a mut impl Iterator<Item = io::Result<String>>) -> io::Result<String> {
    let mut res = String::new();
    let mut counter = 1;
    while let Some(line) = lines.next() {
        if counter > max_lines {
            break;
        }
        match line {
            Ok(l) => if l.starts_with(matchstr) { 
                break 
            } else { 
                res.push_str(&l);
                res.push('\n');
            },
            Err(e) => return Err(e),
        }
    }
    Ok(res)
}

/// Given a String that represents a line of the data input 
pub fn split_line_into_streams<'a>(stream_delimiter: char, line: &'a String) -> Vec<Vec<&'a str>> {
    line.split(stream_delimiter).map(|s| s.split(&[',', ' ']).filter(|ss| ss.len() != 0).collect()).collect()
}

/// given the elements that where found in the stream, assemble them into an ndArray
pub fn build_array<'a>(shape: &Vec<usize>, data: &'a Vec<&'a str>) -> Result<ArrayD<f32>, ParseStreamError> {
    let parsed_data: Vec<f32> = data.iter().map(|s| s.parse::<f32>()).try_collect()?;
    if parsed_data.len() != shape.iter().map(|s| *s as usize).product() {
        return Err(ParseStreamError::WrongElementCount)
    }
    Ok(ArrayD::<f32>::from_shape_vec(IxDyn(&shape[..]), parsed_data).unwrap())
}

/// Read the first n lines of a file that is wrapped in a BufReader
/// This function alters the iterator that represents the current location in the file, leaving it
/// at line n+1. The first n lines are concatinated into a string that is returned at the end of
/// the function
fn read_n_lines<'a>(n: u32, lines: &'a mut impl Iterator<Item = io::Result<String>>) -> io::Result<String> {
    let mut res = String::new();
    for _i in 0..n {
        let line = lines.next();
        match line {
            Some(Ok(l)) => {
                res.push_str(&l);
                res.push('\n');
            }
            Some(Err(e)) => return Err(e),
            None => break,
        }
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use crate::*;
    use std::io::BufReader;
    use std::io::Cursor;
    use std::fs::File;
    use std::io::BufRead;
    use std::iter::zip;
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
    fn test_read_until_data_section() {
        let mut test_input_line_iter = BufReader::new(File::open("../tests/test_read_until_data_section")).lines();
        let metadata_str = read_until_data_section(&mut test_input_line_iter, 100).unwrap();
        assert_eq!(test_input_line_iter.next().unwrap(), "-1");
    }
    #[test]
    fn test_read_n_lines() {
        let mut file = BufReader::new(File::open("tests/read_n_lines_test").unwrap()).lines(); 
        let first_three_lines = read_n_lines(3, &mut file).unwrap();
        assert_eq!(first_three_lines, "1\n2\n3\n");
        let Some(Ok(fourth_line)) = file.next() else {panic!("reading fourth line failed")};
        assert_eq!(fourth_line, "4");
    }

    #[test]
    fn test_read_n_lines_from_string() {
        let mut string_iter = Cursor::new("1\n2\n3\n4\n".to_string()).lines();
        let first_three_lines = read_n_lines(3, &mut string_iter).unwrap();
        assert_eq!(first_three_lines, "1\n2\n3\n");
        let Some(Ok(fourth_line)) = string_iter.next() else {panic!("reading fourth line failed")};
        assert_eq!(fourth_line, "4");
    }

    #[test]
    fn test_read_too_many_lines() {
        let mut string_iter = Cursor::new("1\n2\n3\n4\n").lines();
        let first_six_lines = read_n_lines(6, &mut string_iter).unwrap();
        assert_eq!(first_six_lines, "1\n2\n3\n4\n");
    }

    #[test]
    fn test_filter_out_lines_from_input() {
        let string_iter = Cursor::new("1\n#2\n3\n4\n#5\n6\n").lines();
        let mut filtered_iter = filter_out_comment_lines('#', string_iter);
        let first_three_lines_uncommented = read_n_lines(3, &mut filtered_iter).unwrap();
        assert_eq!(first_three_lines_uncommented, "1\n3\n4\n");
    }
    
    #[test]
    fn test_read_until_data_start() {
        let mut file = BufReader::new(File::open("tests/read_metadata_section").unwrap()).lines();
        let metadata_section = read_until_line_starts_with("Data:", &mut file).unwrap();
        assert_eq!(metadata_section, "Metadata:\n  - name: Stream1\n    dtype: int\n    shape: [1]\n");
        let next_line = file.next().unwrap().unwrap();
        assert_eq!(next_line, "1");
    }

    #[test]
    fn test_stream_spltting() {
        let line = "1, 2 3, 5.15, | 6 7, 8".to_string();
        let reference = vec![vec!["1", "2", "3", "5.15"], vec!["6", "7", "8"]];
        let numbers: Vec<Vec<&str>> = split_line_into_streams('|', &line);
        for content in zip(numbers, reference) {
            let values = zip(content.0, content.1);
            for (v, r) in values {
                assert_eq!(v, r);
            }
        }
    }
}
