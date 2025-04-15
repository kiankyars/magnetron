### File Format
The .mag file format is a cross-platform binary format for storing networks and tensors.
The format allows to store tensors identified by a string key and metadata of various data types.
Memory mapping support allows for very fast loading of large files.
All data is always stored in little endian format. If the host machine is big endian, the data will be converted to little endian format on load.

### File Structure
The file is structured as follows:
- **Header**: Contains metadata about the file, including the version number and the number of tensors.
- **Metadata**: Contains metadata for each tensor, including the shape, data type, and any additional attributes.
- **Tensors**: Each tensor is stored in a separate section, with its own metadata and data.