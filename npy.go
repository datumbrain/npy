package npy

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// DType represents NumPy data types
type DType string

// NumPy data types
const (
	Bool    DType = "bool"
	Int8    DType = "int8"
	Int16   DType = "int16"
	Int32   DType = "int32"
	Int64   DType = "int64"
	Uint8   DType = "uint8"
	Uint16  DType = "uint16"
	Uint32  DType = "uint32"
	Uint64  DType = "uint64"
	Float32 DType = "float32"
	Float64 DType = "float64"
)

// Array represents a NumPy array with type parameter for data
type Array[T any] struct {
	Data    []T
	Shape   []int
	DType   DType
	Fortran bool // True if array is in Fortran order (column-major)
}

// header represents the metadata in a NumPy file
type header struct {
	Shape   []int
	DType   DType
	Fortran bool
}

// ReadFile reads a NumPy array from a .npy file with the specified type
func ReadFile[T any](path string) (*Array[T], error) {
	// Check file extension to ensure we're reading a .npy file
	if !strings.HasSuffix(path, ".npy") {
		return nil, fmt.Errorf("expected .npy file extension, got %s", path)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	return Read[T](f)
}

// WriteFile writes a NumPy array to a .npy file
func WriteFile[T any](path string, arr *Array[T]) error {
	// Ensure correct file extension
	if !strings.HasSuffix(path, ".npy") {
		path += ".npy" // Automatically add extension if missing
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer f.Close()

	return Write(f, arr)
}

// NPZFile represents a NumPy .npz file containing multiple arrays
type NPZFile struct {
	arrays map[string]interface{}
}

// NewNPZFile creates a new empty NPZ file
func NewNPZFile() *NPZFile {
	return &NPZFile{
		arrays: make(map[string]interface{}),
	}
}

// Add adds an array to the NPZ file
func Add[T any](npz *NPZFile, name string, arr *Array[T]) {
	npz.arrays[name] = arr
}

// Get retrieves an array from the NPZ file
func Get[T any](npz *NPZFile, name string) (*Array[T], bool) {
	val, ok := npz.arrays[name]
	if !ok {
		return nil, false
	}

	arr, ok := val.(*Array[T])
	return arr, ok
}

// Keys returns the names of all arrays in the NPZ file
func Keys(npz *NPZFile) []string {
	keys := make([]string, 0, len(npz.arrays))
	for k := range npz.arrays {
		keys = append(keys, k)
	}
	return keys
}

// ReadNPZFile reads multiple NumPy arrays from a .npz file
func ReadNPZFile(path string) (*NPZFile, error) {
	// Check file extension
	if !strings.HasSuffix(path, ".npz") {
		return nil, fmt.Errorf("expected .npz file extension, got %s", path)
	}

	// Open the zip file
	zipReader, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open NPZ file: %w", err)
	}
	defer zipReader.Close()

	// Create NPZ file
	npz := NewNPZFile()

	// Process each file in the zip
	for _, f := range zipReader.File {
		// Skip directories
		if f.FileInfo().IsDir() {
			continue
		}

		// Extract name
		name := f.Name
		name = strings.TrimSuffix(name, ".npy")

		// Open the file
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("failed to open file %s in NPZ: %w", f.Name, err)
		}

		// We need to determine the type of the array before we can read it
		// Since we can't know the type in advance, we'll read the header first to peek at the dtype
		// This is a bit hacky, but we don't have a better option with Go's type system
		// Read magic string and version
		magic := make([]byte, 6)
		if _, err := io.ReadFull(rc, magic); err != nil {
			rc.Close()
			return nil, fmt.Errorf("failed to read magic string from %s: %w", f.Name, err)
		}

		// Read version
		var major, minor uint8
		if err := binary.Read(rc, binary.LittleEndian, &major); err != nil {
			rc.Close()
			return nil, fmt.Errorf("failed to read major version from %s: %w", f.Name, err)
		}
		if err := binary.Read(rc, binary.LittleEndian, &minor); err != nil {
			rc.Close()
			return nil, fmt.Errorf("failed to read minor version from %s: %w", f.Name, err)
		}

		// Read header length
		var headerLen uint16
		if major == 1 {
			if err := binary.Read(rc, binary.LittleEndian, &headerLen); err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read header length from %s: %w", f.Name, err)
			}
		} else if major == 2 {
			var headerLen32 uint32
			if err := binary.Read(rc, binary.LittleEndian, &headerLen32); err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read header length from %s: %w", f.Name, err)
			}
			headerLen = uint16(headerLen32)
		} else {
			rc.Close()
			return nil, fmt.Errorf("unsupported version in %s: %d.%d", f.Name, major, minor)
		}

		// Read header
		headerBytes := make([]byte, headerLen)
		if _, err := io.ReadFull(rc, headerBytes); err != nil {
			rc.Close()
			return nil, fmt.Errorf("failed to read header from %s: %w", f.Name, err)
		}

		// Parse header
		hdr, err := parseHeader(string(headerBytes))
		if err != nil {
			rc.Close()
			return nil, fmt.Errorf("failed to parse header from %s: %w", f.Name, err)
		}

		// Close the reader - we'll reopen the file to read the full array with proper typing
		rc.Close()

		// Reopen the file
		rc, err = f.Open()
		if err != nil {
			return nil, fmt.Errorf("failed to reopen file %s in NPZ: %w", f.Name, err)
		}

		// Read array based on dtype
		var array interface{}
		switch hdr.DType {
		case Bool:
			arr, err := Read[bool](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read bool array from %s: %w", f.Name, err)
			}
			array = arr
		case Int8:
			arr, err := Read[int8](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read int8 array from %s: %w", f.Name, err)
			}
			array = arr
		case Int16:
			arr, err := Read[int16](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read int16 array from %s: %w", f.Name, err)
			}
			array = arr
		case Int32:
			arr, err := Read[int32](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read int32 array from %s: %w", f.Name, err)
			}
			array = arr
		case Int64:
			arr, err := Read[int64](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read int64 array from %s: %w", f.Name, err)
			}
			array = arr
		case Uint8:
			arr, err := Read[uint8](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read uint8 array from %s: %w", f.Name, err)
			}
			array = arr
		case Uint16:
			arr, err := Read[uint16](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read uint16 array from %s: %w", f.Name, err)
			}
			array = arr
		case Uint32:
			arr, err := Read[uint32](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read uint32 array from %s: %w", f.Name, err)
			}
			array = arr
		case Uint64:
			arr, err := Read[uint64](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read uint64 array from %s: %w", f.Name, err)
			}
			array = arr
		case Float32:
			arr, err := Read[float32](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read float32 array from %s: %w", f.Name, err)
			}
			array = arr
		case Float64:
			arr, err := Read[float64](rc)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to read float64 array from %s: %w", f.Name, err)
			}
			array = arr
		default:
			rc.Close()
			return nil, fmt.Errorf("unsupported dtype in %s: %s", f.Name, hdr.DType)
		}

		rc.Close()
		npz.arrays[name] = array
	}

	return npz, nil
}

// WriteNPZFile writes multiple NumPy arrays to a .npz file
func WriteNPZFile(path string, npz *NPZFile) error {
	// Ensure correct file extension
	if !strings.HasSuffix(path, ".npz") {
		path += ".npz" // Automatically add extension if missing
	}

	// Create the zip file
	zipFile, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create NPZ file: %w", err)
	}
	defer zipFile.Close()

	// Create zip writer
	zipWriter := zip.NewWriter(zipFile)
	defer zipWriter.Close()

	// Write each array to the zip
	for name, array := range npz.arrays {
		// Ensure name has .npy extension
		if !strings.HasSuffix(name, ".npy") {
			name += ".npy"
		}

		// Create file in zip
		w, err := zipWriter.Create(name)
		if err != nil {
			return fmt.Errorf("failed to create file %s in NPZ: %w", name, err)
		}

		// Write array based on type
		switch arr := array.(type) {
		case *Array[bool]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write bool array to %s: %w", name, err)
			}
		case *Array[int8]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write int8 array to %s: %w", name, err)
			}
		case *Array[int16]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write int16 array to %s: %w", name, err)
			}
		case *Array[int32]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write int32 array to %s: %w", name, err)
			}
		case *Array[int64]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write int64 array to %s: %w", name, err)
			}
		case *Array[uint8]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write uint8 array to %s: %w", name, err)
			}
		case *Array[uint16]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write uint16 array to %s: %w", name, err)
			}
		case *Array[uint32]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write uint32 array to %s: %w", name, err)
			}
		case *Array[uint64]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write uint64 array to %s: %w", name, err)
			}
		case *Array[float32]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write float32 array to %s: %w", name, err)
			}
		case *Array[float64]:
			if err := Write(w, arr); err != nil {
				return fmt.Errorf("failed to write float64 array to %s: %w", name, err)
			}
		default:
			return fmt.Errorf("unsupported array type in %s", name)
		}
	}

	return nil
}

// readData reads the actual data from the file based on the header information
func readData[T any](r io.Reader, hdr *header) ([]T, error) {
	// Calculate total number of elements
	totalElements := 1
	for _, dim := range hdr.Shape {
		totalElements *= dim
	}

	// Allocate slice for data
	data := make([]T, totalElements)

	// Read data
	if err := binary.Read(r, binary.LittleEndian, &data); err != nil {
		return nil, fmt.Errorf("failed to read data: %w", err)
	}

	return data, nil
}

// generateHeader creates a header string for a NumPy array
func generateHeader[T any](arr *Array[T]) string {
	// Map Go dtype to NumPy dtype
	var dtypeStr string
	switch arr.DType {
	case Bool:
		dtypeStr = "|b1"
	case Int8:
		dtypeStr = "|i1"
	case Int16:
		dtypeStr = "<i2"
	case Int32:
		dtypeStr = "<i4"
	case Int64:
		dtypeStr = "<i8"
	case Uint8:
		dtypeStr = "|u1"
	case Uint16:
		dtypeStr = "<u2"
	case Uint32:
		dtypeStr = "<u4"
	case Uint64:
		dtypeStr = "<u8"
	case Float32:
		dtypeStr = "<f4"
	case Float64:
		dtypeStr = "<f8"
	default:
		dtypeStr = "<f8" // Default to float64
	}

	// Format shape
	shapeStr := "("
	for i, dim := range arr.Shape {
		if i > 0 {
			shapeStr += ", "
		}
		shapeStr += strconv.Itoa(dim)
	}
	// Handle empty or single-dimensional arrays
	if len(arr.Shape) == 0 {
		shapeStr += ","
	} else if len(arr.Shape) == 1 {
		shapeStr += ","
	}
	shapeStr += ")"

	// Create header string
	fortranStr := "False"
	if arr.Fortran {
		fortranStr = "True"
	}

	return fmt.Sprintf("{'descr': '%s', 'fortran_order': %s, 'shape': %s, }", dtypeStr, fortranStr, shapeStr)
}

// Read reads a NumPy array from an io.Reader
func Read[T any](r io.Reader) (*Array[T], error) {
	// Read magic string and version
	magic := make([]byte, 6)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("failed to read magic string: %w", err)
	}
	if !bytes.Equal(magic, []byte("\x93NUMPY")) {
		return nil, fmt.Errorf("invalid magic string: %q", magic)
	}

	// Read version
	var major, minor uint8
	if err := binary.Read(r, binary.LittleEndian, &major); err != nil {
		return nil, fmt.Errorf("failed to read major version: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &minor); err != nil {
		return nil, fmt.Errorf("failed to read minor version: %w", err)
	}

	// Read header length
	var headerLen uint16
	if major == 1 {
		if err := binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
			return nil, fmt.Errorf("failed to read header length: %w", err)
		}
	} else if major == 2 {
		var headerLen32 uint32
		if err := binary.Read(r, binary.LittleEndian, &headerLen32); err != nil {
			return nil, fmt.Errorf("failed to read header length: %w", err)
		}
		headerLen = uint16(headerLen32)
	} else {
		return nil, fmt.Errorf("unsupported version: %d.%d", major, minor)
	}

	// Read header
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	// Parse header
	hdr, err := parseHeader(string(headerBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	// Read data
	data, err := readData[T](r, hdr)
	if err != nil {
		return nil, fmt.Errorf("failed to read data: %w", err)
	}

	return &Array[T]{
		Data:    data,
		Shape:   hdr.Shape,
		DType:   hdr.DType,
		Fortran: hdr.Fortran,
	}, nil
}

// Write writes a NumPy array to an io.Writer
func Write[T any](w io.Writer, arr *Array[T]) error {
	// Validate array
	if arr.Data == nil {
		return fmt.Errorf("array data is nil")
	}
	if arr.Shape == nil {
		return fmt.Errorf("array shape is nil")
	}
	if arr.DType == "" {
		return fmt.Errorf("array dtype is empty")
	}

	// Calculate total number of elements from shape
	totalElements := 1
	for _, dim := range arr.Shape {
		totalElements *= dim
	}

	// Validate data length
	if len(arr.Data) != totalElements {
		return fmt.Errorf("data length (%d) does not match shape dimensions (%d)", len(arr.Data), totalElements)
	}

	// Write magic string and version
	if _, err := w.Write([]byte("\x93NUMPY")); err != nil {
		return fmt.Errorf("failed to write magic string: %w", err)
	}

	// Write version (using v1.0)
	if err := binary.Write(w, binary.LittleEndian, uint8(1)); err != nil {
		return fmt.Errorf("failed to write major version: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, uint8(0)); err != nil {
		return fmt.Errorf("failed to write minor version: %w", err)
	}

	// Generate header
	headerStr := generateHeader(arr)

	// Header needs to be padded to be a multiple of 16 bytes (including the 10 byte file header)
	// for alignment purposes
	paddingLen := 16 - ((10 + len(headerStr)) % 16)
	if paddingLen < 1 {
		paddingLen += 16 // Ensure at least one padding char
	}

	headerStr += strings.Repeat(" ", paddingLen-1) + "\n"

	// Write header length
	if err := binary.Write(w, binary.LittleEndian, uint16(len(headerStr))); err != nil {
		return fmt.Errorf("failed to write header length: %w", err)
	}

	// Write header
	if _, err := w.Write([]byte(headerStr)); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write data
	if err := binary.Write(w, binary.LittleEndian, arr.Data); err != nil {
		return fmt.Errorf("failed to write data: %w", err)
	}

	return nil
}

// parseHeader parses a NumPy header string into a header struct
func parseHeader(headerStr string) (*header, error) {
	// Extract dictionary content from the header string
	re := regexp.MustCompile(`{.*}`)
	dictStr := re.FindString(headerStr)
	if dictStr == "" {
		return nil, fmt.Errorf("invalid header format")
	}

	// Extract shape
	shapeRe := regexp.MustCompile(`'shape':\s*\(([\d,\s]*)\)`)
	shapeMatch := shapeRe.FindStringSubmatch(dictStr)
	if len(shapeMatch) < 2 {
		return nil, fmt.Errorf("shape not found in header")
	}

	shapeStr := shapeMatch[1]
	shapeParts := strings.Split(shapeStr, ",")
	shape := make([]int, 0, len(shapeParts))
	for _, part := range shapeParts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		dim, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid shape dimension: %s", part)
		}
		shape = append(shape, dim)
	}

	// Extract dtype
	dtypeRe := regexp.MustCompile(`'descr':\s*'([^']*)'`)
	dtypeMatch := dtypeRe.FindStringSubmatch(dictStr)
	if len(dtypeMatch) < 2 {
		return nil, fmt.Errorf("dtype not found in header")
	}
	dtypeStr := dtypeMatch[1]

	// Extract endianness and map to Go data type
	var dtype DType
	if len(dtypeStr) >= 2 {
		typeChar := dtypeStr[1:]

		// Endianness doesn't matter for our Go representation
		// We'll use the native Go types and handle endianness during read/write
		switch typeChar {
		case "b1":
			dtype = Bool
		case "i1":
			dtype = Int8
		case "i2":
			dtype = Int16
		case "i4":
			dtype = Int32
		case "i8":
			dtype = Int64
		case "u1":
			dtype = Uint8
		case "u2":
			dtype = Uint16
		case "u4":
			dtype = Uint32
		case "u8":
			dtype = Uint64
		case "f4":
			dtype = Float32
		case "f8":
			dtype = Float64
		default:
			return nil, fmt.Errorf("unsupported dtype: %s", dtypeStr)
		}
	} else {
		return nil, fmt.Errorf("invalid dtype format: %s", dtypeStr)
	}

	// Extract fortran_order (column-major vs row-major)
	fortranRe := regexp.MustCompile(`'fortran_order':\s*(True|False)`)
	fortranMatch := fortranRe.FindStringSubmatch(dictStr)
	if len(fortranMatch) < 2 {
		return nil, fmt.Errorf("fortran_order not found in header")
	}
	fortran := fortranMatch[1] == "True"

	return &header{
		Shape:   shape,
		DType:   dtype,
		Fortran: fortran,
	}, nil
}
