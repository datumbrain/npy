package npy

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// TestWriteReadFloat64 tests writing and reading a float64 array
func TestWriteReadFloat64(t *testing.T) {
	// Create test array
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	shape := []int{2, 3}
	arr := &Array[float64]{
		Data:    data,
		Shape:   shape,
		DType:   Float64,
		Fortran: false,
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Write array to file
	filePath := filepath.Join(tempDir, "test_float64.npy")
	if err := WriteFile(filePath, arr); err != nil {
		t.Fatalf("Failed to write array: %v", err)
	}

	// Read array from file
	readArr, err := ReadFile[float64](filePath)
	if err != nil {
		t.Fatalf("Failed to read array: %v", err)
	}

	// Verify data
	if !reflect.DeepEqual(readArr.Data, arr.Data) {
		t.Errorf("Data mismatch. Got %v, want %v", readArr.Data, arr.Data)
	}

	// Verify shape
	if !reflect.DeepEqual(readArr.Shape, arr.Shape) {
		t.Errorf("Shape mismatch. Got %v, want %v", readArr.Shape, arr.Shape)
	}

	// Verify dtype
	if readArr.DType != arr.DType {
		t.Errorf("DType mismatch. Got %v, want %v", readArr.DType, arr.DType)
	}

	// Verify Fortran order
	if readArr.Fortran != arr.Fortran {
		t.Errorf("Fortran order mismatch. Got %v, want %v", readArr.Fortran, arr.Fortran)
	}
}

// TestWriteReadInt32 tests writing and reading an int32 array
func TestWriteReadInt32(t *testing.T) {
	// Create test array
	data := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	shape := []int{3, 3}
	arr := &Array[int32]{
		Data:    data,
		Shape:   shape,
		DType:   Int32,
		Fortran: false,
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Write array to file
	filePath := filepath.Join(tempDir, "test_int32.npy")
	if err := WriteFile(filePath, arr); err != nil {
		t.Fatalf("Failed to write array: %v", err)
	}

	// Read array from file
	readArr, err := ReadFile[int32](filePath)
	if err != nil {
		t.Fatalf("Failed to read array: %v", err)
	}

	// Verify data
	if !reflect.DeepEqual(readArr.Data, arr.Data) {
		t.Errorf("Data mismatch. Got %v, want %v", readArr.Data, arr.Data)
	}
}

// TestWriteReadBool tests writing and reading a boolean array
func TestWriteReadBool(t *testing.T) {
	// Create test array
	data := []bool{true, false, true, false}
	shape := []int{2, 2}
	arr := &Array[bool]{
		Data:    data,
		Shape:   shape,
		DType:   Bool,
		Fortran: false,
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Write array to file
	filePath := filepath.Join(tempDir, "test_bool.npy")
	if err := WriteFile(filePath, arr); err != nil {
		t.Fatalf("Failed to write array: %v", err)
	}

	// Read array from file
	readArr, err := ReadFile[bool](filePath)
	if err != nil {
		t.Fatalf("Failed to read array: %v", err)
	}

	// Verify data
	if !reflect.DeepEqual(readArr.Data, arr.Data) {
		t.Errorf("Data mismatch. Got %v, want %v", readArr.Data, arr.Data)
	}
}

// TestEmptyArray tests writing and reading an empty array
func TestEmptyArray(t *testing.T) {
	// Create test array
	data := []float64{}
	shape := []int{0}
	arr := &Array[float64]{
		Data:    data,
		Shape:   shape,
		DType:   Float64,
		Fortran: false,
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Write array to file
	filePath := filepath.Join(tempDir, "test_empty.npy")
	if err := WriteFile(filePath, arr); err != nil {
		t.Fatalf("Failed to write array: %v", err)
	}

	// Read array from file
	readArr, err := ReadFile[float64](filePath)
	if err != nil {
		t.Fatalf("Failed to read array: %v", err)
	}

	// Verify data
	if !reflect.DeepEqual(readArr.Data, arr.Data) {
		t.Errorf("Data mismatch. Got %v, want %v", readArr.Data, arr.Data)
	}

	// Verify shape
	if !reflect.DeepEqual(readArr.Shape, arr.Shape) {
		t.Errorf("Shape mismatch. Got %v, want %v", readArr.Shape, arr.Shape)
	}
}

// TestFortranOrder tests writing and reading an array in Fortran (column-major) order
func TestFortranOrder(t *testing.T) {
	// Create test array
	data := []float32{1.0, 4.0, 2.0, 5.0, 3.0, 6.0} // Column-major order
	shape := []int{2, 3}
	arr := &Array[float32]{
		Data:    data,
		Shape:   shape,
		DType:   Float32,
		Fortran: true,
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Write array to file
	filePath := filepath.Join(tempDir, "test_fortran.npy")
	if err := WriteFile(filePath, arr); err != nil {
		t.Fatalf("Failed to write array: %v", err)
	}

	// Read array from file
	readArr, err := ReadFile[float32](filePath)
	if err != nil {
		t.Fatalf("Failed to read array: %v", err)
	}

	// Verify data
	if !reflect.DeepEqual(readArr.Data, arr.Data) {
		t.Errorf("Data mismatch. Got %v, want %v", readArr.Data, arr.Data)
	}

	// Verify Fortran order
	if !readArr.Fortran {
		t.Errorf("Fortran order not preserved. Got %v, want %v", readArr.Fortran, arr.Fortran)
	}
}

// TestMultiDimensionalArray tests writing and reading a multi-dimensional array
func TestMultiDimensionalArray(t *testing.T) {
	// Create test array - 2x2x2 cube
	data := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	shape := []int{2, 2, 2}
	arr := &Array[int64]{
		Data:    data,
		Shape:   shape,
		DType:   Int64,
		Fortran: false,
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Write array to file
	filePath := filepath.Join(tempDir, "test_multidim.npy")
	if err := WriteFile(filePath, arr); err != nil {
		t.Fatalf("Failed to write array: %v", err)
	}

	// Read array from file
	readArr, err := ReadFile[int64](filePath)
	if err != nil {
		t.Fatalf("Failed to read array: %v", err)
	}

	// Verify data
	if !reflect.DeepEqual(readArr.Data, arr.Data) {
		t.Errorf("Data mismatch. Got %v, want %v", readArr.Data, arr.Data)
	}

	// Verify shape
	if !reflect.DeepEqual(readArr.Shape, arr.Shape) {
		t.Errorf("Shape mismatch. Got %v, want %v", readArr.Shape, arr.Shape)
	}
}

// TestNPZFile tests writing and reading multiple arrays in a .npz file
func TestNPZFile(t *testing.T) {
	// Create test arrays
	arr1 := &Array[float64]{
		Data:    []float64{1.0, 2.0, 3.0, 4.0},
		Shape:   []int{2, 2},
		DType:   Float64,
		Fortran: false,
	}

	arr2 := &Array[int32]{
		Data:    []int32{5, 6, 7, 8, 9},
		Shape:   []int{5},
		DType:   Int32,
		Fortran: false,
	}

	// Create NPZ file
	npz := NewNPZFile()
	Add(npz, "array1", arr1)
	Add(npz, "array2", arr2)

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Write NPZ file
	filePath := filepath.Join(tempDir, "test.npz")
	if err := WriteNPZFile(filePath, npz); err != nil {
		t.Fatalf("Failed to write NPZ file: %v", err)
	}

	// Read NPZ file
	readNPZ, err := ReadNPZFile(filePath)
	if err != nil {
		t.Fatalf("Failed to read NPZ file: %v", err)
	}

	// Check keys
	keys := Keys(readNPZ)
	if len(keys) != 2 {
		t.Errorf("Expected 2 keys, got %d", len(keys))
	}

	// Check array1
	arr1Read, ok := Get[float64](readNPZ, "array1")
	if !ok {
		t.Fatalf("Failed to get array1 from NPZ file")
	}
	if !reflect.DeepEqual(arr1Read.Data, arr1.Data) {
		t.Errorf("Data mismatch for array1. Got %v, want %v", arr1Read.Data, arr1.Data)
	}

	// Check array2
	arr2Read, ok := Get[int32](readNPZ, "array2")
	if !ok {
		t.Fatalf("Failed to get array2 from NPZ file")
	}
	if !reflect.DeepEqual(arr2Read.Data, arr2.Data) {
		t.Errorf("Data mismatch for array2. Got %v, want %v", arr2Read.Data, arr2.Data)
	}
}

// TestInMemoryReadWrite tests writing and reading arrays using in-memory buffers
func TestInMemoryReadWrite(t *testing.T) {
	// Create test array
	data := []uint16{100, 200, 300, 400, 500}
	shape := []int{5}
	arr := &Array[uint16]{
		Data:    data,
		Shape:   shape,
		DType:   Uint16,
		Fortran: false,
	}

	// Write array to buffer
	var buf bytes.Buffer
	if err := Write(&buf, arr); err != nil {
		t.Fatalf("Failed to write array to buffer: %v", err)
	}

	// Read array from buffer
	readArr, err := Read[uint16](bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("Failed to read array from buffer: %v", err)
	}

	// Verify data
	if !reflect.DeepEqual(readArr.Data, arr.Data) {
		t.Errorf("Data mismatch. Got %v, want %v", readArr.Data, arr.Data)
	}

	// Verify shape
	if !reflect.DeepEqual(readArr.Shape, arr.Shape) {
		t.Errorf("Shape mismatch. Got %v, want %v", readArr.Shape, arr.Shape)
	}

	// Verify dtype
	if readArr.DType != arr.DType {
		t.Errorf("DType mismatch. Got %v, want %v", readArr.DType, arr.DType)
	}
}

// TestInvalidFile tests handling of invalid files
func TestInvalidFile(t *testing.T) {
	// Try to read a non-existent file
	_, err := ReadFile[float64]("nonexistent.npy")
	if err == nil {
		t.Error("Expected error when reading non-existent file, got nil")
	}

	// Try to read a file with invalid extension
	tempFile, err := os.CreateTemp("", "invalid_file.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	tempFile.Close()
	defer os.Remove(tempFile.Name())

	_, err = ReadFile[float64](tempFile.Name())
	if err == nil {
		t.Error("Expected error when reading file with invalid extension, got nil")
	}

	// Try to read a corrupted NPY file
	corruptedFile, err := os.CreateTemp("", "corrupted.npy")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	corruptedFile.Write([]byte("This is not a valid NPY file"))
	corruptedFile.Close()
	defer os.Remove(corruptedFile.Name())

	_, err = ReadFile[float64](corruptedFile.Name())
	if err == nil {
		t.Error("Expected error when reading corrupted NPY file, got nil")
	}
}

// TestUnsupportedDType tests handling of unsupported data types
func TestUnsupportedDType(t *testing.T) {
	// Create a buffer with a header containing an unsupported dtype
	var buf bytes.Buffer
	buf.Write([]byte("\x93NUMPY")) // Magic string
	buf.Write([]byte{1, 0})        // Version 1.0
	headerStr := "{'descr': '|O', 'fortran_order': False, 'shape': (1,), }"
	headerLen := len(headerStr) + 10 // Padding for alignment
	paddingLen := headerLen % 16
	if paddingLen != 0 {
		headerLen += (16 - paddingLen)
	}
	headerStr += strings.Repeat(" ", headerLen-len(headerStr)-1) + "\n"
	binary.Write(&buf, binary.LittleEndian, uint16(len(headerStr)))
	buf.Write([]byte(headerStr))
	buf.Write([]byte{0, 0, 0, 0}) // Some dummy data

	// Try to read the buffer
	_, err := Read[interface{}](bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Error("Expected error when reading unsupported dtype, got nil")
	}
}
