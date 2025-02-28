package npy

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
)

// TestToCsv_1D tests exporting a 1D array to Csv
func TestToCsv_1D(t *testing.T) {
	// Create test array
	data := []float64{1.1, 2.2, 3.3, 4.4, 5.5}
	shape := []int{5}
	arr := &Array[float64]{
		Data:    data,
		Shape:   shape,
		DType:   Float64,
		Fortran: false,
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-csv-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Export to Csv
	csvPath := filepath.Join(tempDir, "test_1d.csv")
	if err := ToCsv(arr, csvPath); err != nil {
		t.Fatalf("Failed to export to Csv: %v", err)
	}

	// Read the Csv file
	f, err := os.Open(csvPath)
	if err != nil {
		t.Fatalf("Failed to open Csv file: %v", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("Failed to read Csv: %v", err)
	}

	// Verify Csv content
	if len(records) != 1 {
		t.Errorf("Expected 1 row, got %d", len(records))
	}

	if len(records[0]) != 5 {
		t.Errorf("Expected 5 columns, got %d", len(records[0]))
	}

	// Convert Csv strings back to float64 for comparison
	csvData := make([]float64, len(records[0]))
	for i, val := range records[0] {
		csvData[i], err = strconv.ParseFloat(val, 64)
		if err != nil {
			t.Fatalf("Failed to parse Csv value: %v", err)
		}
	}

	// Verify data
	if !reflect.DeepEqual(csvData, data) {
		t.Errorf("Data mismatch. Got %v, want %v", csvData, data)
	}
}

// TestToCsv_2D tests exporting a 2D array to Csv
func TestToCsv_2D(t *testing.T) {
	// Create test array - 2x3 matrix
	data := []int32{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	arr := &Array[int32]{
		Data:    data,
		Shape:   shape,
		DType:   Int32,
		Fortran: false,
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-csv-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Export to Csv
	csvPath := filepath.Join(tempDir, "test_2d.csv")
	if err := ToCsv(arr, csvPath); err != nil {
		t.Fatalf("Failed to export to Csv: %v", err)
	}

	// Read the Csv file
	f, err := os.Open(csvPath)
	if err != nil {
		t.Fatalf("Failed to open Csv file: %v", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("Failed to read Csv: %v", err)
	}

	// Verify Csv content
	if len(records) != 2 {
		t.Errorf("Expected 2 rows, got %d", len(records))
	}

	if len(records[0]) != 3 {
		t.Errorf("Expected 3 columns, got %d", len(records[0]))
	}

	// Verify data
	expectedCsv := [][]string{
		{"1", "2", "3"},
		{"4", "5", "6"},
	}

	for i := range records {
		for j := range records[i] {
			if records[i][j] != expectedCsv[i][j] {
				t.Errorf("Data mismatch at [%d][%d]. Got %s, want %s", i, j, records[i][j], expectedCsv[i][j])
			}
		}
	}
}

// TestToCsv_FortranOrder tests exporting a 2D array in Fortran order to Csv
func TestToCsv_FortranOrder(t *testing.T) {
	// Create test array - 2x3 matrix in Fortran (column-major) order
	// In Fortran order, the data would be laid out as:
	// 1, 4, 2, 5, 3, 6 (columns first, then rows)
	data := []int32{1, 4, 2, 5, 3, 6}
	shape := []int{2, 3}
	arr := &Array[int32]{
		Data:    data,
		Shape:   shape,
		DType:   Int32,
		Fortran: true,
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "npy-csv-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Export to Csv
	csvPath := filepath.Join(tempDir, "test_fortran.csv")
	if err := ToCsv(arr, csvPath); err != nil {
		t.Fatalf("Failed to export to Csv: %v", err)
	}

	// Read the Csv file
	f, err := os.Open(csvPath)
	if err != nil {
		t.Fatalf("Failed to open Csv file: %v", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("Failed to read Csv: %v", err)
	}

	// Verify Csv content - the result should still be the logical 2x3 matrix
	if len(records) != 2 {
		t.Errorf("Expected 2 rows, got %d", len(records))
	}

	if len(records[0]) != 3 {
		t.Errorf("Expected 3 columns, got %d", len(records[0]))
	}

	// Verify data - even though the internal storage is column-major,
	// the Csv should represent the logical matrix
	expectedCsv := [][]string{
		{"1", "2", "3"},
		{"4", "5", "6"},
	}

	for i := range records {
		for j := range records[i] {
			if records[i][j] != expectedCsv[i][j] {
				t.Errorf("Data mismatch at [%d][%d]. Got %s, want %s", i, j, records[i][j], expectedCsv[i][j])
			}
		}
	}
}

// TestNPZToCsvDir tests exporting multiple arrays from an NPZ file
func TestNPZToCsvDir(t *testing.T) {
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

	// Create temporary directories for test files
	tempDir, err := os.MkdirTemp("", "npy-csv-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	npzPath := filepath.Join(tempDir, "test.npz")
	csvDir := filepath.Join(tempDir, "csv")

	// Write NPZ file
	if err := WriteNPZFile(npzPath, npz); err != nil {
		t.Fatalf("Failed to write NPZ file: %v", err)
	}

	// Export NPZ to Csv
	if err := NPZToCsvDir(npzPath, csvDir); err != nil {
		t.Fatalf("Failed to export NPZ to Csv: %v", err)
	}

	// Verify array1.csv
	f1, err := os.Open(filepath.Join(csvDir, "array1.csv"))
	if err != nil {
		t.Fatalf("Failed to open array1.csv: %v", err)
	}
	defer f1.Close()

	reader1 := csv.NewReader(f1)
	records1, err := reader1.ReadAll()
	if err != nil {
		t.Fatalf("Failed to read array1.csv: %v", err)
	}

	// 2x2 matrix
	if len(records1) != 2 || len(records1[0]) != 2 {
		t.Errorf("Unexpected dimensions for array1.csv: %d x %d", len(records1), len(records1[0]))
	}

	// Verify array2.csv
	f2, err := os.Open(filepath.Join(csvDir, "array2.csv"))
	if err != nil {
		t.Fatalf("Failed to open array2.csv: %v", err)
	}
	defer f2.Close()

	reader2 := csv.NewReader(f2)
	records2, err := reader2.ReadAll()
	if err != nil {
		t.Fatalf("Failed to read array2.csv: %v", err)
	}

	// 1D array with 5 elements
	if len(records2) != 1 || len(records2[0]) != 5 {
		t.Errorf("Unexpected dimensions for array2.csv: %d x %d", len(records2), len(records2[0]))
	}
}
