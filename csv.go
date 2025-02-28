package npy

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
)

// ToCsv exports an array to a CSV file
func ToCsv[T any](arr *Array[T], csvPath string) error {
	// Create the file
	f, err := os.Create(csvPath)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %w", err)
	}
	defer f.Close()

	// Create a CSV writer
	writer := csv.NewWriter(f)
	defer writer.Flush()

	// Handle the data based on dimensions
	dimensions := len(arr.Shape)

	if dimensions == 0 || (dimensions == 1 && arr.Shape[0] == 0) {
		// Empty array
		return nil
	} else if dimensions == 1 {
		// 1D array (vector) - write as a single row
		record := make([]string, len(arr.Data))
		for i, val := range arr.Data {
			record[i] = fmt.Sprintf("%v", val)
		}
		if err := writer.Write(record); err != nil {
			return fmt.Errorf("failed to write CSV row: %w", err)
		}
	} else if dimensions == 2 {
		// 2D array (matrix)
		rows := arr.Shape[0]
		cols := arr.Shape[1]

		for r := 0; r < rows; r++ {
			record := make([]string, cols)
			for c := 0; c < cols; c++ {
				// Calculate index based on ordering
				var idx int
				if arr.Fortran {
					// Column-major (Fortran) order
					idx = c*rows + r
				} else {
					// Row-major (C) order
					idx = r*cols + c
				}
				record[c] = fmt.Sprintf("%v", arr.Data[idx])
			}
			if err := writer.Write(record); err != nil {
				return fmt.Errorf("failed to write CSV row: %w", err)
			}
		}
	} else {
		// Higher dimensions
		return fmt.Errorf("arrays with more than 2 dimensions are not supported for Csv export")
	}

	return nil
}

// NPZToCsvDir exports all arrays in an NPZ file to CSV files in the specified directory
func NPZToCsvDir(npzPath string, outputDir string) error {
	// Read the NPZ file
	npz, err := ReadNPZFile(npzPath)
	if err != nil {
		return fmt.Errorf("failed to read NPZ file: %w", err)
	}

	// Make sure the output directory exists
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Export each array based on its type
	for _, key := range Keys(npz) {
		outPath := filepath.Join(outputDir, key+".csv")

		// Here we need to try each type due to Go's type system limitations
		exported := false

		if arr, ok := Get[bool](npz, key); ok {
			if err := ToCsv(arr, outPath); err != nil {
				return fmt.Errorf("failed to export %s: %w", key, err)
			}
			exported = true
		}

		if !exported {
			if arr, ok := Get[int8](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			if arr, ok := Get[int16](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			if arr, ok := Get[int32](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			if arr, ok := Get[int64](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			if arr, ok := Get[uint8](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			if arr, ok := Get[uint16](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			if arr, ok := Get[uint32](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			if arr, ok := Get[uint64](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			if arr, ok := Get[float32](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			if arr, ok := Get[float64](npz, key); ok {
				if err := ToCsv(arr, outPath); err != nil {
					return fmt.Errorf("failed to export %s: %w", key, err)
				}
				exported = true
			}
		}

		if !exported {
			return fmt.Errorf("unsupported data type for array %s", key)
		}
	}

	return nil
}
