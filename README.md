# npy

A Go library for reading and writing NumPy's `.npy` and `.npz` file formats with support for mixed types and Go generics.

## Features

- Type-safe API using Go generics
- Support for all common NumPy data types (bool, int8/16/32/64, uint8/16/32/64, float32/64)
- Read/write single arrays (`.npy` files)
- Read/write multiple arrays (`.npz` files)
- Support for multi-dimensional arrays
- Support for both row-major (C order) and column-major (Fortran order) arrays

## Installation

```bash
go get github.com/datumbrain/npy
```

## Import

```go
import "github.com/datumbrain/npy"
```

## Usage Examples

### Working with `.npy` Files

#### Creating and Writing a NumPy Array

```go
package main

import (
    "fmt"
    "log"

    "github.com/datumbrain/npy"
)

func main() {
    // Create a 2x3 float64 matrix
    data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    shape := []int{2, 3}

    // Create a NumPy array
    arr := &npy.Array[float64]{
        Data:    data,
        Shape:   shape,
        DType:   npy.Float64,
        Fortran: false, // Use row-major (C) order
    }

    // Write to a .npy file
    err := npy.WriteFile("matrix.npy", arr)
    if err != nil {
        log.Fatalf("Failed to write array: %v", err)
    }

    fmt.Println("Array successfully written to matrix.npy")
}
```

#### Reading a NumPy Array

```go
package main

import (
    "fmt"
    "log"

    "github.com/datumbrain/npy"
)

func main() {
    // Read a .npy file with float64 data
    arr, err := npy.ReadFile[float64]("matrix.npy")
    if err != nil {
        log.Fatalf("Failed to read array: %v", err)
    }

    // Print shape
    fmt.Printf("Array shape: %v\n", arr.Shape)

    // Access data
    fmt.Printf("Element at (0,0): %f\n", arr.Data[0])

    // Calculate index for position [1,2] in a 2x3 matrix
    // Index = row*width + col = 1*3 + 2 = 5
    fmt.Printf("Element at (1,2): %f\n", arr.Data[5])
}
```

### Working with `.npz` Files

#### Creating and Writing Multiple Arrays

```go
package main

import (
    "fmt"
    "log"

    "github.com/datumbrain/npy"
)

func main() {
    // Create first array (float64)
    arr1 := &npy.Array[float64]{
        Data:    []float64{1.0, 2.0, 3.0, 4.0},
        Shape:   []int{2, 2},
        DType:   npy.Float64,
        Fortran: false,
    }

    // Create second array (int32)
    arr2 := &npy.Array[int32]{
        Data:    []int32{5, 6, 7, 8, 9},
        Shape:   []int{5},
        DType:   npy.Int32,
        Fortran: false,
    }

    // Create NPZ file
    npzFile := npy.NewNPZFile()

    // Add arrays to NPZ file
    npy.Add(npzFile, "matrix", arr1)
    npy.Add(npzFile, "vector", arr2)

    // Write NPZ file
    err := npy.WriteNPZFile("data.npz", npzFile)
    if err != nil {
        log.Fatalf("Failed to write NPZ file: %v", err)
    }

    fmt.Println("NPZ file successfully written to data.npz")
}
```

#### Reading Multiple Arrays

```go
package main

import (
    "fmt"
    "log"

    "github.com/datumbrain/npy"
)

func main() {
    // Read NPZ file
    npzFile, err := npy.ReadNPZFile("data.npz")
    if err != nil {
        log.Fatalf("Failed to read NPZ file: %v", err)
    }

    // List all arrays in the file
    fmt.Printf("Arrays in NPZ file: %v\n", npy.Keys(npzFile))

    // Get float64 array
    matrix, ok := npy.Get[float64](npzFile, "matrix")
    if !ok {
        log.Fatal("Matrix not found in NPZ file")
    }

    // Get int32 array
    vector, ok := npy.Get[int32](npzFile, "vector")
    if !ok {
        log.Fatal("Vector not found in NPZ file")
    }

    // Print data
    fmt.Printf("Matrix: %v\n", matrix.Data)
    fmt.Printf("Vector: %v\n", vector.Data)
}
```

## Working with Different Types

The library supports all common NumPy data types:

```go
// Create arrays with different types
boolArr := &npy.Array[bool]{
    Data:  []bool{true, false, true},
    Shape: []int{3},
    DType: npy.Bool,
}

int8Arr := &npy.Array[int8]{
    Data:  []int8{-1, 0, 1},
    Shape: []int{3},
    DType: npy.Int8,
}

uint16Arr := &npy.Array[uint16]{
    Data:  []uint16{100, 200, 300},
    Shape: []int{3},
    DType: npy.Uint16,
}

float32Arr := &npy.Array[float32]{
    Data:  []float32{1.1, 2.2, 3.3},
    Shape: []int{3},
    DType: npy.Float32,
}
```

## Multi-dimensional Arrays

When working with multi-dimensional arrays, remember that NumPy arrays are stored in either:

- C order (row-major, default): last dimension varies fastest
- Fortran order (column-major): first dimension varies fastest

For example, a 2x3 array in C order would have elements in this sequence:

```raw
[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]
```

When specifying multi-dimensional data, ensure your Go slice follows this ordering based on your `Fortran` flag.

## CSV Export

The library also provides functionality to export NumPy arrays to CSV format:

### Exporting a Single Array to CSV

```go
package main

import (
    "fmt"
    "log"

    "github.com/datumbrain/npy"
)

func main() {
    // Read a .npy file with float64 data
    arr, err := npy.ReadFile[float64]("matrix.npy")
    if err != nil {
        log.Fatalf("Failed to read array: %v", err)
    }

    // Export to CSV
    err = npy.ToCSV(arr, "matrix.csv")
    if err != nil {
        log.Fatalf("Failed to export to CSV: %v", err)
    }

    fmt.Println("Successfully exported to CSV")
}
```

### Exporting All Arrays from an NPZ File

```go
package main

import (
    "fmt"
    "log"

    "github.com/datumbrain/npy"
)

func main() {
    // Export all arrays in an NPZ file to individual CSV files
    err := npy.NPZToCSVDir("data.npz", "./csv_output")
    if err != nil {
        log.Fatalf("Failed to export NPZ to CSV: %v", err)
    }

    fmt.Println("Successfully exported all arrays to CSV files")
    // Creates files like:
    // - ./csv_output/array1.csv
    // - ./csv_output/array2.csv
}
```

The CSV export supports:

- 1D arrays (exported as a single row)
- 2D arrays (exported as rows and columns)
- Both row-major (C order) and column-major (Fortran order) arrays
- All NumPy data types supported by the library

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
