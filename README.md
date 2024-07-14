# Aqua

This is a C++ library providing tensor variables and calculation of tensor, which can be used to implement neural networks

## Usage

### Import

```cpp
#include <Tensor.hpp>
```

### Declaration

```cpp
// Aqua::Tensor t<variable_type, num_of_dimension>;
Aqua::Tensor t<double, 2>;
```


### Assignment

```cpp
Aqua::Tensor t<double, 2> t = {
    {1, 2, 3},
    {4, 5, 6}
};
```

### Operations

#### Outputting
```cpp
Tensor<int, 3> t = {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};
t.disp();

// Outputs:
// [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]
```

#### Element Accessing
```cpp
Tensor<int, 3> t = {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};
cout << t.at(0, 1, 2);

// Outputs:
// 6
```