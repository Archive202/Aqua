# Aqua

This is a C++ library providing tensor variables and calculation of tensor, which can be used to implement neural networks

## Usage

### Declaration

```cpp
Aqua::Tensor t<variable_type, num_of_dimension>;
```

Example:

```cpp
Aqua::Tensor t<double, 2>;
```


### Assignment
A tensor variable can be assignmented by a vector variable.

Example:

```cpp
std::vector<std::vector<double>> data = {
    {1, 2, 3},
    {4, 5, 6}
};

Aqua::Tensor t<double, 2> = data;

```