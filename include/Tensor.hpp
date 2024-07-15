#pragma once

#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace Aqua {
    template<typename T, size_t N>
    struct TensorData {
        using type = std::vector<typename TensorData<T, N-1>::type>;
    };

    template<typename T>
    struct TensorData<T, 0> {
        using type = T;
    };

    template<typename T, size_t N>
    class Tensor {
    public:
        using DataType = typename TensorData<T, N>::type;

        Tensor() : data() {}
        Tensor(std::initializer_list<typename TensorData<T, N-1>::type> list) : data(list) {}

        ~Tensor() {}

        const DataType& getData() const {
            return data;
        }

        template<typename... Args>
        T& at(Args... args) {
            return accessElement(data, args...);
        }

        void disp() const {
            printVector(data);
            std::cout << std::endl;
        }

        // 运算符重载
        /*-----------------------------------------------------------*/
        Tensor<T, N> operator + (const Tensor<T, N>& other) const {
            return elementWiseOperation(other, std::plus<T>());
        }

        Tensor<T, N> operator - (const Tensor<T, N>& other) const {
            return elementWiseOperation(other, std::minus<T>());
        }

        Tensor<T, N> operator * (const Tensor<T, N>& other) const {
            return elementWiseOperation(other, std::multiplies<T>());
        }

        Tensor<T, N> operator / (const Tensor<T, N>& other) const {
            return elementWiseOperation(other, std::divides<T>());
        }

        Tensor<T, N> operator % (const Tensor<T, N>& other) const {
            if (N != 2) {
                throw std::invalid_argument("Matrix multiplication is only defined for 2D tensors.");
            }

            if (this->data[0].size() != other.data.size()) {
                throw std::invalid_argument("Matrices dimensions do not match for multiplication.");
            }

            size_t rows = this->data.size();
            size_t cols = other.data[0].size();
            size_t inner_dim = other.data.size();

            Tensor<T, N> result;
            result.data.resize(rows, std::vector<T>(cols, T()));

            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    for (size_t k = 0; k < inner_dim; ++k) {
                        result.data[i][j] += this->data[i][k] * other.data[k][j];
                    }
                }
            }

            return result;
        }

        Tensor<T, N>& operator += (const Tensor<T, N>& other) {
            *this = *this + other;
            return *this;
        }

        Tensor<T, N>& operator -= (const Tensor<T, N>& other) {
            *this = *this - other;
            return *this;
        }

        Tensor<T, N>& operator *= (const Tensor<T, N>& other) {
            *this = *this * other;
            return *this;
        }

        Tensor<T, N>& operator /= (const Tensor<T, N>& other) {
            *this = *this / other;
            return *this;
        }

        Tensor<T, N>& operator %= (const Tensor<T, N>& other) {
            *this = *this % other;
            return *this;
        }
        /*-----------------------------------------------------------*/

        /*工具*/
        /*-----------------------------------------------------------*/
        T max() const {
            T maximum = std::numeric_limits<T>::min();
            maxImpl(maximum, data);
            return maximum;
        }

        T min() const {
            T minimum = std::numeric_limits<T>::max();
            minImpl(minimum, data);
            return minimum;
        }

        /*标准化函数*/
        /*-----------------------------------------------------------*/
        Tensor<T, N> minmaxNormalize() const {
            T min = this->min();
            T max = this->max();
            return elementWiseOperation(*this, [min, max](T elem, T) {
                return (elem - min) / (max - min);
            });
        }

    private:
        DataType data;

        template<typename U>
        void printVector(const U& elem) const {
            std::cout << elem;
        }

        template<typename U>
        void printVector(const std::vector<U>& vec) const {
            std::cout << "[";
            for (const auto& elem : vec) {
                printVector(elem);
                if (&elem != &vec.back()) std::cout << ", ";
            }
            std::cout << "]";
        }

        template<typename U, typename... Args>
        auto& accessElement(std::vector<U>& vec, size_t index, Args... args) {
            return accessElement(vec[index], args...);
        }

        template<typename U>
        U& accessElement(U& elem) {
            return elem;
        }

        template<typename Operation>
        Tensor<T, N> elementWiseOperation(const Tensor<T, N>& other, Operation op) const {
            Tensor<T, N> result;
            elementWiseOperationImpl(result.data, this->data, other.data, op);
            return result;
        }

        // 广播支持
        template<typename Operation, typename U>
        void elementWiseOperationImpl(std::vector<U>& result, const std::vector<U>& a, const std::vector<U>& b, Operation op) const {
            size_t size_a = a.size();
            size_t size_b = b.size();

            if (size_a != size_b && size_a != 1 && size_b != 1) {
                throw std::invalid_argument("Tensors are not broadcastable.");
            }

            size_t max_size = std::max(size_a, size_b);
            result.resize(max_size);

            for (size_t i = 0; i < max_size; i++) {
                if (size_a == size_b) {
                    elementWiseOperationImpl(result[i], a[i], b[i], op);
                } else if (size_a == 1) {
                    elementWiseOperationImpl(result[i], a[0], b[i], op);
                } else if (size_b == 1) {
                    elementWiseOperationImpl(result[i], a[i], b[0], op);
                }
            }
        }

        template<typename Operation>
        void elementWiseOperationImpl(T& result, const T& a, const T& b, Operation op) const {
            result = op(a, b);
        }

        template<typename U>
        void maxImpl(T& maximum, const std::vector<U>& vec) const {
            for (const auto& elem : vec) {
                maxImpl(maximum, elem);
            }
        }

        void maxImpl(T& maximum, const T& elem) const {
            if (elem > maximum) {
                maximum = elem;
            }
        }

        template<typename U>
        void minImpl(T& minimum, const std::vector<U>& vec) const {
            for (const auto& elem : vec) {
                minImpl(minimum, elem);
            }
        }

        void minImpl(T& minimum, const T& elem) const {
            if (elem < minimum) {
                minimum = elem;
            }
        }
    };
}
