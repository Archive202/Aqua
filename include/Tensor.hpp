#pragma once

#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>

namespace Aqua {
    template<typename T, size_t N>
    struct TensorData {
        using type = std::vector<typename TensorData<T, N-1>::type>;
    };

    template<typename T>
    struct TensorData<T, 1> {
        using type = std::vector<T>;
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

        template<typename Operation, typename U>
        void elementWiseOperationImpl(std::vector<U>& result, const std::vector<U>& a, const std::vector<U>& b, Operation op) const {
            if (a.size() != b.size()) {
                throw std::invalid_argument("Tensors must have the same dimensions for element-wise operations.");
            }
            result.resize(a.size());
            for (size_t i = 0; i < a.size(); i++) {
                elementWiseOperationImpl(result[i], a[i], b[i], op);
            }
        }

        template<typename Operation>
        void elementWiseOperationImpl(T& result, const T& a, const T& b, Operation op) const {
            result = op(a, b);
        }
    };
}
