#pragma once

#include <iostream>
#include <vector>
#include <initializer_list>

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
    };
}
