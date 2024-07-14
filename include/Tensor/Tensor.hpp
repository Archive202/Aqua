#pragma once

#include <iostream>
#include <vector>

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
        Tensor(const DataType& initData) : data(initData) {}

        ~Tensor() {}
        
        const DataType& getData() const {
            return data;
        }

        void disp() const {
            if constexpr (N == 2) {
                for (const auto& row : data) {
                    for (const auto& elem : row) {
                        std::cout << elem << " ";
                    }
                    std::cout << std::endl;
                }
            } else {
                std::cout << "Shape Error!" << std::endl;
            }
        }

    private:
        DataType data;
    };
}
