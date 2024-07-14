#include "Aqua.h"

namespace Aqua {

    template<typename T, size_t N>
    Tensor<T, N>::Tensor() : data() {}

    template<typename T, size_t N>
    Tensor<T, N>::Tensor(const DataType& initData) : data(initData) {}

    template<typename T, size_t N>
    Tensor<T, N>::~Tensor() {}

    template<typename T, size_t N>
    const typename Tensor<T, N>::DataType& Tensor<T, N>::getData() const {
        return data;
    }

    template<typename T, size_t N>
    void Tensor<T, N>::disp() const {
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
}