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

        Tensor();
        Tensor(const DataType& initData);

        ~Tensor();
        
        const DataType& getData() const;

        void disp() const;

    private:
        DataType data;
    };
}

#include "Aqua.cpp"