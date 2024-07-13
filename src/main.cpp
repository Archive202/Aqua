#include <iostream>
#include <vector>
#include "../include/Aqua.h"
using namespace std;

int main() {
    vector<vector<int>> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    vector<vector<int>> b = {{1, 2, 3}, {4, 5, 6}};

    Aqua::Tensor<int, 2> t1 = a; 
    t1.disp();
}