#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>

void computeSincos(const std::vector<double>& input, std::vector<double>& sinOutput, std::vector<double>& cosOutput) {
    size_t size = input.size();
    for (size_t i = 0; i < size; ++i) {
        sinOutput[i] = std::sin(input[i]);
        // cosOutput[i] = std::cos(input[i]);
    }
}

int main() {
    const size_t size = 100; // Increase size for better timing
    std::vector<double> input(size);
    std::vector<double> sinOutput(size);
    std::vector<double> cosOutput(size);

    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(0.0, M_PI); 

    // Generate random input values
    for (size_t i = 0; i < size; ++i) {
        input[i] = dis(gen);
    }

    const int iterations = 1000000; // Number of iterations for averaging
    double totalDuration = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {

        computeSincos(input, sinOutput, cosOutput);

    }
    auto end = std::chrono::high_resolution_clock::now();
    totalDuration = std::chrono::duration<double, std::nano>(end - start).count(); // Store duration in nanoseconds

    std::cout << "Average time taken to compute: " << (totalDuration / iterations) << " nanoseconds" << std::endl;

    return 0;
}
