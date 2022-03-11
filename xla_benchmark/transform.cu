#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

int main(){
    int64_t N = 12288*30522;
    thrust::device_vector<float> X(N);
    thrust::device_vector<float> Y(N);
    thrust::device_vector<float> Z(N);
    for (int i = 0; i < 100; i++) {
      thrust::transform(X.begin(), X.end(), Y.begin(), X.begin(), thrust::plus<float>());
    }
}
