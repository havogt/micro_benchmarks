#pragma once

class pointer_chaser {
public:
    using ptr_type = uint64_t;

    void** h_array;
    void** d_array;
    size_t size; // in bytes
    size_t distance; // in bytes
    size_t max_steps; // number of possible hops until end of array

    pointer_chaser(size_t size, size_t distance)
        : size(size / sizeof(void*) * sizeof(void*)) // only in multiples of pointer size
        , distance(distance)
        , max_steps(size / distance)
    {
        h_array = new void*[size / sizeof(void*)];
        cudaMalloc((void**)&d_array, size);

        for(size_t i = 0; i < max_steps; ++i) {
            h_array[i] = d_array + (i + 1) * (distance / sizeof(void*));
        }
        cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    }
};
