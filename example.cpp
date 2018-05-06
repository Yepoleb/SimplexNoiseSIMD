#include <stdint.h>
#include <fstream>
#include <stdio.h>
#include <cassert>
#include <vector>
#include <array>

#include "simplexnoise.hpp"

constexpr int SIZE = 1024;
constexpr int IMG_SIZE = SIZE * SIZE;


int main() {
    std::vector<uint8_t> img(IMG_SIZE);

    for (int y = 0; y < SIZE; y+=4) {
        for (int x = 0; x < SIZE; x+=4) {
            std::array<float, 16> values;
            values = noiseblock(x / 64.0, y / 64.0, (x + 4) / 64.0, (y + 4) / 64.0);
            for (int block_y = 0; block_y < 4; block_y++) {
                for (int block_x = 0; block_x < 4; block_x++) {
                    size_t img_pos = (y + block_y) * SIZE + (x + block_x);
                    size_t block_pos = block_y * 4 + block_x;
                    assert(img_pos < IMG_SIZE);
                    img[img_pos] = (int)((values[block_pos] + 1) * 127.5);
                }
            }
        }
    }

    std::ofstream stream("noise.bin", std::ios::out | std::ios::binary);
    if (!stream.good()) {
        printf("Bad stream\n");
        return 1;
    }
    stream.write(reinterpret_cast<char*>(img.data()), SIZE * SIZE);
    stream.close();

    return 0;
}
