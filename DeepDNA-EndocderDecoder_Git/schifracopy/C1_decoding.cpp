
#define NO_GFLUT
#include "schifra_galois_field.hpp"
#undef NO_GFLUT
#include "schifra_galois_field_polynomial.hpp"
#include "schifra_sequential_root_generator_polynomial_creator.hpp"
#include "schifra_reed_solomon_encoder.hpp"
#include "schifra_reed_solomon_decoder.hpp"
#include "schifra_reed_solomon_block.hpp"
#include "schifra_error_processes.hpp"
#include "schifra_utilities.hpp"
#include <random>
#include <unordered_set>
#include <thread>
#include <memory>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iterator>
#include <sstream>
//#include "json11.hpp"

#include "CommonDefinitions.h"


auto constexpr M_ = M_orig;
auto constexpr M = 65535 - t2;
auto constexpr M_without_padding = M_ + 2 * t1; //; //actually it is M+2t1
auto constexpr N = 13;
auto constexpr total_data_len = M * N;
auto constexpr total_code_len = (M + t2) * N;
//auto word_size_in_bits = M * 16; // 8 symbols per letter

/* Finite Field Parameters */
std::size_t constexpr field_descriptor = 16;
std::size_t constexpr generator_polynomial_index = 0; //0
std::size_t constexpr generator_polynomial_root_count = t2;

/* Reed Solomon Code Parameters */
std::size_t constexpr code_length = M + t2;
std::size_t constexpr fec_length = t2;
std::size_t constexpr data_length = code_length - fec_length;



template<typename T>
static void ConvertToCyclic(std::shared_ptr<T[]> data, std::shared_ptr<T[]> ans, uint16_t M, uint16_t N) {
    for (std::size_t i = 0; i < N; i++) {
        for (std::size_t j = 0; j < M; j++) {
            ans[i * M + j] = data[((i + j) * M + j) % (M * N)];
        }
    }
}

template<typename T>
static void ConvertFromCyclic(std::shared_ptr<T[]> data, std::shared_ptr<T[]> ans, uint16_t M, uint16_t N) {
    for (std::size_t i = 0; i < N; i++) {
        for (std::size_t j = 0; j < M; j++) {
            ans[((i + j) * M + j) % (M * N)] = data[i * M + j];
        }
    }
}

template<typename T>
bool RsEncode(T* data, T* ans, std::shared_ptr<const schifra::galois::field> field,
    std::shared_ptr< schifra::galois::field_polynomial> generator_polynomial) {

    /* Instantiate Encoder (Codec) */
    typedef schifra::reed_solomon::encoder<code_length, fec_length, data_length> encoder_t;
    const encoder_t encoder(*field.get(), *generator_polynomial.get());

    std::vector<uint16_t> dataVec(data, data + data_length);
    dataVec.resize(code_length);

    schifra::reed_solomon::block<code_length, fec_length> block;
    schifra::reed_solomon::copy(&dataVec[0], dataVec.size(), block);

    /* Transform message into Reed-Solomon encoded codeword */
    if (!encoder.encode(block))
    {
        std::cout << "Error - Critical encoding failure! "
            << "Msg: " << block.error_as_string() << std::endl;
        return false;
    }
    else {
        std::cout << "Encryption Finished" << std::endl;
        std::copy_n(block.data, code_length, ans);
        return true;
    }
}

template<typename T>
bool RsDecode(T* data, T* ans,
    schifra::reed_solomon::erasure_locations_t erasure_location_list,
    std::shared_ptr<const schifra::galois::field> field)
{

    /* Instantiate Decoder (Codec) */
    typedef schifra::reed_solomon::decoder<code_length, fec_length, data_length> decoder_t;
    const decoder_t decoder(*field.get(), generator_polynomial_index);

    std::vector<uint16_t> dataVec(data, data + code_length);

    schifra::reed_solomon::block<code_length, fec_length> block;

    schifra::reed_solomon::copy(&dataVec[0], dataVec.size(), block);

    if (!decoder.decode(block, erasure_location_list))
    {
        std::cout << "Error - Critical decoding failure! "
            << "Msg: " << block.error_as_string() << std::endl;
        return false;
    }
    else
    {
        std::copy_n(block.data, data_length, ans); //DVIRRRRRR change this
        std::cout << "Decryption Finished" << std::endl;
        return true;
    }

}

int main()
{

    schifra::utils::timer timer;
    timer.start();

    /* Instantiate Finite Field and Generator Polynomials */
    auto field = std::make_shared<const schifra::galois::field>(field_descriptor,
        schifra::galois::primitive_polynomial_size14,
        schifra::galois::primitive_polynomial14);

    auto generator_polynomial = std::make_shared<schifra::galois::field_polynomial>(*field.get());

    if (
        !schifra::make_sequential_root_generator_polynomial(*field.get(),
            generator_polynomial_index,
            generator_polynomial_root_count,
            *generator_polynomial.get())
        )
    {
        std::cout << "Error - Failed to create sequential root generator!" << std::endl;
        return -1;
    }

    std::vector<std::thread> threads;

    std::shared_ptr<uint16_t[]> data(new uint16_t[(M_without_padding + t2) * N]);
    std::shared_ptr<uint16_t[]> tmp(new uint16_t[(M_without_padding + t2) * N]);
    std::shared_ptr<uint16_t[]> cyclic_data(new uint16_t[total_code_len]);
    std::shared_ptr<uint16_t[]> cyclic_data_tmp(new uint16_t[(M_without_padding + t2) * N]);
    std::shared_ptr<uint16_t[]> decryptedData(new uint16_t[total_code_len]);
    std::shared_ptr<uint16_t[]> decryptedData_no_padding(new uint16_t[(M_without_padding + t2) * N]);
    std::shared_ptr<uint16_t[]> finalRes(new uint16_t[(M_without_padding + t2) * N]);


    FILE* filp_in = fopen("c1_decoding_in.bin", "rb");
    int bytes_read = fread(data.get(), sizeof(uint16_t), (M_without_padding + t2) * N, filp_in);

    //std::cout << "here1" << std::endl;


    for (size_t i = 0; i < (M_without_padding + t2) * N; i++)
    {
        data[i] = __builtin_bswap16(data[i]);
    }
    ConvertToCyclic<uint16_t>(data, tmp, M_without_padding + t2, N);


    for (std::size_t i = 0; i < sizeof(cyclic_data); ++i) {
        cyclic_data[i] = 0;
    }


    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M_without_padding; ++j)
        {
            cyclic_data[i * (M + t2) + j] = tmp[i * (M_without_padding + t2) + j];
        }
        for (std::size_t j = 0; j < t2; ++j)

        {
            cyclic_data[i * (M + t2) + M + j] = tmp[i * (M_without_padding + t2) + M_without_padding + j];
        }
    }

    schifra::reed_solomon::erasure_locations_t erasure_location_list;
    std::ifstream myfile("c1_erasures_locations.txt");
    std::copy(std::istream_iterator<size_t>(myfile),
        std::istream_iterator<size_t>(),
        back_inserter(erasure_location_list));

    threads.clear();
    //std::cout << "here" << std::endl;

    //Decryption
    schifra::utils::timer decryptionTimer;
    decryptionTimer.start();

    for (std::size_t i = 0; i < N; i++) {
        threads.push_back(std::thread(RsDecode<uint16_t>,
            cyclic_data.get() + (i * (M + t2)),
            decryptedData.get() + (i * (M + t2)),
            erasure_location_list, field));
    }


    for (std::size_t i = 0; i < N; i++) {
        threads[i].join();
    }

    decryptionTimer.stop();

    std::cout << "Total encryption time: " << decryptionTimer.time() << std::endl;


    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M_without_padding; ++j)
        {
            decryptedData_no_padding[i * (M_without_padding + t2) + j] = decryptedData[i * (M + t2) + j];
        }
        for (std::size_t j = 0; j < t2; ++j)

        {
            decryptedData_no_padding[i * (M_without_padding + t2) + M_without_padding + j] = decryptedData[i * (M + t2) + M + j];
        }
    }

    ConvertFromCyclic<uint16_t>(decryptedData_no_padding, finalRes, M_without_padding + t2, N);


    timer.stop();
    for (size_t i = 0; i < (M_without_padding + t2) * N; i++)
    {
        finalRes[i] = __builtin_bswap16(finalRes[i]);
    }

    FILE* filp_out = fopen("c1_decoding_out.bin", "wb");
    fwrite(finalRes.get(), sizeof(uint16_t), (M_without_padding + t2) * N, filp_out);
    fclose(filp_out);
    return 0;

}
