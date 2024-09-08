
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
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
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

/*static std::unordered_set<size_t> pickSet(int N, int k, std::mt19937& gen)
{
    std::uniform_int_distribution<> dis(0, N);
    std::unordered_set<size_t> elems;

    while (elems.size() < k) {
        elems.insert(dis(gen));
    }

    return elems;
}

static std::vector<size_t> pick(int N, int k) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::unordered_set<size_t> elems = pickSet(N, k, gen);

    // ok, now we have a set of k elements. but now
    // it's in a [unknown] deterministic order.
    // so we have to shuffle it:

    std::vector<size_t> result(elems.begin(), elems.end());
    std::shuffle(result.begin(), result.end(), gen);
    return result;
}*/

template<typename T>
static void ConvertToCyclic(std::shared_ptr<T[]> data, std::shared_ptr<T[]> ans, uint16_t M, uint16_t N) {
    for (std::size_t i = 0; i < N; i++) {
        for (std::size_t j = 0; j < M; j++) {
            ans[i * M + j] = data[((i + j) * M + j) % (M*N)]; //TODO : ADD parallerism multi-threaded
        }
    }
}

template<typename T>
static void ConvertFromCyclic(std::shared_ptr<T[]> data, std::shared_ptr<T[]> ans, uint16_t M, uint16_t N) {
    for (std::size_t i = 0; i < N; i++) {
        for (std::size_t j = 0; j < M; j++) {
            ans[((i + j) * M + j) % (M*N)] = data[i * M + j];
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

//template<typename T>
//bool RsDecode(T* data , T* ans,
//               schifra::reed_solomon::erasure_locations_t erasure_location_list,
//    std::shared_ptr<const schifra::galois::field> field)
//{
//
//    /* Instantiate Decoder (Codec) */
//    typedef schifra::reed_solomon::decoder<code_length, fec_length, data_length> decoder_t;
//    const decoder_t decoder(*field.get(), generator_polynomial_index);
//
//    std::vector<uint16_t> dataVec(data , data + code_length);
//
//    schifra::reed_solomon::block<code_length, fec_length> block;
//
//    schifra::reed_solomon::copy(&dataVec[0], dataVec.size(), block);
//
//
//    if (!decoder.decode(block, erasure_location_list))
//    {
//        std::cout << "Error - Critical decoding failure! "
//            << "Msg: " << block.error_as_string() << std::endl;
//        return false;
//    }
//    /*else if (!schifra::are_blocks_equivelent(block, original_block, code_length, true, true))
//    {
//        std::cout << "Error - Error correction failed!" << std::endl;
//        return false;
//    }*/
//    else
//    {
//        std::cout << "Decryption Finished" << std::endl;
//        std::copy_n(block.data, data_length, ans);
//        return true;
//    }
//
//}

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

    std::shared_ptr<uint16_t[]> data(new uint16_t[total_data_len]);
    std::shared_ptr<uint16_t[]> cyclic_data(new uint16_t[total_data_len]);
    std::shared_ptr<uint16_t[]> encryptedData(new uint16_t[total_code_len]);
    std::shared_ptr<uint16_t[]> encryptedData_no_padding(new uint16_t[(M_without_padding + t2)*N]);
    std::shared_ptr<uint16_t[]> encrypted_no_cyclic_Data(new uint16_t[(M_without_padding + t2) * N]);
    std::shared_ptr<uint16_t[]> decryptedData(new uint16_t[total_data_len]);
    std::shared_ptr<uint16_t[]> finalRes(new uint16_t[total_data_len]);


    FILE* filp_in = fopen("c1_encoding_in.bin", "rb");
    int bytes_read = fread(data.get(), sizeof(uint16_t), total_data_len, filp_in);


    for (size_t i = 0; i < total_data_len; i++)
    {
        data[i] = __builtin_bswap16(data[i]); //data[i] = _byteswap_ushort(data[i]);
    }

    ConvertToCyclic<uint16_t>(data, cyclic_data, M, N);

    std::vector<std::thread> threads;

    //Encryption
    schifra::utils::timer encryptionTimer;
    encryptionTimer.start();
    for (std::size_t  i = 0; i < N; i++) {
                threads.push_back(std::thread(RsEncode<uint16_t>,
                                            cyclic_data.get() + (i * M),
                                            encryptedData.get() + (i* (M + t2)),
                                            field, generator_polynomial  ));
    }
    for (std::size_t i = 0; i < N; i++) {
        threads[i].join();
    }
    encryptionTimer.stop();

    std::cout << "Total encryption time: " << encryptionTimer.time() << std::endl;



    //remove padding

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M_without_padding; ++j)
        {
            encryptedData_no_padding[i * (M_without_padding + t2) + j] = encryptedData[i * (M + t2) + j];
        }
        for (std::size_t j = 0; j < t2; ++j)

        {
            encryptedData_no_padding[i * (M_without_padding + t2) + M_without_padding + j] = encryptedData[i * (M + t2) + M + j];
        }
    }

    ConvertFromCyclic<uint16_t>(encryptedData_no_padding, encrypted_no_cyclic_Data, M_without_padding + t2, N);


    //write encrypted data to file
    for (size_t i = 0; i < (M_without_padding + t2) * N; i++)
    {
        /* encrypted_no_cyclic_Data[i] = _byteswap_ushort(encrypted_no_cyclic_Data[i]); */
        encrypted_no_cyclic_Data[i] = __builtin_bswap16(encrypted_no_cyclic_Data[i]);
    }

    FILE* filp_out = fopen("c1_encoding_out.bin", "wb");
    fwrite(encrypted_no_cyclic_Data.get(), sizeof(uint16_t), (M_without_padding + t2)* N, filp_out);
    fclose(filp_out);

    return 0;
}
