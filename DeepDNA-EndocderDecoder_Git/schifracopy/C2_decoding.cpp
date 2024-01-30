
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
#include <sstream>
//#include "json11.hpp"

#include "CommonDefinitions.h"


auto constexpr N = 2;
auto constexpr c2_redundency = 2 * t1 + t2;
auto constexpr M = 65535 - c2_redundency;
auto constexpr total_data_len = M * N;
auto constexpr total_code_len = (M + c2_redundency) * N;
auto constexpr word_size_in_bits = M * 16; // 8 symbols per letter

/* Finite Field Parameters */
std::size_t constexpr field_descriptor = 16;
std::size_t constexpr generator_polynomial_index = 0; //0
std::size_t constexpr generator_polynomial_root_count = c2_redundency;

/* Reed Solomon Code Parameters */
std::size_t constexpr code_length = M + c2_redundency;
std::size_t constexpr fec_length = c2_redundency;
std::size_t constexpr data_length = code_length - fec_length;

template<typename T>
bool RsDecode(T* data , T* ans,
               schifra::reed_solomon::erasure_locations_t erasure_location_list, 
    std::shared_ptr<const schifra::galois::field> field)
{

    /* Instantiate Decoder (Codec) */
    typedef schifra::reed_solomon::decoder<code_length, fec_length, data_length> decoder_t;
    const decoder_t decoder(*field.get(), generator_polynomial_index);

    std::vector<uint16_t> dataVec(data , data + code_length);

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
        std::cout << "Decryption Finished" << std::endl;
        std::copy_n(block.data, code_length, ans);
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

    std::shared_ptr<uint16_t[]> encryptedData(new uint16_t[total_code_len]);
    std::shared_ptr<uint16_t[]> decryptedData(new uint16_t[total_code_len]);

    //for (std::size_t i = 0; i < total_data_len; ++i)
    //{
    //    data[i] = static_cast<uint16_t>(i + 1);
    //}

    FILE* filp = fopen("c2_decoding_in.bin", "rb");
    int bytes_read = fread(encryptedData.get(), sizeof(uint16_t), total_code_len, filp);


    for (size_t i = 0; i < total_code_len; i++)
    {
        //encryptedData[i] = _byteswap_ushort(encryptedData[i]);
        encryptedData[i] = __builtin_bswap16(encryptedData[i]);
    }

   
    schifra::reed_solomon::erasure_locations_t erasure_location_list;
    std::ifstream myfile("c2_erasures_locations.txt");
    std::copy(std::istream_iterator<size_t>(myfile),
        std::istream_iterator<size_t>(),
        back_inserter(erasure_location_list));
    std::vector<std::thread> threads;

    threads.clear();

    //Decryption
    schifra::utils::timer decryptionTimer;
    decryptionTimer.start();

    std::cout << "HI: "  << std::endl;


    for (std::size_t i = 0; i < N; i++) {
        threads.push_back(std::thread(RsDecode<uint16_t>,
            encryptedData.get() + (i * (M + c2_redundency)),
            decryptedData.get() + (i * (M + c2_redundency)),
            erasure_location_list, field));
    }


    for (std::size_t i = 0; i < N; i++) {
        threads[i].join();
    }

    decryptionTimer.stop();

    std::cout << "Total decryption time: " << decryptionTimer.time() << std::endl;


    timer.stop();
    double total_time = timer.time();
 
    //write decrypted data to file
    for (size_t i = 0; i < total_code_len; i++)
    {
        //decryptedData[i] = _byteswap_ushort(decryptedData[i]);
        decryptedData[i] = __builtin_bswap16(decryptedData[i]);
    }

    FILE* filp_out = fopen("c2_decoding_out.bin", "wb");
    fwrite(decryptedData.get(), sizeof(uint16_t), total_code_len, filp_out);
    fclose(filp_out);

    return 0;
}
