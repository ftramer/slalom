#ifndef SGXDNN_MODEL_H_
#define SGXDNN_MODEL_H_

#include <stdio.h>
#include <iostream>
#include <memory>

#include "mempool.hpp"
#include "layers/layer.hpp"
#include "layers/conv2d.hpp"
#include "layers/maxpool2d.hpp"
#include "layers/activation.hpp"
#include "layers/flatten.hpp"
#include "layers/dense.hpp"
#include "layers/padding.hpp"
#include "layers/depthwise_conv2d.hpp"
#include "layers/global_pool.hpp"
#include "layers/reshape.hpp"

#include "json11.hpp"

#ifdef USE_SGX
#include "Enclave.h"
#endif

using namespace std;

namespace SGXDNN
{

	template <typename T>
	class Model {

	public:

		explicit Model(){
			mem_pool = nullptr;
		}
		virtual ~Model(){}

		// load a model described by a json file
		void load_model(const char* model_json, T** weights, bool is_verif_mode, bool verif_preproc) {
            printf("In load model with verif mode=%d, verif_preproc=%d\n", is_verif_mode, verif_preproc);

			mem_pool = new MemPool(2, 224*224*64*sizeof(float));

			std::string err;
			auto model_obj = json11::Json::parse(model_json, err);
            printf("Json parsed\n");

            array4d temp_shape = {0, 0, 0, 0};
			int weight_idx = 0;

			for (auto &json_layer : model_obj["layers"].array_items())
			{
				auto name = json_layer["name"].string_value();

				if (name == "Input")
				{
					printf("loading Input layer (None, %d, %d, %d)\n",
							json_layer["shape"][1].int_value(),
							json_layer["shape"][2].int_value(),
							json_layer["shape"][3].int_value());
					temp_shape = {0,
								  json_layer["shape"][1].int_value(),
								  json_layer["shape"][2].int_value(),
								  json_layer["shape"][3].int_value()};

					input_shape[0] = json_layer["shape"][1].int_value();
					input_shape[1] = json_layer["shape"][2].int_value();
					input_shape[2] = json_layer["shape"][3].int_value();
				}
				else if (name == "Conv2D")
				{
					printf("loading Conv2D layer\n");

					array4d kernel_size = {
							json_layer["kernel_size"][0].int_value(),
							json_layer["kernel_size"][1].int_value(),
							json_layer["kernel_size"][2].int_value(),
							json_layer["kernel_size"][3].int_value()
					};
					auto strides = json_layer["strides"].array_items();
					auto padding = get_padding(json_layer["padding"].string_value());

					T *r_left, *r_right, *kernel, *bias;
					if (verif_preproc) {
						r_left = weights[weight_idx];
						r_right = weights[weight_idx + 1];
						kernel = weights[weight_idx + 2];
						bias = weights[weight_idx + 3];
						weight_idx += 4;
					} else {
						kernel = weights[weight_idx];
                        bias = weights[weight_idx + 1];
						weight_idx += 2;
					}

					layers.push_back(shared_ptr<Layer<T>>(
							new Conv2D<T>("conv",
										  temp_shape,
										  kernel_size,
										  strides[0].int_value(),
										  strides[1].int_value(),
										  padding,
										  r_left,
										  r_right,
										  kernel,
										  bias,
										  mem_pool,
										  is_verif_mode,
										  verif_preproc,
										  json_layer["activation"].string_value())
					));
					temp_shape = layers[layers.size()-1]->output_shape();

					if (json_layer["activation"].string_value() != "linear" ) {
						layers.push_back(std::shared_ptr<Layer<T>>(
							new Activation<T>("activation",
											  temp_shape,
											  json_layer["activation"].string_value(),
											  verif_preproc)
						));
					}

				}
				else if (name == "MaxPooling2D")
				{
					printf("loading MaxPool layer\n");

					auto pool_size = json_layer["pool_size"].array_items();
					auto strides = json_layer["strides"].array_items();
					auto padding = get_padding(json_layer["padding"].string_value());

					layers.push_back(shared_ptr<Layer<T>>(
							new MaxPool2D<T>("maxpool",
									temp_shape,
									pool_size[0].int_value(),
									pool_size[0].int_value(),
									strides[0].int_value(),
									strides[1].int_value(),
									padding,
									mem_pool)
					));
					temp_shape = layers[layers.size()-1]->output_shape();
				}
				else if (name == "Flatten")
				{
				}
				else if (name == "Dense")
				{
					printf("loading Dense layer\n");

					auto h_in = json_layer["kernel_size"][0].int_value();
					auto h_out = json_layer["kernel_size"][1].int_value();
					auto kernel = weights[weight_idx];
					auto bias = weights[weight_idx + 1];

					layers.push_back(shared_ptr<Layer<T>>(
							new Dense<T>("dense",
										 temp_shape,
										 h_in, h_out,
										 kernel, bias,
										 mem_pool,
										 is_verif_mode,
										 verif_preproc)
					));
					weight_idx += 2;
					temp_shape = layers[layers.size()-1]->output_shape();

					if (json_layer["activation"].string_value() != "linear" ) {
						layers.push_back(std::shared_ptr<Layer<T>>(
							new Activation<T>("activation",
											  temp_shape,
											  json_layer["activation"].string_value(),
											  verif_preproc)
						));
					}
				}
				else if (name == "ZeroPadding2D")
				{
					printf("loading ZeroPadding layer\n");
                    
                    auto pad_rows = json_layer["padding"][0].int_value();
                    auto pad_cols = json_layer["padding"][1].int_value();

					layers.push_back(shared_ptr<Layer<T>>(
							new Padding2D<T>("zeropad2D", temp_shape, pad_rows, pad_cols, mem_pool)
					));
					temp_shape = layers[layers.size()-1]->output_shape();
				}
				else if (name == "Activation")
				{
					printf("loading Activation\n");
					Conv2D<T>* prev_conv = dynamic_cast<Conv2D<T>*>(layers[layers.size() - 1].get());
					if (prev_conv) {
						prev_conv->set_activation_type(json_layer["type"].string_value());
					}

					layers.push_back(std::shared_ptr<Layer<T>>(
                        new Activation<T>("activation",
                                          temp_shape,
                                          json_layer["type"].string_value(),
										  verif_preproc)
                    ));
				}
				else if (name == "GlobalAveragePooling2D")
                {
                    printf("loading Global Avg Pooling layer\n");

                    layers.push_back(shared_ptr<Layer<T>>(
                            new GlobalPool<T>("globalAvgPool2D", temp_shape, mem_pool, verif_preproc)
                    ));
                    temp_shape = layers[layers.size()-1]->output_shape();
                }
				else if (name == "DepthwiseConv2D")
                {   
                    printf("loading DepthwiseConv2D layer\n");
                    
                    array4d kernel_size = {
                            json_layer["kernel_size"][0].int_value(),
                            json_layer["kernel_size"][1].int_value(),
                            json_layer["kernel_size"][2].int_value(),
							1,
                    };
                    auto strides = json_layer["strides"].array_items();
                    auto padding = get_padding(json_layer["padding"].string_value());

					T *r_left, *kernel, *bias;
                    if (verif_preproc) {
                        r_left = weights[weight_idx];
                        kernel = weights[weight_idx + 1];
                        bias = weights[weight_idx + 2];
                        weight_idx += 3;
                    } else {
                        kernel = weights[weight_idx];
                        bias = weights[weight_idx + 1];
                        weight_idx += 2;
                    }
                    
                    layers.push_back(shared_ptr<Layer<T>>(
                            new DepthwiseConv2D<T>("depthwise_conv",
                                          temp_shape,
                                          kernel_size,
                                          strides[0].int_value(),
                                          strides[1].int_value(),
                                          padding,
										  r_left,
                                          kernel,
                                          bias,
                                          mem_pool,
                                          is_verif_mode,
										  verif_preproc)
                    ));
                    temp_shape = layers[layers.size()-1]->output_shape();
					assert(json_layer["activation"] == "linear");
                }
				else if (name == "Reshape")
				{
					printf("loading Reshape layer\n");
					array3d new_shape = {
                            json_layer["shape"][0].int_value(),
                            json_layer["shape"][1].int_value(),
                            json_layer["shape"][2].int_value(),
                    };
					layers.push_back(shared_ptr<Layer<T>>(
						new Reshape<T>("reshape", temp_shape, new_shape)
					));
                    temp_shape = layers[layers.size()-1]->output_shape();
				}
				else
				{
					printf("Unknown layer: %s\n", json_layer["name"].string_value().c_str());
					assert(0);
				}
			}
		}

		int input_shape[3];
		std::vector<std::shared_ptr<Layer<T>>> layers;
		MemPool* mem_pool;

	protected:
		Eigen::PaddingType get_padding(std::string padding)
		{
			if (padding == "same")
			{
				return Eigen::PaddingType::PADDING_SAME;
			}
			else if (padding == "valid")
			{
				return Eigen::PaddingType::PADDING_SAME;
			}
			else
			{
				printf("Unknown padding");
				return Eigen::PaddingType::PADDING_VALID;
			}
		}
	};
}
#endif
