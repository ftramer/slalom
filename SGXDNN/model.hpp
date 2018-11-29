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
#include "layers/block.hpp"

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

		void load_layer(const json11::Json& json_layer, T** weights, int *weight_idx, array4d *temp_shape,
						bool is_verif_mode, bool verif_preproc, std::vector<std::shared_ptr<Layer<T>>> &layers, Layer<T> *prev_layer) 
		{
			auto name = json_layer["name"].string_value();

			if (name == "Input")
			{
				printf("loading Input layer (None, %d, %d, %d)\n",
						json_layer["shape"][1].int_value(),
						json_layer["shape"][2].int_value(),
						json_layer["shape"][3].int_value());
				*temp_shape = {0,
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
					r_left = weights[*weight_idx];
					r_right = weights[*weight_idx + 1];
					kernel = weights[*weight_idx + 2];
					bias = weights[*weight_idx + 3];
					*weight_idx += 4;
				} else {
					kernel = weights[*weight_idx];
					bias = weights[*weight_idx + 1];
					*weight_idx += 2;
				}

				layers.push_back(shared_ptr<Layer<T>>(
						new Conv2D<T>("conv",
									  *temp_shape,
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
				*temp_shape = layers[layers.size()-1]->output_shape();

				if (json_layer["activation"].string_value() != "linear") {
					layers.push_back(std::shared_ptr<Layer<T>>(
						new Activation<T>("activation",
										  *temp_shape,
										  json_layer["activation"].string_value(),
										  json_layer["bits_w"].int_value(),
										  json_layer["bits_x"].int_value(),
										  verif_preproc)
					));
				}
			}
			else if (name == "MaxPooling2D")
			{
				printf("loading MaxPool2D layer\n");

				auto pool_size = json_layer["pool_size"].array_items();
				auto strides = json_layer["strides"].array_items();
				auto padding = get_padding(json_layer["padding"].string_value());

				layers.push_back(shared_ptr<Layer<T>>(
						new MaxPool2D<T>("maxpool",
								*temp_shape,
								pool_size[0].int_value(),
								pool_size[0].int_value(),
								strides[0].int_value(),
								strides[1].int_value(),
								padding,
								false,
								mem_pool)
				));
				*temp_shape = layers[layers.size()-1]->output_shape();
			}
			else if (name == "AveragePooling2D")
			{
				printf("loading AvgPool2D layer\n");

				auto pool_size = json_layer["pool_size"].array_items();
				auto strides = json_layer["strides"].array_items();
				auto padding = get_padding(json_layer["padding"].string_value());

				layers.push_back(shared_ptr<Layer<T>>(
						new MaxPool2D<T>("avgpool",
								*temp_shape,
								pool_size[0].int_value(),
								pool_size[0].int_value(),
								strides[0].int_value(),
								strides[1].int_value(),
								padding,
								true,
								mem_pool)
				));
				*temp_shape = layers[layers.size()-1]->output_shape();
			}
			else if (name == "Flatten")
			{
			}
			else if (name == "Dense")
			{
				printf("loading Dense layer\n");

				auto h_in = json_layer["kernel_size"][0].int_value();
				auto h_out = json_layer["kernel_size"][1].int_value();
				auto kernel = weights[*weight_idx];
				auto bias = weights[*weight_idx + 1];

				layers.push_back(shared_ptr<Layer<T>>(
						new Dense<T>("dense",
									 *temp_shape,
									 h_in, h_out,
									 kernel, bias,
									 mem_pool,
									 is_verif_mode,
									 verif_preproc)
				));
				*weight_idx += 2;
				*temp_shape = layers[layers.size()-1]->output_shape();

				if (json_layer["activation"].string_value() != "linear" ) {
					layers.push_back(std::shared_ptr<Layer<T>>(
						new Activation<T>("activation",
										  *temp_shape,
										  json_layer["activation"].string_value(),
										  json_layer["bits_w"].int_value(),
										  json_layer["bits_x"].int_value(),
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
						new Padding2D<T>("zeropad2D", *temp_shape, pad_rows, pad_cols, mem_pool)
				));
				*temp_shape = layers[layers.size()-1]->output_shape();
			}
			else if (name == "Activation")
			{
				printf("loading Activation\n");
				Conv2D<T>* prev_conv = dynamic_cast<Conv2D<T>*>(prev_layer);
				if (prev_conv) {
					prev_conv->set_activation_type(json_layer["type"].string_value());
				}

				layers.push_back(std::shared_ptr<Layer<T>>(
					new Activation<T>("activation",
									  *temp_shape,
									  json_layer["type"].string_value(),
									  json_layer["bits_w"].int_value(),
									  json_layer["bits_x"].int_value(),
									  verif_preproc)
				));
			}
			else if (name == "GlobalAveragePooling2D")
			{
				printf("loading Global Avg Pooling layer\n");

				layers.push_back(shared_ptr<Layer<T>>(
						new GlobalPool<T>("globalAvgPool2D", *temp_shape, mem_pool, verif_preproc)
				));
				*temp_shape = layers[layers.size()-1]->output_shape();
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
					r_left = weights[*weight_idx];
					kernel = weights[*weight_idx + 1];
					bias = weights[*weight_idx + 2];
					*weight_idx += 3;
				} else {
					kernel = weights[*weight_idx];
					bias = weights[*weight_idx + 1];
					*weight_idx += 2;
				}

				layers.push_back(shared_ptr<Layer<T>>(
						new DepthwiseConv2D<T>("depthwise_conv",
									  *temp_shape,
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
				*temp_shape = layers[layers.size()-1]->output_shape();
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
					new Reshape<T>("reshape", *temp_shape, new_shape)
				));
				*temp_shape = layers[layers.size()-1]->output_shape();
			}
			else if (name == "ResNetBlock")
			{
				printf("loading ResNetBlock layer\n");

				std::vector<std::shared_ptr<Layer<T>>> path1;
				std::vector<std::shared_ptr<Layer<T>>> path2;
				array4d temp_shape2 = {(*temp_shape)[0], (*temp_shape)[1], (*temp_shape)[2], (*temp_shape)[3]};
				Layer<T> *sub_prev_layer;
				for (auto &json_sublayer : json_layer["path1"].array_items()) {
					load_layer(json_sublayer, weights, weight_idx, temp_shape, is_verif_mode, verif_preproc, path1, sub_prev_layer);
					sub_prev_layer = path1[path1.size() - 1].get();
				}

				printf("loaded path1\n");

				sub_prev_layer = nullptr;
				for (auto &json_sublayer : json_layer["path2"].array_items()) {
					load_layer(json_sublayer, weights, weight_idx, &temp_shape2, is_verif_mode, verif_preproc, path2, sub_prev_layer);
					sub_prev_layer = path2[path2.size() - 1].get();
				}

				printf("loaded path2\n");

				layers.push_back(shared_ptr<Layer<T>>(
						new ResNetBlock<T>("resnet_block",
										   *temp_shape,
										   json_layer["identity"].bool_value(),
										   path1, 
										   path2,
										   json_layer["bits_w"].int_value(),
                                      	   json_layer["bits_x"].int_value(), 
										   mem_pool)
				));
				*temp_shape = layers[layers.size()-1]->output_shape();

				printf("loaded ResNetBlock layer\n");
			}
			else
			{
				printf("Unknown layer: %s\n", json_layer["name"].string_value().c_str());
				assert(0);
			}
		}

		// load a model described by a json file
		void load_model(const char* model_json, T** weights, bool is_verif_mode, bool verif_preproc) {
            printf("In load model with verif mode=%d, verif_preproc=%d\n", is_verif_mode, verif_preproc);

            std::string err;
			auto model_obj = json11::Json::parse(model_json, err);
            printf("Json parsed\n");

            int shift_w = model_obj["shift_w"].int_value();
            int shift_x = model_obj["shift_x"].int_value();
            inv_shift8f = _mm256_set1_ps((float)(1.0/shift_w));
            six8f = _mm256_set1_ps((float) 6 * shift_w * shift_x);

            int max_tensor_size = model_obj["max_tensor_size"].int_value();
			mem_pool = new MemPool(2, max_tensor_size*sizeof(float));

            array4d temp_shape = {0, 0, 0, 0};
			int weight_idx = 0;

			Layer<T> *prev_layer;

			for (auto &json_layer : model_obj["layers"].array_items())
			{
				std::vector<std::shared_ptr<Layer<T>>> new_layers;

				load_layer(json_layer, weights, &weight_idx, &temp_shape, is_verif_mode, verif_preproc, new_layers, prev_layer);

				layers.insert(layers.end(), new_layers.begin(), new_layers.end());
				if (layers.size() > 0) {
					prev_layer = layers[layers.size() - 1].get();
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
				return Eigen::PaddingType::PADDING_VALID;
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
