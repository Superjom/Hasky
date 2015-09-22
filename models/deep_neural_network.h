#pragma once
#include "../layers/neuron_layer.h"
#include "../layers/map_layer.h"
#include "../layers/loss_layer.h"
#include "../layers/neural_network_layer.h"

template<typename T>
class DeepNeuralNetworks {
public:
    typedef typename Layer<T>::param_t param_t;
    typedef typename Layer<T>::vec_t vec_t;
    /*
     * @sizes: length of hidden vector, from bottom to top
     */
    void setup(vector<int> sizes) {
        CHECK_GT(sizes[0], 1) << "input layer should have more than 1 element";
        CHECK_EQ(sizes[ sizes.size()-1 ], 1) << "output layer should have only 1 element";
        shape_t shape;
        setup_input_layer(sizes[0]);
        shape.size = sizes[ sizes.size() - 1];
        shape.width = shape.size;
        setup_output_layer(shape);
        // setup sigmoid layers and neuron layers
        layers.resize(sizes.size());
        for (int i = 1; i < sizes.size() - 1; i++) {
            LOG(WARNING) << "-- setup " << i << "th layer";
            layers[i].reset(new NeuronNetworkLayer<T>);
            auto& layer = *layers[i];
            layer.set_name(swift_snails::format_string("neuron-%d", i));
            shape = {sizes[i], sizes[i-1]};
            layer.setup(shape);
            if (i > 1) {
                layer.set_bottom_layer(layers[i-1].get());
                layers[i-1]->set_top_layer(layers[i].get());
            }
        }
        layers[ layers.size() - 1].reset(new NeuronLayer<T>);
        auto& layer = *layers[ layers.size() - 1];
        layer.set_bottom_layer(layers[ layers.size() - 2].get());
        layers[1]->set_bottom_layer(&data_layer);
        layer.set_top_layer(&rmse_loss_layer);
        rmse_loss_layer.set_bottom_layer(layers[layers.size()-1].get());
    }

    float learn(vec_t& vec, T label) {
        rmse_loss_layer.param().label()[0] = label;
        data_layer.forward(vec);
        //LOG(INFO) << data_layer.name() << "\t>>\t" << data_layer.param().z();
        Layer<T>* layer = layers[1].get();
        // forward
        //DLOG(INFO) << "forward ...";
        while(layer != nullptr) {
            CHECK(layer->bottom_layer() != nullptr) << layer->name();
            //DLOG(INFO) << ".. layer." << layer->name() << " forward from\t" << layer->bottom_layer()->name();
            CHECK(layer->bottom_layer() != nullptr) << layer->name();
            layer->forward(layer->bottom_layer()->param());
            //LOG(INFO) << layer->name() << "\t>>\t" << layer->param().z();
            if (layer->kind() == OUTPUT_LAYER) break;
            if (layer->kind() == HIDDEN_LAYER) layer = layer->top_layer();
        }
        auto loss = rmse_loss_layer.param().z()[0];
        //DLOG(INFO) << "backward ...";
        // backward
        rmse_loss_layer.backward(layers[layers.size()-1]->param());
        layer = layers[ layers.size() - 1].get();
        while (layer != nullptr) {
            if (layer->kind() != INPUT_LAYER) 
            {
                CHECK(layer->top_layer() != nullptr) << layer->name();
                CHECK(layer->bottom_layer() != nullptr) << layer->name();
                layer->backward(
                    layer->top_layer()->param(), 
                    layer->bottom_layer()->param());
                //DLOG(INFO) << ".. layer." << layer->name() << " backward from\t" << layer->top_layer()->name();
                //LOG(INFO) << layer->name() << "\t>>\t" << layer->param().loss();
            } else break;
            layer = layer->bottom_layer();
        }
        return loss;
    }

    T predict(vec_t& vec) {

    }
protected:
    void setup_input_layer(int size) {
        data_layer.set_name("INPUT");
        data_layer.setup(size);
    }
    /*
     * set sigmoid as output function
     */
    void setup_output_layer(const shape_t& shape) {
        rmse_loss_layer.set_name("RMSE");
        rmse_loss_layer.setup(shape);
    }
    /*
     * a neural-network-layer is a neuron-layer with a tanh-layer
     */
    void set_neural_network_layer(int id, const shape_t &shape_, Layer<T>& neuron_layer, Layer<T>& tanh_layer) {
        shape_t shape = shape_;
        neuron_layer.set_name("neuron-%d", id);
        tanh_layer.set_name("tanh-%d", id);
        neuron_layer.gaus_dist().init(0, 0.7);
        neuron_layer.setup(shape);
        shape.width = shape.size;
        tanh_layer.setup(shape);
    }

private:
    DataLayer<T> data_layer;
    // neuron layers and sigmoid layers
    vector<std::shared_ptr<Layer<T> > > layers;
    RMSELayer<T> rmse_loss_layer;
};
