#include <iostream>
#include <cmath>
#include <armadillo>

class NeuralNetwork{
    public:
        NeuralNetwork();
        NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, float learning_rate){
            i_nodes = input_nodes;
            h_nodes = hidden_nodes;
            o_nodes = output_nodes;
            lr = learning_rate;

            h_weights = arma::randn(h_nodes);
            o_weights = arma::randn(o_nodes);
        }
        ~NeuralNetwork(){}

        float sigmoid(float value){
            return 1/(1 + std::pow(arma::datum::e, -value));
        }

        void forward(arma::vec inputs){
            arma::vec h_inputs = arma::dot(h_weights, inputs);
            arma::vec h_outputs = h_inputs.transform([](float x){sigmoid(x);});
        }

        void get_weights(){
            h_weights.print();
            o_weights.print();
        }
        
    
    private:
        int i_nodes;
        int h_nodes;
        int o_nodes;
        float lr; // learning rate

        arma::vec h_weights;
        arma::vec o_weights;

        
};

int main(){
    NeuralNetwork n(3, 3, 3, 0.5);

    // std::cout << n.sigmoid(0) << std::endl;
}