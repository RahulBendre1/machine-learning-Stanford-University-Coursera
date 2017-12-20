package com.mkis.assignments.neuralnetwork;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * For this exercise, you will use logistic regression and neural networks to
 * recognize handwritten digits (from 0 to 9). Automated handwritten digit
 * recognition is widely used today - from recognizing zip codes (postal codes)
 * on mail envelopes to recognizing amounts written on bank checks. This
 * exercise will show you how the methods you've learned can be used for this
 * classification task.
 * There are 5000 training examples (originally .mat file), where each training
 * example is a 20 pixel by 20 pixel grayscale image of the digit.
 * <p>
 * The .mat file were converted to a txt file with Octave.
 * One instance/training example is 400 variables (greyscale pixel float),
 * and 1 (401st attribute) class variable (1-10, 10 is the zero!).
 * The network consists of an input layer (400 neurons + the bias), a hidden layer with 200 neurons + the bias and the output layer of the 10 classes(neurons)
 */

/**Special thanks go to Ryan Harris for his perfect explanation of the subject (youtube).*/

public class FeedForwardAndBackPropagation {

    private double[][] output; //output of every neuron, indexes: layer, neuron
    private double[][][] weights; //indexes: layer, neuron, neuron in the previous layer that is connected with
    private double[][] bias; //bias weights
    private double[][] errors; //error of every neuron, indexes: layer, neuron
    private double[][] sigmoid_derivatives; //derivatives of every neuron, indexes: layer, neuron

    private final int[] NETWORK_LAYER_SIZES; //neuron/nodes in each layer
    private final int INPUT_SIZE; //number of input neurons
    private final int OUTPUT_SIZE; //number of output neurons
    private final int NETWORK_SIZE; //amount of layers in the network

    private static NumberFormat nf = new DecimalFormat("##.##");

    public static void main(String[] args) throws java.io.IOException {

        FeedForwardAndBackPropagation test = new FeedForwardAndBackPropagation(400, 200, 10);
        LoadData loadData = new LoadData();
        String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\neuralnetwork\\data1.txt";
        List<LoadData.Instance> dataSet = loadData.loadData(file, true);
        if (dataSet.get(0).inputVariables.length != test.INPUT_SIZE || dataSet.get(0).classValues.length != test.OUTPUT_SIZE) {
            System.out.println("The number of neurons in the input layer has to match the number of input variables, also " +
                    "the number of neurons in the output layer has to match the number of classes.");
            return;
        }
        test.train(dataSet, 200, 0.5);

        System.out.println("\nAccuracy tested on training dataset: " + test.calcAccuracyOfModel(dataSet) + " %\n");
        System.out.println("\nBiases and Weights: \n");
        for (int layer = 1; layer < test.NETWORK_SIZE; layer++) {
            System.out.println("Layer " + layer + ":");
            for (int neuron = 0; neuron < test.NETWORK_LAYER_SIZES[layer]; neuron++) {
                System.out.println(test.bias[layer][neuron]);
                System.out.println(Arrays.toString(test.weights[layer][neuron]));
            }
        }
        //Randomly selected row from the dataset (only the variables ofc)
        double[] testValue = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000238, -0.00129, -0.014578, -0.020715, -0.01944, -0.007742, 0.000058, 0.00007, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.006307, 0.020444, 0.513928, 0.677929, 0.298813, 0.056915, -0.008026, 0.000021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.007176, 0.036011, 0.453742, 0.832098, 1.004046, 0.557563, 0.027653, -0.007432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000325, -0.00376, -0.012749, 0.134303, 0.753256, 0.976497, 0.120639, -0.018443, 0, 0, 0, 0, 0, -0, -0, 0, 0, 0, 0, 0, 0.000031, 0.000061, -0.006126, -0.019472, 0.579224, 1.049207, 0.144629, -0.020939, 0, 0, 0, 0.000241, -0.001412, -0.016668, -0.010865, 0.00049, 0.000073, -0.000308, -0.00447, -0.002123, 0.000131, 0, -0.002664, -0.005319, 0.58279, 1.052438, 0.145767, -0.021062, 0, 0, 0, -0.000084, -0.01693, 0.298567, 0.460269, -0.004085, -0.001583, -0.006337, -0.002011, -0.011324, 0.00064, 0.000098, -0.0056, 0.019079, 0.819844, 0.791951, 0.064307, -0.012297, 0, 0, 0.000187, -0.026739, 0.188917, 0.902936, 0.43188, -0.022156, -0.012559, 0.073476, 0.491621, 0.319049, -0.024066, -0.000353, -0.028202, 0.304376, 1.007257, 0.534364, -0.016419, -0.003601, 0, 0, -0.002467, -0.025656, 0.49501, 0.92033, 0.092737, -0.011055, -0.029162, 0.406327, 1.044801, 0.75352, 0.007596, -0.0456, 0.183718, 0.844726, 0.940465, 0.236931, -0.030711, 0.00008, 0, 0, -0.011948, 0.061037, 0.786943, 0.735692, 0.005065, -0.012599, 0.033774, 0.682488, 1.048655, 0.924586, 0.304997, 0.283057, 0.878936, 1.00768, 0.391209, -0.00703, -0.002232, 0.000028, 0, 0, -0.014933, 0.088879, 0.875892, 0.633767, -0.000041, -0.032064, 0.4444, 0.974917, 0.606141, 0.964741, 0.904645, 0.991414, 0.958816, 0.510526, 0.020711, -0.006175, 0.000256, 0, 0, 0, -0.017701, 0.114577, 0.957316, 0.580026, -0.04571, 0.253835, 0.944223, 0.625163, 0.02941, 0.480822, 0.920977, 0.682414, 0.229035, 0.00001, -0.007604, 0.00028, 0, 0, 0, 0, -0.017839, 0.115849, 0.95508, 0.663875, 0.283374, 0.807259, 0.90133, 0.077443, -0.02009, -0.01047, 0.112147, 0.011312, -0.021737, -0.005614, 0.000229, 0.000021, 0, 0, 0, 0, -0.016207, 0.099499, 0.900587, 0.982451, 0.959932, 0.980519, 0.3237, -0.025662, 0.000778, -0.000726, -0.011671, -0.002846, 0.000332, 0, 0, 0, 0, 0, 0, 0, -0.004371, 0.000551, 0.464333, 0.889571, 0.804037, 0.317319, -0.022565, -0.003227, 0.000031, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000312, -0.007859, 0.031322, 0.150447, 0.089407, -0.0297, -0.000707, 0.000238, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000752, -0.00619, -0.020717, -0.013979, 0.000843, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        System.out.println("----------------------------------------------");
        for (int i = 0; i < 10; i++) {
            System.out.println("Possibility that the number is a " + i + ": " + nf.format(100 * test.feedForward(loadData.createFeatureNormalizedTestValue(testValue))[i]) + " %");
        }
    }

    //Architecture of the network:
    private FeedForwardAndBackPropagation(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];
        this.errors = new double[NETWORK_SIZE][];
        this.sigmoid_derivatives = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.errors[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.sigmoid_derivatives[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.bias[i] = initWeights(NETWORK_LAYER_SIZES[i]);
            //First/input Layer does not have weights
            if (i > 0) {
                weights[i] = initWeights(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1]);
            }
        }
    }

    //Feed forward method to return the outputs/activations for each neuron, input is the input variables in the dataset
    private double[] feedForward(double... input) {
        //Output of the input layer is the array of the input variables:
        this.output[0] = input;
        //Iterate through all the other (layers's) neurons to get the activations for every one of them:
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = bias[layer][neuron]; //init with the bias weight
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    //sum(activation in the previous layers's neurons * weights of the previous layers
                    sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                //activation of this neuron:
                output[layer][neuron] = sigmoid(sum);
                //derivative term of sigmoid of this neuron:
                sigmoid_derivatives[layer][neuron] = (output[layer][neuron] * (1 - output[layer][neuron]));
            }
        }
        //Output of the network at the last layer
        return output[NETWORK_SIZE - 1];
    }

    //Sigmoid function:
    private double sigmoid(double z) {
        return 1d / (1 + Math.exp(-z));
    }

    //Initialize biases/weights:
    private double[] initWeights(int size) {
        if (size < 1) {
            System.out.println("Number of neurons in a layer has to be a positive whole number.");
            return null;
        }
        double[] arr = new double[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            arr[i] = (double) random.nextInt(100) / 100;
        }
        return arr;
    }

    //Initialize weights:
    private double[][] initWeights(int size, int sizeNext) {
        if (size < 1 || sizeNext < 1) {
            System.out.println("Number of neurons in a layer has to be a positive whole number.");
            return null;
        }
        double[][] arr = new double[size][sizeNext];
        for (int i = 0; i < size; i++) {
            arr[i] = initWeights(sizeNext);
        }
        return arr;
    }

    //Mean squared error of 1 training example
    private double MSE(double[] input, double[] target) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) return 0;
        feedForward(input);
        double v = 0;
        for (int i = 0; i < target.length; i++) {
            v += Math.pow((target[i] - output[NETWORK_SIZE - 1][i]), 2);
        }
        return v / (2d * target.length);
    }

    //Error of the training set - cost function
    private double costFunction(List<LoadData.Instance> instances) {
        double v = 0;
        for (LoadData.Instance instance : instances) {
            v += MSE(instance.inputVariables, instance.classValues);
        }
        return v / instances.size();
    }

    //Train the dataset
    private void train(List<LoadData.Instance> instances, int iterations, double learning_rate) {
        for (int iteration = 0; iteration < iterations; iteration++) {
            for (LoadData.Instance instance1 : instances) {
                double[] x = instance1.inputVariables;
                double[] y = instance1.classValues;
                this.train(x, y, learning_rate);
            }
            System.out.println("Error of the instance at iteration (" + iteration + "):  " + costFunction(instances));
        }
    }

    //Training 1 training example:
    private void train(double[] input, double[] target, double learning_rate) {
        feedForward(input);
        backPropError(target);
        updateWeights(learning_rate);
    }

    //Back propagation starting from the output layer's target(s)
    private void backPropError(double[] target) {
        //Error's of output neurons:
        for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
            errors[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron]) * sigmoid_derivatives[NETWORK_SIZE - 1][neuron];
        }
        //Hidden layer errors (From last hidden layer to the first), first/input layer does not have errors ofc
        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) {
                    //sum of the: previous (+1) layer's error times the weights going to that neuron from the current neuron
                    sum += weights[layer + 1][nextNeuron][neuron] * errors[layer + 1][nextNeuron];
                }
                this.errors[layer][neuron] = sum * sigmoid_derivatives[layer][neuron];
            }
        }
    }

    //First hidden layer to the output layer, 1 iteration, updating our weights: W + deltaW -> W, and Biases: B + deltaW -> B (deltaB is equal to deltaW)
    private void updateWeights(double learning_rate) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                //for the bias:
                double deltaW = -learning_rate * errors[layer][neuron];
                bias[layer][neuron] += deltaW;
                //for the rest:
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    weights[layer][neuron][prevNeuron] += deltaW * output[layer - 1][prevNeuron];
                }
            }
        }
    }

    //Prediction (1 if >= .5)
    private double predict(double[] x) {
        if (feedForward(x)[0] >= 0.5) return 1;
        return 0;
    }

    //Calculate accuracy of the network
    private double calcAccuracyOfModel(List<LoadData.Instance> instances) {
        int counter = 0;
        for (LoadData.Instance instance : instances) {
            double[] x = instance.inputVariables;
            double[] y = instance.classValues;
            if (predict(x) != y[0]) counter++;
        }
        return (1 - counter / (double) instances.size()) * 100;
    }

}
