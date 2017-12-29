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
    private double threshold = 0.5; //threshold for the prediction

    private final int[] NETWORK_LAYER_SIZES; //neuron/nodes in each layer
    private final int INPUT_SIZE; //number of input neurons
    private final int OUTPUT_SIZE; //number of output neurons
    private final int NETWORK_SIZE; //amount of layers in the network

    private static NumberFormat nf = new DecimalFormat("##.##");

    public static void main(String[] args) throws java.io.IOException {
        FeedForwardAndBackPropagation test = new FeedForwardAndBackPropagation(400, 200, 10);
        LoadData loadData = new LoadData();
        String file = "C:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\neuralnetwork\\data1.txt";
        loadData.loadData(file, true, 80);
        test.train(loadData.getTrainingSet(), 100, 1, 0);

        System.out.println("Accuracy tested on the cross-validation set: " + test.calcAccuracyOfModel(loadData.getCrossValidationSet()) + " %");

        double[] testPicture = loadData.getTestSet().get(0).inputVariables;

        for (int i = 0; i < 10; i++) {
            System.out.println("The probability that the number is a " + i + ": " + nf.format(100 * test.feedForward(testPicture)[i]) + " %");
        }
        System.out.printf("The number should be: " + Arrays.toString(loadData.getTestSet().get(0).classValues));
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
    private double calculateMSE(double[] input, double[] target) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) return 0;
        feedForward(input);
        double v = 0;
        for (int i = 0; i < target.length; i++) {
            v += Math.pow((target[i] - output[NETWORK_SIZE - 1][i]), 2);
        }
        return v / (2d * target.length);
    }

    //Error of the training set - cost function
    private double calculateTrainingSetError(List<LoadData.Instance> instances) {
        double v = 0;
        for (LoadData.Instance instance : instances) {
            v += calculateMSE(instance.inputVariables, instance.classValues);
        }
        return v / instances.size();
    }

    //Train the dataset
    private void train(List<LoadData.Instance> instances, int iterations, double learning_rate, double lambda) {
        for (int iteration = 0; iteration < iterations; iteration++) {
            for (LoadData.Instance instance1 : instances) {
                double[] x = instance1.inputVariables;
                double[] y = instance1.classValues;
                this.train(x, y, learning_rate, lambda);
            }
            System.out.println("Error of the instance at iteration (" + iteration + "):  " + calculateTrainingSetError(instances));
        }
    }

    //Training 1 training example:
    private void train(double[] input, double[] target, double learning_rate, double lambda) {
        feedForward(input);
        backPropError(target);
        updateWeights(learning_rate, lambda);
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
    private void updateWeights(double learning_rate, double lambda) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                //for the bias:
                double deltaW = -learning_rate * errors[layer][neuron];
                bias[layer][neuron] += deltaW;
                //for the rest:
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    weights[layer][neuron][prevNeuron] += deltaW * output[layer - 1][prevNeuron] + lambda * weights[layer][neuron][prevNeuron];
                }
            }
        }
    }

    //Prediction (1 if >= .5)
    private double[] predict(double[] x) {
        double[] output = feedForward(x);
        double[] predicted = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < output.length; i++) {
            if (output[i] >= threshold) {
                predicted[i] = 1;
            } else {
                predicted[i] = 0;
            }
        }
        return predicted;
    }

    //Calculate accuracy of the network
    private double calcAccuracyOfModel(List<LoadData.Instance> instances) {
        int counter = 0;
        main:
        for (LoadData.Instance instance : instances) {
            double[] x = instance.inputVariables;
            double[] y = instance.classValues;
            for(int i = 0; i < y.length; i++) {
                if (predict(x)[i] != y[i]) counter++;
                continue main;
            }
        }
        return (1 - counter / (double) instances.size()) * 100;
    }

}
