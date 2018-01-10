package com.mkis.assignments.neuralnetwork;

import org.jfree.ui.RefineryUtilities;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
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

/**
 * Special thanks go to Ryan Harris for his perfect explanation of the subject (youtube).
 */

public class FeedForwardAndBackPropagation {

    private double[][] outputs; //output of every neuron, indexes: layer, neuron
    private double[][][] weights; //indexes: layer, neuron, neuron in the previous layer that is connected with
    private double[][] biases; //bias weights
    private double[][] errors; //error of every neuron, indexes: layer, neuron
    private double[][] derivatives; //derivatives of every neuron, indexes: layer, neuron

    private double[][] initialOutput; //initial outputs
    private double[][][] initialWeights; //initial weights
    private double[][] initialBiases; //initial biases
    private double[][] initialErrors; //initial errors
    private double[][] initialDerivatives; //initial derivatives

    private double m; //number of training examples

    private static List<Double> trainingSetCostFunction = new ArrayList<>();
    private static List<Double> CVSetSetCostFunction = new ArrayList<>();

    private final int[] NETWORK_LAYER_SIZES; //neuron/nodes in each layer
    private final int NETWORK_SIZE; //amount of layers in the network

    private static NumberFormat nf = new DecimalFormat("##.##");

    public static void main(String[] args) throws java.io.IOException {

        String file = "D" +
                ":\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\neuralnetwork\\data1.txt";
        FeedForwardAndBackPropagation test = new FeedForwardAndBackPropagation(400, 200, 10);
        LoadData loadData = new LoadData();
        loadData.loadData(file, true, 80);
        List<LoadData.Instance> trSet = loadData.getTrainingSet();
        List<LoadData.Instance> cvSet = loadData.getCrossValidationSet();
        List<LoadData.Instance> testSet = loadData.getTestSet();

        double[] lambda = new double[]{0, 0.01, 0.03, 0.05, 0.08, 0.1, 1, 3, 4};

        for (int i = 0; i < lambda.length; i++) {
            System.out.println("\nTraining for lambda (" + lambda[i] + ")...");
            test.weights = test.initialWeights;
            test.biases = test.initialBiases;
            test.outputs = test.initialOutput;
            test.derivatives = test.initialDerivatives;
            test.errors = test.initialErrors;
            test.train(trSet, cvSet, 100, 1, lambda[i]);
            //Prediction with no regularization
            if (i == 0) {
                System.out.println("\nAccuracy tested on the cross-validation set: " + test.calcAccuracyOfModel(cvSet) + " %");

                double[] testPicture = testSet.get(0).inputVariables;

                for (int classV = 0; classV < 10; classV++) {
                    System.out.println("The probability that the number is a " + classV + ": " + nf.format(100 * test.feedForward(testPicture)[classV]) + " %");
                }
                System.out.printf("The number should be: " + Arrays.toString(testSet.get(0).classValues) + "\n");
            }
        }

        //Plotting/evaluation
        Evaluation eval = new Evaluation("Evaluation", trainingSetCostFunction, CVSetSetCostFunction, lambda);
        eval.pack();
        RefineryUtilities.centerFrameOnScreen(eval);
        eval.setResizable(true);
        eval.setVisible(true);
    }

    //Architecture of the network:
    private FeedForwardAndBackPropagation(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;

        this.outputs = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.biases = new double[NETWORK_SIZE][];
        this.errors = new double[NETWORK_SIZE][];
        this.derivatives = new double[NETWORK_SIZE][];

        this.initialOutput = new double[NETWORK_SIZE][];
        this.initialWeights = new double[NETWORK_SIZE][][];
        this.initialBiases = new double[NETWORK_SIZE][];
        this.initialErrors = new double[NETWORK_SIZE][];
        this.initialDerivatives = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.outputs[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.errors[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.derivatives[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.biases[i] = initWeights(NETWORK_LAYER_SIZES[i]);

            this.initialOutput[i] = this.outputs[i];
            this.initialErrors[i] = this.errors[i];
            this.initialDerivatives[i] = this.derivatives[i];
            this.initialBiases[i] = this.biases[i];

            //First/input Layer does not have weights
            if (i > 0) {
                this.weights[i] = initWeights(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1]);
                this.initialWeights[i] = this.weights[i];
            }
        }
    }

    //Feed forward method to return the outputs/activations for each neuron, input is the input variables in the dataset
    private double[] feedForward(double... input) {
        //Output of the input layer is the array of the input variables:
        this.outputs[0] = input;
        //Iterate through all the other (layers's) neurons to get the activations for every one of them:
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = this.biases[layer][neuron]; //init with the bias weight
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    //sum(activation in the previous layers's neurons * weights of the previous layers
                    sum += this.outputs[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                //activation of this neuron:
                this.outputs[layer][neuron] = sigmoid(sum);
                //derivative term of sigmoid of this neuron:
                this.derivatives[layer][neuron] = (this.outputs[layer][neuron] * (1 - this.outputs[layer][neuron]));
            }
        }
        //Output of the network at the last layer
        return outputs[NETWORK_SIZE - 1];
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

    //Train
    private void train(List<LoadData.Instance> trainingSet, List<LoadData.Instance> CVSet, int iterations, double learning_rate, double lambda) {
        this.m = (double) trainingSet.size();
        for (int iteration = 0; iteration < iterations; iteration++) {
            for (LoadData.Instance instance1 : trainingSet) {
                double[] x = instance1.inputVariables;
                double[] y = instance1.classValues;
                this.train(x, y, learning_rate, lambda);
            }
            if (iteration == iterations - 1) {
                trainingSetCostFunction.add(createRegularizedCostFunction(trainingSet, lambda));
                CVSetSetCostFunction.add(createCostFunction(CVSet));
            }
        }
    }

    //Training (1 training example):
    private void train(double[] input, double[] target, double learning_rate, double lambda) {
        feedForward(input);
        backPropError(target);
        updateWeights(learning_rate, lambda);
    }

    //Creating regularized cost function for evaluation
    private double createRegularizedCostFunction(List<LoadData.Instance> instances, double lambda) {
        double cost = 0.0;
        for (LoadData.Instance instance : instances) {
            double[] x = instance.inputVariables;
            double[] y = instance.classValues;
            double classSum = 0.0;
            for (int classNumber = 0; classNumber < y.length; classNumber++) {
                if (feedForward(x)[classNumber] == 0 || feedForward(x)[classNumber] == 1) continue; //avoiding NaN
                classSum += y[classNumber] * Math.log(feedForward(x)[classNumber]) + (1 - y[classNumber]) * Math.log(1 - feedForward(x)[classNumber]); //regularized
            }
            cost += classSum;
        }
        /*//The weights:
        for(int i= 1; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for(int z = 0; z < weights[i][j].length; z++) {
                    System.out.println("weight[layer:"+i+"][from "+z+" mode][to "+j+"th node] :" + weights[i][j][z]);
                }
            }
        }*/
        double regSum = 0;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    regSum += Math.pow(this.weights[layer][neuron][prevNeuron], 2);
                }
            }
        }
        return (-1 * cost / m) + regSum * lambda / (2 * m);
    }

    //Create cost function (not regularized) for CV set for evaluation
    private double createCostFunction(List<LoadData.Instance> instances) {
        double cost = 0.0;
        for (LoadData.Instance instance : instances) {
            double[] x = instance.inputVariables;
            double[] y = instance.classValues;
            double classSum = 0.0;
            for (int classNumber = 0; classNumber < y.length; classNumber++) {
                if (feedForward(x)[classNumber] == 0 || feedForward(x)[classNumber] == 1) continue; //avoiding NaN
                classSum += y[classNumber] * Math.log(feedForward(x)[classNumber]) + (1 - y[classNumber]) * Math.log(1 - feedForward(x)[classNumber]); //not regularized
            }
            cost += classSum;
        }
        return -1 * cost / (double) instances.size();
    }

    //Back propagation starting from the output layer's target(s)
    private void backPropError(double[] target) {
        //Error's of output neurons:
        for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
            this.errors[NETWORK_SIZE - 1][neuron] = outputs[NETWORK_SIZE - 1][neuron] - target[neuron];
        }
        //Hidden layer errors (From last hidden layer to the first), first/input layer does not have errors ofc
        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) {
                    //sum of the: previous (+1) layer's error times the weights going to that neuron from the current neuron
                    sum += weights[layer + 1][nextNeuron][neuron] * errors[layer + 1][nextNeuron];
                }
                this.errors[layer][neuron] = sum * derivatives[layer][neuron];
            }
        }
    }

    //First hidden layer to the output layer, 1 iteration, updating the biases and the weights with L2 regularization
    private void updateWeights(double learning_rate, double lambda) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                //for the bias:
                double deltaW = -learning_rate * errors[layer][neuron];
                this.biases[layer][neuron] += deltaW;
                //for the rest:
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    this.weights[layer][neuron][prevNeuron] = (1 - learning_rate * lambda / m) * weights[layer][neuron][prevNeuron] + deltaW * outputs[layer - 1][prevNeuron];
                }
            }
        }
    }

    //Prediction (1 if >= .5)
    private double[] predict(double[] x) {
        double[] output = feedForward(x);
        double[] predicted = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < output.length; i++) {
            double threshold = 0.5;
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
            for (int i = 0; i < y.length; i++) {
                if (predict(x)[i] != y[i]) counter++;
                continue main;
            }
        }
        return (1 - counter / (double) instances.size()) * 100;
    }

}
