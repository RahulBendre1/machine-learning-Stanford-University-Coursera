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
 * <p>
 * Mini-batch training option added.
 * Learning rate adjustment during training added (useful when using stochastic gradient descent i.e. batch_size = 1).
 */

public class ConvolutionalNeuralNetwork {

    private double[][] outputs; //output of every neuron, indexes: layer, neuron
    private double[][][] weights; //indexes: layer, neuron, neuron in the previous layer that is connected with
    private double[][][] DELTAweights; //Tri-delta weights (to compute partial derivatives of the weights)
    private double[][] biases; //bias weights
    private double[][] DELTAbiases; //Tri-delta biases (to compute partial derivatives of  the biases)
    private double[][] errors; //error of every neuron, indexes: layer, neuron
    private double[][] derivatives; //derivatives of every neuron, indexes: layer, neuron

    private double m; //number of training examples

    private static List<Double> trainingSetCostFunction = new ArrayList<>();
    private static List<Double> CVSetSetCostFunction = new ArrayList<>();

    private final int[] NETWORK_LAYER_SIZES; //neuron/nodes in each layer
    private final int NETWORK_SIZE; //amount of layers in the network

    private static NumberFormat nf = new DecimalFormat("##.##");

    public static void main(String[] args) throws java.io.IOException {

        String file = "D" +
                ":\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\neuralnetwork\\data1.txt";
        ConvolutionalNeuralNetwork test = new ConvolutionalNeuralNetwork(400, 200, 10);
        LoadData loadData = new LoadData();
        loadData.loadData(file, true, 80);
        List<LoadData.Instance> trSet = loadData.getTrainingSet();
        List<LoadData.Instance> cvSet = loadData.getCrossValidationSet();
        List<LoadData.Instance> testSet = loadData.getTestSet();

        double[] lambda = new double[]{0, 0.001, 0.003, 0.005};

        //So the initial weights and biases are the same for all lambdas
        double initialBiases[][] = new double[test.NETWORK_SIZE][];
        double initialWeights[][][] = new double[test.NETWORK_SIZE][][];
        for (int k = 0; k < test.NETWORK_SIZE; k++) {
            initialBiases[k] = test.biases[k];
            if (k > 0) {
                initialWeights[k] = test.weights[k];
            }
        }

        for (int i = 0; i < lambda.length; i++) {
            System.out.println("\n--------------------------------------------------------------");
            System.out.println("\nTraining for lambda (" + lambda[i] + ")...");
            for (int j = 0; j < test.NETWORK_SIZE; j++) {
                test.outputs[j] = test.setValuesToZero(test.NETWORK_LAYER_SIZES[j]);
                test.derivatives[j] = test.setValuesToZero(test.NETWORK_LAYER_SIZES[j]);
                test.errors[j] = test.setValuesToZero(test.NETWORK_LAYER_SIZES[j]);
                if (j > 0) {
                    test.biases[j] = initialBiases[j];
                    test.DELTAbiases[j] = test.setValuesToZero(test.NETWORK_LAYER_SIZES[j]);
                    test.weights[j] = initialWeights[j];
                    test.DELTAweights[j] = test.setValuesToZero(test.NETWORK_LAYER_SIZES[j], test.NETWORK_LAYER_SIZES[j - 1]);
                }
            }
            //Set batch size to 1 to get stochastic gradient descent (SGD), set it equal to training set size to get batch gradient descent
            test.train(trSet, cvSet, 10, 10, 1, lambda[i]);
            //Prediction with no regularization
            if (i == 0) {
                System.out.println("\nAccuracy tested on the cross-validation set: " + test.calcAccuracyOfModel(cvSet) + " %");

                double[] testPicture = testSet.get(0).inputVariables;

                for (int classV = 0; classV < 10; classV++) {
                    System.out.println("The probability that the number is a " + classV + ": " + nf.format(100 * test.feedForward(testPicture)[classV]) + " %");
                }

                int index = -1;
                for (int z = 0; (z < testSet.get(0).classValues.length) && (index == -1); z++) {
                    if (testSet.get(0).classValues[z] == 1d) {
                        index = z;
                    }
                }
                System.out.println("The number should be a: " + index + "\n");
            }
        }

        //Plotting/evaluation
        Evaluation eval = new Evaluation("Evaluation", trainingSetCostFunction, CVSetSetCostFunction, lambda);
        eval.pack();
        RefineryUtilities.centerFrameOnScreen(eval);
        eval.setResizable(true);
        eval.setVisible(true);
    }

    //Architecture init of the network:
    private ConvolutionalNeuralNetwork(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;

        this.weights = new double[NETWORK_SIZE][][];
        this.DELTAweights = new double[NETWORK_SIZE][][];
        this.biases = new double[NETWORK_SIZE][];
        this.DELTAbiases = new double[NETWORK_SIZE][];
        this.outputs = new double[NETWORK_SIZE][];
        this.errors = new double[NETWORK_SIZE][];
        this.derivatives = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.outputs[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.errors[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.derivatives[i] = new double[NETWORK_LAYER_SIZES[i]];

            if (i > 0) {
                this.biases[i] = initWeights(NETWORK_LAYER_SIZES[i]);
                this.DELTAbiases[i] = setValuesToZero(NETWORK_LAYER_SIZES[i]);
                this.weights[i] = initWeights(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1]);
                this.DELTAweights[i] = setValuesToZero(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1]);
            }
        }
    }

    //Initialize biases/weights:
    private double[] initWeights(int size) {
        double[] arr = new double[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            arr[i] = (double) random.nextInt(100) / 100;
        }
        return arr;
    }

    //Set values to zero:
    private double[] setValuesToZero(int size) {
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) {
            arr[i] = 0;
        }
        return arr;
    }

    //Initialize weights:
    private double[][] initWeights(int size, int sizePrev) {
        double[][] arr = new double[size][sizePrev];
        for (int i = 0; i < size; i++) {
            arr[i] = initWeights(sizePrev);
        }
        return arr;
    }

    //Set values to zero:
    private double[][] setValuesToZero(int size, int sizePrev) {
        double[][] arr = new double[size][sizePrev];
        for (int i = 0; i < size; i++) {
            arr[i] = setValuesToZero(sizePrev);
        }
        return arr;
    }

    //Train network using batches (1 to training set size)
    private void train(List<LoadData.Instance> trainingSet, List<LoadData.Instance> CVSet, int iterations, int batch_size, double learning_rate, double lambda) {
        this.m = (double) trainingSet.size();
        if (batch_size > m) batch_size = (int) m;
        int numberOfBatches = (int) (m / batch_size);
        for (int iteration = 0; iteration < iterations; iteration++) {
            /*System.out.println("iteration: " + iteration + ", iterations: " + iterations);
            learning_rate = (double)iterations / ((double)iteration + (double)iterations);
            System.out.println("alpha: "+learning_rate);
            if(iteration % 10 == 0){
                learning_rate = 0.75 * learning_rate;
            }*/
            int counter = 0;
            for (int batch = 0; batch < numberOfBatches; batch++) {
                for (int i = 0; i < NETWORK_SIZE; i++) {
                    this.DELTAbiases[i] = setValuesToZero(NETWORK_LAYER_SIZES[i]);
                    if (i > 0) {
                        this.DELTAweights[i] = setValuesToZero(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1]);
                    }
                }
                for (int b = counter; b < counter + batch_size; b++) {
                    double[] x = trainingSet.get(b).inputVariables;
                    double[] y = trainingSet.get(b).classValues;
                    this.prepare(x, y);
                }
                updateWeights(learning_rate, batch_size, lambda);
                counter += batch_size;
            }
            if (iteration == iterations - 1) {
                trainingSetCostFunction.add(createRegularizedCostFunction(trainingSet, lambda));
                CVSetSetCostFunction.add(createRegularizedCostFunction(CVSet, 0));
            }
            System.out.println("Cost function at iteration " + (iteration + 1) + " (Number of batches: " + numberOfBatches + ", Batch size: " + batch_size + "): " + createRegularizedCostFunction(trainingSet, lambda));
        }
    }

    //Feed forward, back prop and calculate TriDelta terms
    private void prepare(double[] input, double[] target) {
        feedForward(input);
        backPropError(target);
        updateDeltas();
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

    //Back propagation starting from the output layer's target(s)
    private void backPropError(double[] target) {
        //Error's of output neurons:
        for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
            //this.errors[NETWORK_SIZE - 1][neuron] = (outputs[NETWORK_SIZE - 1][neuron] - target[neuron]) * derivatives[NETWORK_SIZE - 1][neuron]; WRONG!
            this.errors[NETWORK_SIZE - 1][neuron] = outputs[NETWORK_SIZE - 1][neuron] - target[neuron];
            //System.out.println("target of neuron["+neuron+"]: " + target[neuron]);
            //System.out.println("output error of last layer, neuron["+neuron+"]: " + this.errors[NETWORK_SIZE - 1][neuron]);
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

    //First hidden layer to the output layer, 1 iteration, updating the Deltas of biases and the weights
    private void updateDeltas() {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                this.DELTAbiases[layer][neuron] += errors[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    this.DELTAweights[layer][neuron][prevNeuron] += outputs[layer - 1][prevNeuron] * errors[layer][neuron];
                }
            }
        }
    }

    //First hidden layer to the output layer, 1 iteration, updating the biases and the weights using the partial derivatives
    private void updateWeights(double learning_rate, double batch_size, double lambda) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                //for the biases:
                this.biases[layer][neuron] -= learning_rate * DELTAbiases[layer][neuron] / batch_size;
                //for the rest:
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    this.weights[layer][neuron][prevNeuron] -= learning_rate * (DELTAweights[layer][neuron][prevNeuron] / batch_size + lambda * weights[layer][neuron][prevNeuron]); //reg. term included
                }
            }
        }
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
