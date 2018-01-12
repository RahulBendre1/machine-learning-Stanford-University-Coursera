package com.mkis.assignments.neuralnetwork;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Load in csv data, first-last column are the input variables/features/attributes, last column is/are the class value/s
 */

public class LoadData {

    private List<Instance> dataSet = new ArrayList<>(); //list containing all the data
    private List<Instance> trainingSet = new ArrayList<>(); //list containing the training examples
    private List<Instance> crossValidationSet = new ArrayList<>(); //list containing the cross validation examples (dataset-trainingset)/2
    private List<Instance> testSet = new ArrayList<>(); //list containing the test examples (dataset-trainingset)/2
    private int n; // number of columns in the dataset, last being the class value
    private List<Double> meansOfFeatures = new ArrayList<>(); //list containing the means of each feature
    private List<Double> maxMinusMinOfFeatures = new ArrayList<>(); //list containing max-min value of each feature

    public static class Instance {
        double[] classValues;
        double[] inputVariables;

        Instance(double[] yValues, double[] xVariables) {
            this.classValues = yValues;
            this.inputVariables = xVariables;
        }
    }

    //Get the number or tr. examples, the class values, and prepare the input variables for mean normalization
    private void getInfoOnData(String file) throws java.io.IOException {
        FileReader reader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(reader);
        String line;
        String[] columns;
        List<List<Double>> tempList = new ArrayList<>();
        double m = 0;
        while ((line = bufferedReader.readLine()) != null) {
            columns = line.split(",");
            n = columns.length;
            if (m == 0) {
                for (int i = 0; i < n - 1; i++) {
                    List<Double> tempInnerList = new ArrayList<>();  //create inner lists for the features, only once
                    tempList.add(i, tempInnerList);
                }
            }
            for (int i = 0; i < n - 1; i++) {
                tempList.get(i).add(Double.parseDouble(columns[i]));
            }
            m++;
        }
        //System.out.println("The numer of rows in the dataset: " + m);
        //for mean normalization get means and (max-min)s of all all features
        for (int i = 0; i < n - 1; i++) {
            meansOfFeatures.add(i, tempList.get(i).stream().mapToDouble(val -> val).average().getAsDouble());
            maxMinusMinOfFeatures.add(i, Collections.max(tempList.get(i)) - Collections.min(tempList.get(i)));
        }
        bufferedReader.close();
        reader.close();
    }

    //Load the data from the file, outputs the shuffled dataSet (list of Instances)
    //trainingSetRatio is double value between 0-100
    public void loadData(String file, boolean featureNormalize, double trainingSetRatio) throws java.io.IOException {
        getInfoOnData(file);
        FileReader reader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(reader);
        String line;
        String[] columns;
        double[] classValue = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        while ((line = bufferedReader.readLine()) != null) {
            columns = line.split(",");
            double y = Double.parseDouble(columns[n - 1]);
            if (y == 10) {
                classValue = new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            } else if (y == 1) {
                classValue = new double[]{0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
            } else if (y == 2) {
                classValue = new double[]{0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
            } else if (y == 3) {
                classValue = new double[]{0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
            } else if (y == 4) {
                classValue = new double[]{0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
            } else if (y == 5) {
                classValue = new double[]{0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
            } else if (y == 6) {
                classValue = new double[]{0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
            } else if (y == 7) {
                classValue = new double[]{0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
            } else if (y == 8) {
                classValue = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
            } else if (y == 9) {
                classValue = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
            }
            double xArray[] = new double[n - 1];
            for (int i = 0; i < n - 1; i++) {
                if (featureNormalize) {
                    //Feature scaling to get variable values between 0 and 1:
                    //In case all values of this feature are zero, to avoid division with zero
                    if (meansOfFeatures.get(i) == 0 || maxMinusMinOfFeatures.get(i) == 0) {
                        xArray[i] = 0;
                    }
                    //In case all values of this features are the same, but non-zero (typical problem in case of picture images)
                    if (meansOfFeatures.get(i) == Double.parseDouble(columns[i]) && maxMinusMinOfFeatures.get(i) == 0) {
                        xArray[i] = 1;
                    } else {
                        xArray[i] = (Double.parseDouble(columns[i]) - meansOfFeatures.get(i)) / maxMinusMinOfFeatures.get(i);
                    }
                } else {
                    xArray[i] = Double.parseDouble(columns[i]);
                }
            }
            /*//Feature Scaled Input variables:
            for (int i = 0; i < n - 1; i++) {
                System.out.print(xArray[i] + " | ");
            }
            System.out.println();*/
            Instance instance = new Instance(classValue, xArray);
            dataSet.add(instance);
        }
        bufferedReader.close();
        reader.close();
        Collections.shuffle(dataSet, new Random());
        groupRandomizedDataSet(dataSet, trainingSetRatio/100);
    }

    //Create training, cross-validation and test set
    private void groupRandomizedDataSet(List<Instance> data, double ratio) {
        int sizeTraining = (int)((double)data.size() * ratio);
        int sizeCrossValidation = (data.size() - sizeTraining) / 2;
        for (int i = 0; i < data.size(); i++) {
            if (i < sizeTraining) trainingSet.add(data.get(i));
            if (sizeTraining < i && i < sizeTraining + sizeCrossValidation) crossValidationSet.add(data.get(i));
            if (sizeTraining + sizeCrossValidation < i) testSet.add(data.get(i));
        }
    }

    //Getters the different type of sets:
    public List<Instance> getTrainingSet(){
        return trainingSet;
    }
    public List<Instance> getCrossValidationSet(){
        return crossValidationSet;
    }
    public List<Instance> getTestSet(){
        return testSet;
    }

    //Feature normalizes the test variables
    public double[] createFeatureNormalizedTestValue(double[] testVariables) {
        double[] featureNormalizedTestValue = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            //In case all values of this feature are zeros, to avoid division with zero
            if (meansOfFeatures.get(i) == 0 || maxMinusMinOfFeatures.get(i) == 0) {
                featureNormalizedTestValue[i] = 0;
            }
            //In case all values of this features are the same, but non-zero (typical problem in case of picture images)
            if (meansOfFeatures.get(i) == testVariables[i] && maxMinusMinOfFeatures.get(i) == 0) {
                featureNormalizedTestValue[i] = 1; //the double number is the maximum value of this feature
            } else {
                featureNormalizedTestValue[i] = (testVariables[i] - meansOfFeatures.get(i)) / maxMinusMinOfFeatures.get(i);
            }
        }
        return featureNormalizedTestValue;
    }

}
