package com.mkis.assignments.neuralnetwork;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Load in csv data, first-last column are the input variables/features/attributes, last column is the class value(s)
 */

public class LoadData {

    private List<Instance> dataSet = new ArrayList<>(); //list containing 1 row of training example
    private double m; // number of training examples
    private int n; // number of columns in dataset with last being the class value
    private List<Double> meansOfFeatures = new ArrayList<>(); //list containing the means of each feature (m training examples)
    private List<Double> maxMinusMinOfFeatures = new ArrayList<>(); //list containing max-min value of each feature (m training examples)

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
        m = 0;
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
        //for mean normalization get means and (max-min)s of all all features
        for (int i = 0; i < n - 1; i++) {
            meansOfFeatures.add(i, tempList.get(i).stream().mapToDouble(val -> val).average().getAsDouble());
            maxMinusMinOfFeatures.add(i, Collections.max(tempList.get(i)) - Collections.min(tempList.get(i)));
        }
        bufferedReader.close();
        reader.close();
    }

    //Load the data from the file, outputs the dataSet (list of Instances)
    public List<Instance> loadData(String file, boolean featureNormalize) throws java.io.IOException {
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
                    // feature scaling to get variable values between 0 and 1
                    if (meansOfFeatures.get(i) == 0 || maxMinusMinOfFeatures.get(i) == 0) {
                        xArray[i] = Double.parseDouble(columns[i]);
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
        return dataSet;
    }

    //Feature normalizes the test variables
    public double[] createFeatureNormalizedTestValue(double[] testVariables) {
        double[] featureNormalizedTestValue = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            if (meansOfFeatures.get(i) == 0 || maxMinusMinOfFeatures.get(i) == 0) {
                featureNormalizedTestValue[i] = testVariables[i];
            } else {
                featureNormalizedTestValue[i] = (testVariables[i] - meansOfFeatures.get(i)) / maxMinusMinOfFeatures.get(i);
            }
        }
        return featureNormalizedTestValue;
    }

}
