package com.mkis.assignments.gradientdescentandnormalequation;

import org.apache.commons.math3.linear.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.DoubleStream;

//Linear Regression(multiple variables(2)) with Gradient Descent and Normal Equations (Vectorized)

public class GradientDescentAndNormalEquationMultipleVariables {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\gradientdescentandnormalequation\\data2.txt";
    private static double[][] data; // dataset array
    private static double[][] variablesWithOneArray;  //variable array with ones added to the first row (X matrix)
    private static int m; // number of training examples
    private static int n; // number of features (including x0)

    private static RealMatrix X; // matrix X
    private static RealVector y; // vector y
    private static RealVector theta; //vector theta (parameters)

    //Feature Normalization for Variable X1s:
    private static List<Double> listVar1 = new ArrayList<>();
    private static double muVar1;
    private static double sigmaVar1;
    //Feature Normalization for Variable X2s:
    private static List<Double> listVar2 = new ArrayList<>();
    private static double muVar2;
    private static double sigmaVar2;

    public static void main(String[] args) {
        getNumberOfFeaturesAndTrainingExamples();
        loadData();
        createMatrixX();
        createFeatureNormalizedX();
        createVectorY();
        initTheta();
        createCostFunction();
        doGradientDescent();
        //doNormalEquations();

        NumberFormat nf = new DecimalFormat("##.##");

        System.out.println("Predicted price of a 1650 sq-ft, 3 br house in $s:");
        System.out.println(nf.format(theta.getEntry(0) + theta.getEntry(1) * ((1650 - muVar1) / sigmaVar1)
                + theta.getEntry(2) * ((3 - muVar2) / sigmaVar2))); // Gradient descent
        //System.out.println(theta.getEntry(0) + theta.getEntry(1) * ((1650 - muVar1) / sigmaVar1) + theta.getEntry(2) * ((3 - muVar2) / sigmaVar2)); // Normal Equation
    }

    private static void getNumberOfFeaturesAndTrainingExamples() {
        try {
            FileReader reader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line;
            String[] columns;
            m = 0;
            while ((line = bufferedReader.readLine()) != null) {
                columns = line.split(",");
                n = columns.length;
                m++;
            }
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //Load the data from the txt file into an array
    private static void loadData() {
        try {
            FileReader reader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line;
            data = new double[m][n];
            String[] columns;
            int i = 0;
            while ((line = bufferedReader.readLine()) != null) {
                columns = line.split(",");
                for (int j = 0; j < (n - 1); j++) {
                    data[i][j] = Double.parseDouble(columns[j]);
                }
                listVar1.add(data[i][0]);
                listVar2.add(data[i][1]);
                data[i][n - 1] = Double.parseDouble(columns[n - 1]);
                //System.out.println(data[i][0] + "|" + data[i][1] + "|" + data[i][2]);
                i++;
            }
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //Create matrix X (variables with ones added to the first column)
    private static void createMatrixX() {
        variablesWithOneArray = new double[m][n];
        for (int i = 1; i < m; i++) {
            variablesWithOneArray[i][0] = 1;
            for (int j = 1; j < n; j++) {
                variablesWithOneArray[i][j] = data[i][j - 1];
            }
        }
        X = new Array2DRowRealMatrix(variablesWithOneArray);
    }

    //Feature normalize x values for gradient descent to get smoother descent
    private static void createFeatureNormalizedX() {
        muVar1 = listVar1.stream().mapToDouble(val -> val).average().getAsDouble();
        muVar2 = listVar2.stream().mapToDouble(val -> val).average().getAsDouble();
        sigmaVar1 = Collections.max(listVar1) - Collections.min(listVar1);
        sigmaVar2 = Collections.max(listVar2) - Collections.min(listVar2);
        for (int i = 0; i < m; i++) {
            variablesWithOneArray[i][1] = (variablesWithOneArray[i][1] - muVar1) / sigmaVar1;
            variablesWithOneArray[i][2] = (variablesWithOneArray[i][2] - muVar2) / sigmaVar2;
        }
        X = new Array2DRowRealMatrix(variablesWithOneArray);
    }

    //Create vector y (values)
    private static void createVectorY() {
        double[] valuesArray = new double[m];
        for (int i = 0; i < m; i++) {
            valuesArray[i] = data[i][n - 1];
        }
        y = new ArrayRealVector(valuesArray);
    }

    //Initiate theta with [0;0;0]
    private static void initTheta() {
        double[] thetaArray = new double[n];
        for (int i = 0; i < n; i++) {
            thetaArray[i] = 0.0;
        }
        theta = new ArrayRealVector(thetaArray);
    }

    private static void createCostFunction() {
        RealVector h = X.operate(theta);  // h = X*theta
        RealVector sqrErrors = h.subtract(y).ebeMultiply(h.subtract(y)); //qrErrors = (h-y).^2;
        double[] temp = sqrErrors.toArray();
        double sumOfsqrErrors = DoubleStream.of(temp).sum();
        double J = (1 / (2 * (double) m)) * sumOfsqrErrors;
        System.out.println("Cost function value with theta [" + theta.getEntry(0) + ", " + theta.getEntry(1) + ", " + theta.getEntry(2) + "] : " + J);
    }

    private static void doGradientDescent() {
        for (int k = 0; k < 1000; k++) {
            RealVector delta = ((X.transpose().multiply(X).operate(theta)).subtract(X.transpose().operate(y))).mapMultiply(1 / (double) m); // delta=1/m*(X'*X*theta-X'*y)
            double alpha = 0.1;  //alpha value to play with
            theta = theta.subtract(delta.mapMultiply(alpha)); // theta=theta-alpha.*delta
        }
        System.out.println("theta after 400 iterations: " + theta);
        System.out.println("h = " + theta.getEntry(0) + " + " + theta.getEntry(1) + "x1 + " + theta.getEntry(2) + "x2");
        createCostFunction();
    }

    //Normal equation:
    private static void doNormalEquations() {
        RealMatrix inverse = new LUDecomposition(X.transpose().multiply(X)).getSolver().getInverse();
        theta = inverse.multiply(X.transpose()).operate(y); // theta = pinv(X'*X)*X'*y;
        System.out.println("theta with normal equations: " + theta);
    }

}
