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

//Linear Regression(multiple variables(2)) with Gradient Descent and Normal Equations

public class GradientDescentAndNormalEquationMultipleVariables {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\gradientdescentandnormalequation\\data2.txt";
    private static double[][] data; // data array
    private static double[][] variablesWithOneArray;  //variable array with ones added to the first row (X matrix)
    private static double[] valuesArray;  // values array (y vector)
    private static int dataRow = 47; //number of rows in data set (m)
    private static int dataCols = 3; // number of columns in data set (n)
    private static double m; // number of training examples
    private static int n; // number of features

    private static double alpha = 0.1; // learning rate
    private static int iterationsVar = 1;
    private static int iterations = 400; // number of iterations for gradient descent

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
        loadData();
        createMatrixX();
        createFeatureNormalizedX();
        createVectorY();
        createTheta();
        createCostFunction();
        doGradientDescent();
        //doNormalEquations();

        NumberFormat nf = new DecimalFormat("##.##");

        System.out.println("Predicted price of a 1650 sq-ft, 3 br house in $s:");
        System.out.println(nf.format(theta.getEntry(0) + theta.getEntry(1)*((1650-muVar1)/sigmaVar1)
                + theta.getEntry(2)*((3-muVar2)/sigmaVar2))); // Gradient descent
        //System.out.println(theta.getEntry(0) + theta.getEntry(1)*1650 + theta.getEntry(2)*3); // Normal Equation
    }

    //load the data from the txt file into an array
    private static void loadData() {
        try {
            FileReader reader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line;
            data = new double[dataRow][dataCols];
            String[] lines;
            int i = 0;
            List<String> allLinesList = new ArrayList<>(); //To get the number of rows
            while ((line = bufferedReader.readLine()) != null) {
                allLinesList.add(line); //To get the number of rows
                lines = line.split(",");
                n = lines.length;
                data[i][0] = Double.parseDouble(lines[0]);
                listVar1.add(data[i][0]);
                data[i][1] = Double.parseDouble(lines[1]);
                listVar2.add(data[i][1]);
                data[i][2] = Double.parseDouble(lines[2]);
                i++;
            }
            m = allLinesList.size();
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //create matrix X (variables with ones added to the first column)
    private static void createMatrixX() {
        variablesWithOneArray = new double[47][3];
        for (int i = 0; i < dataRow; i++) {
            variablesWithOneArray[i][0] = 1;
            variablesWithOneArray[i][1] = data[i][0];
            variablesWithOneArray[i][2] = data[i][1];
        }
        X = new Array2DRowRealMatrix(variablesWithOneArray);
    }

    //feature normalize x values for gradient descent to get smoother descent
    private static void createFeatureNormalizedX() {
        muVar1 = listVar1.stream().mapToDouble(val -> val).average().getAsDouble();
        muVar2 = listVar2.stream().mapToDouble(val -> val).average().getAsDouble();
        sigmaVar1 = Collections.max(listVar1) - Collections.min(listVar1);
        sigmaVar2 = Collections.max(listVar2) - Collections.min(listVar2);
        for (int i = 0; i < dataRow; i++) {
            variablesWithOneArray[i][1] = (variablesWithOneArray[i][1]-muVar1)/sigmaVar1;
            variablesWithOneArray[i][2] = (variablesWithOneArray[i][2]-muVar2)/sigmaVar2;
        }
        X = new Array2DRowRealMatrix(variablesWithOneArray);
    }

    //create vector y (values)
    private static void createVectorY() {
        valuesArray = new double[47];
        for (int i = 0; i < dataRow; i++) {
            valuesArray[i] = data[i][2];
        }
        y = new ArrayRealVector(valuesArray);


    }

    //initiate theta with [0;0;0]
    private static void createTheta () {
        theta = new ArrayRealVector(new double[] { 0, 0, 0});
    }

    private static void createCostFunction () {
        RealVector h = X.operate(theta);  // h = X*theta
        RealVector sqrErrors = h.subtract(y).ebeMultiply(h.subtract(y)); //qrErrors = (h-y).^2;
        double[] temp = sqrErrors.toArray();
        double sumOfsqrErrors = DoubleStream.of(temp).sum();
        double J = (1/(2*m)) * sumOfsqrErrors;
        System.out.println("Cost function value with theta [0;0;0] : " + J);
    }

    private static boolean doGradientDescent () {
        RealVector delta = ((X.transpose().multiply(X).operate(theta)).subtract(X.transpose().operate(y))).mapMultiply(1/m); // delta=1/m*(X'*X*theta-X'*y)
        theta = theta.subtract(delta.mapMultiply(alpha)); // theta=theta-alpha.*delta
        if(iterationsVar==iterations) {
            System.out.println("theta after 400 iterations: " + theta);
            System.out.println("h = " + theta.getEntry(0) + " + " + theta.getEntry(1) + "x1 + " + theta.getEntry(2) + "x2");
            return true;
        }
        iterationsVar++;
        return doGradientDescent();
    }

    private static void doNormalEquations () {
        RealMatrix inverse = new LUDecomposition(X.transpose().multiply(X)).getSolver().getInverse();
        theta = inverse.multiply(X.transpose()).operate(y); // theta = pinv(X'*X)*X'*y;
        System.out.println("theta with normal equations: " + theta);
    }

}
