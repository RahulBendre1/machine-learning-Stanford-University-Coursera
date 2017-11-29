package com.mkis.assignments.gradientdescentandnormalequation;

import org.apache.commons.math3.linear.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GradientDescentAndNormalEquationMultipleVariables {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\gradientdescentandnormalequation\\data2.txt";
    private static double[][] data;
    private static double[][] variablesWithOneArray;  //variable matrix with ones added to the first row (X matrix)
    private static double[] valuesArray;  // (y vector)
    private static int dataRow = 47;
    private static int dataCols = 3;
    private static int m; // number of training examples
    private static int n; // number of features
    //Feature Normalization for Variable X1s:
    private static List<Double> listVar1 = new ArrayList<>();
    private static double muVar1;
    private static double sigmaVar1;
    //Feature Normalization for Variable X2s:
    private static List<Double> listVar2 = new ArrayList<>();
    private static double muVar2;
    private static double sigmaVar2;

    private static double alpha = 0.1;
    private static int iterations = 400;

    private static RealMatrix X;
    private static RealVector y;
    private static RealVector theta;

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
            /*System.out.println("Data: ");
            for (int k = 0; k < dataRow; k++) {
                for (int j = 0; j < dataCols; j++) {
                    System.out.print(data[k][j] + "\t");
                }
                System.out.println();
            }*/
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    private static void createX() {
        variablesWithOneArray = new double[47][3];
        for (int i = 0; i < dataRow; i++) {
            variablesWithOneArray[i][0] = 1;
            variablesWithOneArray[i][1] = data[i][0];
            variablesWithOneArray[i][2] = data[i][1];
        }
    }

    private static void createFeatureNormalizedX() {
        muVar1 = listVar1.stream().mapToDouble(val -> val).average().getAsDouble();
        muVar2 = listVar2.stream().mapToDouble(val -> val).average().getAsDouble();
        sigmaVar1 = Collections.max(listVar1) - Collections.min(listVar1);
        sigmaVar2 = Collections.max(listVar2) - Collections.min(listVar2);
        for (int i = 0; i < dataRow; i++) {
            variablesWithOneArray[i][1] = (variablesWithOneArray[i][1]-muVar1)/sigmaVar1;
            variablesWithOneArray[i][2] = (variablesWithOneArray[i][2]-muVar2)/sigmaVar2;
        }
        X = MatrixUtils.createRealMatrix(variablesWithOneArray);
    }

    private static void createy() {
        valuesArray = new double[47];
        for (int i = 0; i < dataRow; i++) {
            valuesArray[i] = data[i][2];
        }
        y = new ArrayRealVector(valuesArray);


    }

    private static void createTheta () {
        theta = new ArrayRealVector(new double[] { 0, 0, 0});
    }

    private static void createCostFunction () {

        RealVector h = X.operate(theta);  // h = X*theta
        RealVector sqrErrors = h.subtract(y).ebeMultiply(h.subtract(y)); //qrErrors = (h-y).^2;
        /*double[] temp = new double[47] sqrErrors.toArray();

        Double J = 1/(2*m) * sqrErrors;

        *//*delta=1/m*(X'*X*theta-X'*y);
        theta=theta-alpha.*delta;*/
    }


    public static void main(String[] args) {

        /*// Invert p, using LU decomposition
        RealMatrix pInverse = new LUDecomposition(p).getSolver().getInverse();
        System.out.println(pInverse);*/

        loadData();
        createy();
        createX();
        createTheta();
        createFeatureNormalizedX();
        createCostFunction();
    }

}
