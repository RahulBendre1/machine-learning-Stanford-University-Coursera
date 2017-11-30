package com.mkis.assignments.gradientdescentandnormalequation;

import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;
import org.apache.commons.math3.linear.*;
import org.jfree.data.xy.DefaultXYDataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;

//First weekly assignments part 1

public class GradientDescentLinearRegressionOneVariable extends Application{

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\gradientdescentandnormalequation\\data1.txt";
    private static double[][] data; // data array
    private static double[][] variablesWithOneArray;  //variable array with ones added to the first row (X matrix)
    private static double[] valuesArray;  // values array (y vector)
    private static int dataRow = 97; //number of rows in data set (m)
    private static int dataCols = 2; // number of columns in data set (n)
    private static double m; // number of training examples
    private static int n; // number of features

    private static double alpha = 0.01; // learning rate
    private static int iterationsVar = 1;
    private static int iterations = 1500; // number of iterations for gradient descent

    private static RealMatrix X; // matrix X (Population of City in 10,000s)
    private static RealVector y; // vector y (Profit in $10,000s)
    private static RealVector theta; //vector theta (parameters)

    private static List<Double> xAxisValues = new ArrayList<>();
    private static List<Double> yAxisValues = new ArrayList<>();
    private static ObservableList<Double> xValuesObsList = FXCollections.observableArrayList();
    private static ObservableList<Double> yValuesObsList = FXCollections.observableArrayList();

    public void start(Stage main) throws Exception{

        main.setTitle("Visualization of data");
        main.setResizable(false);

        NumberAxis xAxis = new NumberAxis(4, 24, 1);
        NumberAxis yAxis = new NumberAxis(-5, 25, 1);
        xAxis.setLabel("Population of City in 10,000s");
        yAxis.setLabel("Profit in $10,000s");

        ScatterChart dataChart = new ScatterChart(xAxis, yAxis, getChartData());

        StackPane layout = new StackPane();
        layout.getChildren().addAll(dataChart);
        Scene scene = new Scene(layout, 400, 400);
        main.setScene(scene);
        main.show();
    }

    public static void main(String[] args) {
        loadData();
        createMatrixX();
        createVectorY();
        createTheta();
        createCostFunction();
        doGradientDescent();

        NumberFormat nf = new DecimalFormat("##.##");
        System.out.println("The value (profit) prediction in $s for city (population) size: 35,000:");
        System.out.println(nf.format((theta.getEntry(0) + theta.getEntry(1)*3.5)*10000));

        launch();
    }

    private ObservableList<XYChart.Series<Double, Double>> getChartData() {
        ObservableList<XYChart.Series<Double, Double>> data = FXCollections.observableArrayList();
        XYChart.Series <Double, Double > dataSeries = new XYChart.Series<>();
        for (int i =0 ; i < xAxisValues.size(); i++) {
            dataSeries.getData().add(new XYChart.Data<>(xAxisValues.get(i), yAxisValues.get(i)));
        }
        data.addAll(dataSeries);
        return data;
    }

    private ObservableList<XYChart.Series<Double, Double>> getLineData() {
        ObservableList<XYChart.Series<Double, Double>> data = FXCollections.observableArrayList();
        XYChart.Series <Double, Double > lineSeries = new XYChart.Series<>();

        for (int i =0 ; i < xAxisValues.size(); i++) {
            lineSeries.getData().add(new XYChart.Data<>((double)i, -3.6302914394044015 + 1.1663623503355864*i));
        }

        data.addAll(lineSeries);
        return data;
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
                data[i][1] = Double.parseDouble(lines[1]);
                xAxisValues.add(Double.parseDouble(lines[0]));
                yAxisValues.add(Double.parseDouble(lines[1]));

                i++;
            }
            xValuesObsList.addAll(xAxisValues);
            yValuesObsList.addAll(yAxisValues);
            m = allLinesList.size();
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //create matrix X (variables with ones added to the first column)
    private static void createMatrixX() {
        variablesWithOneArray = new double[97][2];
        for (int i = 0; i < dataRow; i++) {
            variablesWithOneArray[i][0] = 1;
            variablesWithOneArray[i][1] = data[i][0];
        }
        X = new Array2DRowRealMatrix(variablesWithOneArray);
    }

    //create vector y (values)
    private static void createVectorY() {
        valuesArray = new double[97];
        for (int i = 0; i < dataRow; i++) {
            valuesArray[i] = data[i][1];
        }
        y = new ArrayRealVector(valuesArray);
    }

    //initiate theta with [0;0]
    private static void createTheta () {
        theta = new ArrayRealVector(new double[] { 0, 0});
    }

    private static void createCostFunction () {
        RealVector h = X.operate(theta);  // h = X*theta
        RealVector sqrErrors = h.subtract(y).ebeMultiply(h.subtract(y)); //qrErrors = (h-y).^2;
        double[] temp = sqrErrors.toArray();
        double sumOfsqrErrors = DoubleStream.of(temp).sum();
        double J = (1/(2*m)) * sumOfsqrErrors;
        System.out.println("Cost function value with theta [0;0] : " + J);
    }

    private static boolean doGradientDescent () {
        RealVector delta = ((X.transpose().multiply(X).operate(theta)).subtract(X.transpose().operate(y))).mapMultiply(1/m); // delta=1/m*(X'*X*theta-X'*y)
        theta = theta.subtract(delta.mapMultiply(alpha)); // theta=theta-alpha.*delta
        if(iterationsVar==iterations) {
            System.out.println("theta after 1500 iterations: " + theta);
            System.out.println("h = " + theta.getEntry(0) + " + " + theta.getEntry(1) + "x");
            return true;
        }
        iterationsVar++;
        return doGradientDescent();
    }

}

