package com.mkis.assignments.gradientdescentandnormalequation;

import org.apache.commons.math3.linear.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;

//First weekly assignments part 1 Linear Regression with gradient descent vectorized solution

public class GradientDescentLinearRegressionOneVariableVectorized extends ApplicationFrame {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\gradientdescentandnormalequation\\data1.txt";
    private static double[][] data; // data array
    private static int m; // number of training examples
    private static int n; // number of features (including x0)

    private static RealMatrix X; // matrix X (Population of City in 10,000s)
    private static RealVector y; // vector y (Profit in $10,000s)
    private static RealVector theta; //vector theta (parameters)

    public static void main(String[] args) {

        getNumberOfFeaturesAndTrainingExamples();
        loadData();
        createMatrixX();
        createVectorY();
        initTheta();
        createCostFunction();
        doGradientDescent();

        NumberFormat nf = new DecimalFormat("##.##");
        System.out.println("The value (profit) prediction in $s for city (population) size: 35,000:");
        System.out.println(nf.format((theta.getEntry(0) + theta.getEntry(1) * 3.5) * 10000));

        GradientDescentLinearRegressionOneVariableVectorized visualizationOfData =
                new GradientDescentLinearRegressionOneVariableVectorized("Visualization of data");
        visualizationOfData.pack();
        RefineryUtilities.centerFrameOnScreen(visualizationOfData);
        visualizationOfData.setResizable(false);
        visualizationOfData.setVisible(true);
    }

    private GradientDescentLinearRegressionOneVariableVectorized(final String title) {

        super(title);

        XYSeriesCollection seriesCollection = new XYSeriesCollection();

        // Create the scatter data series
        XYSeries series = new XYSeries("Profits depending on city population");
        for (int i = 0; i < m; i++) {
            series.add(data[i][0], data[i][1]);
        }
        seriesCollection.addSeries(series);

        // Create the line data series
        XYSeries lineSeries = new XYSeries("Regression function");
        for (int i = 0; i < 30; i++) {
            lineSeries.add((double) i, theta.getEntry(0) + theta.getEntry(1) * i);
        }
        seriesCollection.addSeries(lineSeries);

        // Create the chart with the plot and a legend
        JFreeChart chart = ChartFactory.createXYLineChart("Data set - Linear Regression", "Population in 10.000s", "Profit levels in 10.000$s", seriesCollection, PlotOrientation.VERTICAL, true, true, false);

        XYPlot plot = (XYPlot) chart.getPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

        //Dataset
        renderer.setSeriesLinesVisible(0, false);
        renderer.setSeriesShapesVisible(0, true);
        //Line
        renderer.setSeriesLinesVisible(1, true);
        renderer.setSeriesShapesVisible(1, false);

        plot.setRenderer(renderer);
        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 800));
        setContentPane(chartPanel);
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

    //load the data from the txt file into an array
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
                data[i][n - 1] = Double.parseDouble(columns[n - 1]);
                //System.out.println(data[i][0] + "|" + data[i][1]);
                i++;
            }
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //create matrix X (variables with ones added to the first column)
    private static void createMatrixX() {
        double[][] variablesWithOneArray = new double[m][n];
        for (int i = 1; i < m; i++) {
            variablesWithOneArray[i][0] = 1;
            for (int j = 1; j < n; j++) {
                variablesWithOneArray[i][j] = data[i][j - 1];
            }
        }
        X = new Array2DRowRealMatrix(variablesWithOneArray);
    }

    //create vector y (values)
    private static void createVectorY() {
        double[] valuesArray = new double[m];
        for (int i = 0; i < m; i++) {
            valuesArray[i] = data[i][n - 1];
        }
        y = new ArrayRealVector(valuesArray);
    }

    //initiate theta
    private static void initTheta() {
        double[] thetaArray = new double[n];
       /* thetaArray[0] = -20;
        thetaArray[1] = 5;
        thetaArray[2] = 1;*/
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
        System.out.println("Cost function value with theta [" + theta.getEntry(0) + ", " + theta.getEntry(1) + "] : " + J);
    }

    private static void doGradientDescent() {
        for (int k = 0; k < 1500; k++) {
            // delta=1/m*(X'*X*theta-X'*y)
            RealVector delta = ((X.transpose().multiply(X).operate(theta)).subtract(X.transpose().operate(y))).mapMultiply(1 / (double) m);
            double alpha = 0.01;
            theta = theta.subtract(delta.mapMultiply(alpha)); // theta=theta-alpha.*delta
        }
        System.out.println("theta after 1500 iterations: " + theta);
        System.out.println("h = " + theta.getEntry(0) + " + " + theta.getEntry(1) + "x");
        createCostFunction();
    }

}

