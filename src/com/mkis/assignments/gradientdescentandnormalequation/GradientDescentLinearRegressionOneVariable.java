package com.mkis.assignments.gradientdescentandnormalequation;

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
import java.util.Arrays;
import java.util.List;

//First weekly assignments part 1 Linear Regression with gradient descent (non vectorized solution)

public class GradientDescentLinearRegressionOneVariable extends ApplicationFrame {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\gradientdescentandnormalequation\\data1.txt";
    private static double m; // number of training examples
    private static int n; //number of features (including x0)
    private static int iterations = 0; // number of iterations for gradient descent
    private static double theta[]; // parameters/weights array
    private static List<Instance> dataSet = new ArrayList<>(); //list containing 1 row of training example

    public static void main(String[] args) {

        getNumberOfFeaturesAndTrainingExamples();
        loadData();
        initTheta();
        System.out.println("Cost function value at start: " + createCostFunction(dataSet));
        doGradientDescent(dataSet);

        NumberFormat nf = new DecimalFormat("##.##");
        System.out.println("The value (profit) prediction in $s for city (population) size: 35,000:");
        System.out.println(nf.format((theta[0] + theta[1] * 3.5) * 10000));

        GradientDescentLinearRegressionOneVariable visualizationOfData =
                new GradientDescentLinearRegressionOneVariable("Visualization of data");
        visualizationOfData.pack();
        RefineryUtilities.centerFrameOnScreen(visualizationOfData);
        visualizationOfData.setResizable(false);
        visualizationOfData.setVisible(true);
    }

    private GradientDescentLinearRegressionOneVariable(final String title) {

        super(title);

        XYSeriesCollection seriesCollection = new XYSeriesCollection();

        // Create the scatter data series
        XYSeries series = new XYSeries("Profits depending on city population");
        for (int i = 0; i < dataSet.size(); i++) {
            series.add(dataSet.get(i).xVariables[1], dataSet.get(i).yValue);
        }
        seriesCollection.addSeries(series);

        // Create the line data series
        XYSeries lineSeries = new XYSeries("Regression function");
        for (int i = 0; i < 30; i++) {
            lineSeries.add((double) i, theta[0] + theta[1] * i);
        }
        seriesCollection.addSeries(lineSeries);

        // Create the chart with the plot and a legend
        JFreeChart chart = ChartFactory.createXYLineChart("Data set - Linear Regression", "Population in 10.000s", "Profit levels in 10.000$s", seriesCollection, PlotOrientation.VERTICAL, true, true, false);

        XYPlot plot = (XYPlot)chart.getPlot();
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

    public static class Instance {
        double yValue;
        double[] xVariables;

        Instance(double yValue, double[] xVariables) {
            this.yValue = yValue;
            this.xVariables = xVariables;
        }
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
            String[] columns;
            while ((line = bufferedReader.readLine()) != null) {
                columns = line.split(",");
                double y = Double.parseDouble(columns[n - 1]);
                double xArray[] = new double[n];
                xArray[0] = 1.0;
                for (int i = 0; i < n - 1; i++) {
                    xArray[i + 1] = Double.parseDouble(columns[i]);
                }
                Instance instance = new Instance(y, xArray);
                dataSet.add(instance);
            }
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //initiate theta
    private static void initTheta() {
        theta = new double[n];
        for (int i = 0; i < n; i++) {
            theta[i] = 0.0;
        }
    }

    private static double createHypothesis(double[] x) {
        double hypothesis = 0.0;
        for (int i = 0; i < n; i++) {
            hypothesis += theta[i] * x[i];
        }
        return hypothesis;
    }

    //Create Cost function
    private static double createCostFunction(List<Instance> instances) {
        double sumOfsqrErrors = 0.0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double hypothesis = createHypothesis(x);
            double y = instance.yValue;
            sumOfsqrErrors += Math.pow((hypothesis - y), 2);
        }
        double J = (1 / (2 * m)) * sumOfsqrErrors;
        //System.out.println("Cost function value with theta(" + Arrays.toString(theta) + "): " + J);
        return J;
    }

    private static void doGradientDescent(List<Instance> instances) {
        for (int k = 0; k < 1500; k++) {
            double[] temp = new double[n];
            for (int i = 0; i < n; i++) {
                temp[i] = 0.0;
            }
            double costFunctionOld = createCostFunction(dataSet);
            for (Instance instance : instances) {
                double[] x = instance.xVariables;
                double hypothesis = createHypothesis(x);
                double y = instance.yValue;
                for (int i = 0; i < n; i++) {
                    temp[i] = temp[i] + (hypothesis-y)*x[i];
                }
            }
            double alpha = 0.01;
            for (int i = 0; i < n; i++) {
                theta[i] = theta[i] - (alpha / m ) * temp[i];
            }
            iterations++;
            //Cost function to descend, theta after each iteration:
            //System.out.println("Iteration: " + iterations + " " + Arrays.toString(theta));
            //System.out.println(createCostFunction(dataSet));
            if (costFunctionOld - createCostFunction(dataSet) < 0.000001) {
                System.out.println("Number of iterations: " + iterations);
                return;
            }
        }
        System.out.println("Number of iterations: " + iterations);
        System.out.println("Cost function value at the end: " + createCostFunction(dataSet));
    }

}