package com.mkis.assignments.logisticregression;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/*Predict whether microchips from a fabrication plant passes quality assurance (QA).
During QA, each microchip goes through various tests to ensure it is functioning correctly.
Suppose you are the product manager of the factory and you have the
test results for some microchips on two different tests. From these two tests,
you would like to determine whether the microchips should be accepted or
rejected. To help you make the decision, you have a data set of test results
on past microchips, from which you can build a logistic regression model.

With non-linear decision boundary using feature mapping.*/

public class RegularizedLogisticRegression extends ApplicationFrame {
    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\logisticregression\\data2.txt";
    private static double m; // number of training examples
    private static int n; // number of features

    private static double lambda = 1; // weight of the regularization
    private static int iterations = 0;// number of iterations needed for gradient descent

    private static double theta[]; // parameters/weights array
    private static List<Instance> dataSet = new ArrayList<>(); //list containing 1 row of training example
    private static List<Instance> dataSetFeatureMapping = new ArrayList<>(); //list containing 1 row of training example with feature mapping

    public static void main(String[] args) {

        NumberFormat nf = new DecimalFormat("##.##");

        loadData();
        doFeatureMapping(dataSet);
        initTheta();
        System.out.println("Cost Function at theta: " + Arrays.toString(theta) + " : " + createCostFunction(dataSetFeatureMapping));
        createGradients(dataSetFeatureMapping);
        doGradientDescent(dataSetFeatureMapping);
        System.out.println("Cost Function at theta: " + Arrays.toString(theta) + " : " + createCostFunction(dataSetFeatureMapping));
        System.out.println("\nAccuracy of the model: " + nf.format(calcAccuracyOfModel(dataSetFeatureMapping)) + " %");
        //doLBFGS(dataSet);

        //We predict 1 if  h(x) >= 0 !
        double x1 = 0.2;
        double x2 = 0.2;
        System.out.println("\nFor a microchip with test 1 scores 1: " + x1 + " and test 2 score: " + x2 + " , we predict an acceptance probability: ");
        //System.out.println(theta[0] + theta[1]*x1 + theta[2]*x2  + theta[3] * Math.pow(x1, 2) + theta[4] * Math.pow(x2, 2));
        System.out.println(nf.format(100 * (1- (createHypothesis(new double[]{1, x1, x2, Math.pow(x1, 2), Math.pow(x2, 2)})))) + " %"); // 1s inside circle so it is the other way around compared to the previous part
        System.out.println(predict(new double[]{1, x1, x2, Math.pow(x1, 2), Math.pow(x2, 2)}) == 1 ? "Accepted." : "Rejected.");

        RegularizedLogisticRegression visualizationOfData = new RegularizedLogisticRegression("Visualization of data");
        visualizationOfData.pack();
        RefineryUtilities.centerFrameOnScreen(visualizationOfData);
        visualizationOfData.setResizable(false);
        visualizationOfData.setVisible(true);
    }

    //For plotting
    private RegularizedLogisticRegression(final String title) {

        super(title);

        // Create a single plot containing both the scatter and line
        XYPlot plot = new XYPlot();

        //SETUP SCATTER graph for admitted students
        // Create the scatter data, renderer, and axis
        XYSeries seriesEx1 = new XYSeries("Accepted");
        for (int i = 0; i < dataSet.size(); i++) {
            if (dataSet.get(i).yValue == 1) seriesEx1.add(dataSet.get(i).xVariables[1], dataSet.get(i).xVariables[2]);
        }
        XYDataset dataSetVisEx1 = getData(seriesEx1);
        XYItemRenderer rendererDataEx1 = new XYLineAndShapeRenderer(false, true);
        rendererDataEx1.setBasePaint(Color.GREEN);
        ValueAxis xAxisDataEx1 = new NumberAxis("Microchip test 1");
        ValueAxis yAxisDataEx1 = new NumberAxis("Microchip test 2");
        xAxisDataEx1.setLowerBound(-1.5);
        xAxisDataEx1.setUpperBound(1.5);
        yAxisDataEx1.setLowerBound(-1.5);
        yAxisDataEx1.setUpperBound(1.5);

        // Set the scatter data, renderer, and axis into plot
        plot.setDataset(0, dataSetVisEx1);
        plot.setRenderer(0, rendererDataEx1);
        plot.setDomainAxis(0, xAxisDataEx1);
        plot.setRangeAxis(0, yAxisDataEx1);

        //SETUP SCATTER graph for students not admitted
        // Create the scatter data, renderer, and axis
        XYSeries seriesEx2 = new XYSeries("Rejected");
        for (int i = 0; i < dataSet.size(); i++) {
            if (dataSet.get(i).yValue == 0) seriesEx2.add(dataSet.get(i).xVariables[1], dataSet.get(i).xVariables[2]);
        }
        XYDataset dataSetVisEx2 = getData(seriesEx2);
        XYItemRenderer rendererDataEx2 = new XYLineAndShapeRenderer(false, true);
        rendererDataEx2.setBasePaint(Color.ORANGE);
        ValueAxis xAxisDataEx2 = new NumberAxis();
        ValueAxis yAxisDataEx2 = new NumberAxis();
        xAxisDataEx2.setLowerBound(-1.5);
        xAxisDataEx2.setUpperBound(1.5);
        xAxisDataEx2.setAxisLineVisible(false);
        xAxisDataEx2.setVerticalTickLabels(false);
        yAxisDataEx2.setLowerBound(-1.5);
        yAxisDataEx2.setUpperBound(1.5);
        yAxisDataEx2.setAxisLineVisible(false);
        yAxisDataEx2.setVerticalTickLabels(false);

        // Set the scatter data, renderer, and axis into plot
        plot.setDataset(1, dataSetVisEx2);
        plot.setRenderer(1, rendererDataEx2);
        plot.setDomainAxis(1, xAxisDataEx2);
        plot.setRangeAxis(1, yAxisDataEx2);

        //SETUP LINE graph
        // Create the line data, renderer, and axis
        XYSeries lineSeries = new XYSeries("Decision boundary", false); //Set autoSort to false to not get lines all over the place
        double x = -3;
        while (x <= 3) {
            //      0 = t0 + t1 * x1 + t2 * x2 + t3 * x1^2 + t4 * x2^2, let x2 be y, solver for y
            //y = +- ...
            lineSeries.add(x, (-1*theta[1]  + Math.sqrt(Math.pow(theta[1], 2) - 4 * theta[3] *(theta[0] + theta[2] * x + theta [4] * Math.pow(x, 2))))/(2*theta[4]));
            //lineSeries.add(x, (-1*theta[1]  - Math.sqrt(Math.pow(theta[1], 2) - 4 * theta[3] *(theta[0] + theta[2] * x + theta [4] * Math.pow(x, 2))))/(2*theta[4]));
            //      0 = t0 + 1 * x1^2 + t2 * x2^2, let x2 be y, solver for y
            //y = +- ...
            // y  = -1*(Math.sqrt(-1*theta[0]- theta[1] * Math.pow(x, 2)))/Math.sqrt(theta[2]) , c!=0
            //lineSeries.add(x, -1*(Math.sqrt(-1*theta[0]- theta[1] * Math.pow(x, 2)))/Math.sqrt(theta[2]));
            x += 0.05;
        }
        double y = 3;
        while (y >= -3) {
            //      0 = t0 + t1 * x1 + t2 * x2 + t3 * x1^2 + t4 * x2^2, let x2 be y, solver for y
            //y = +- ...
            lineSeries.add(y, (-1*theta[1]  - Math.sqrt(Math.pow(theta[1], 2) - 4 * theta[3] *(theta[0] + theta[2] * y + theta [4] * Math.pow(y, 2))))/(2*theta[4]));
            //      0 = t0 + 1 * x1^2 + t2 * x2^2, let x2 be y, solver for y
            //y = +- ...
            // y  = Math.sqrt(-1*theta[0]- theta[1] * Math.pow(x, 2)))/Math.sqrt(theta[2], c!=0
            //lineSeries.add(y, Math.sqrt(-1*theta[0]- theta[1] * Math.pow(y, 2))/Math.sqrt(theta[2]));
            y -= 0.05;
        }
        XYDataset lineDataSet = getLineData(lineSeries);
        XYItemRenderer rendererLine = new XYLineAndShapeRenderer(true, false);   // Lines only
        ValueAxis xAxisLine = new NumberAxis();
        ValueAxis yAxisLine = new NumberAxis();
        xAxisLine.setLowerBound(-1.5);
        xAxisLine.setUpperBound(1.5);
        xAxisLine.setAxisLineVisible(false);
        xAxisLine.setVerticalTickLabels(false);
        yAxisLine.setLowerBound(-1.5);
        yAxisLine.setUpperBound(1.5);
        yAxisLine.setAxisLineVisible(false);
        yAxisLine.setVerticalTickLabels(false);

        // Set the line data, renderer, and axis into plot
        plot.setDataset(2, lineDataSet);
        plot.setRenderer(2, rendererLine);
        plot.setDomainAxis(2, xAxisLine);
        plot.setRangeAxis(2, yAxisLine);

        // Map the line to the second xAxis and second yAxis
        plot.mapDatasetToDomainAxis(1, 0);
        plot.mapDatasetToDomainAxis(1, 0);
        plot.mapDatasetToRangeAxis(2, 0);
        plot.mapDatasetToRangeAxis(2, 0);

        // Create the chart with the plot and a legend
        JFreeChart chart = new JFreeChart("Data set - Regularized Logistic Regression", JFreeChart.DEFAULT_TITLE_FONT, plot, true);

        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 800));
        setContentPane(chartPanel);
    }

    //Load the data from the txt file into an array
    private static void loadData() {
        try {
            FileReader reader = new FileReader(file);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String line;
            String[] columns;
            while ((line = bufferedReader.readLine()) != null) {
                columns = line.split(",");
                n = columns.length;
                double y = Double.parseDouble(columns[n - 1]);
                double xArray[] = new double[columns.length];
                xArray[0] = 1.0;
                for (int i = 0; i < columns.length - 1; i++) {
                    xArray[i + 1] = Double.parseDouble(columns[i]);
                }
                Instance instance = new Instance(y, xArray);
                dataSet.add(instance);
            }
            m = dataSet.size();
            bufferedReader.close();
            reader.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    //Initialize theta, play with initial theta values to get better fit
    private static void initTheta() {
        theta = new double[dataSetFeatureMapping.get(0).xVariables.length];
        theta[0] = -2;
        theta[1] = 1;
        theta[2] = 0.5;
        theta[3] = 4;
        theta[4] = 5;
        /*for (int i = 0; i < dataSetFeatureMapping.get(0).xVariables.length; i++) {
            theta[i] = 0.0;
        }*/
    }

    //Regularized cost function
    private static double createCostFunction(List<Instance> instances) {
        double cost = 0.0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double y = instance.yValue;
            cost += -y * Math.log(createHypothesis(x)) - (1 - y) * Math.log(1 - createHypothesis(x));
        }
        double sum = 0.0;
        for (int i = 0; i < instances.get(0).xVariables.length; i++) {
            sum += theta[i];
        }
        double J = (1 / m) * cost + (lambda / (2 * m)) * sum; //regularized
        //System.out.println("Cost function value with theta " + Arrays.toString(theta) + ": " + J);
        return J;
    }

    //Sigmoid function
    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    //Creating our hypothesis (h(x)) / Prediction
    // h(x) = P(y=1 | x; theta)  <- Probability that y = 1, given x parameterized by theta.
    private static double createHypothesis(double[] x) {
        double hypothesis = 0.0;
        for (int i = 0; i < theta.length; i++) {
            hypothesis += theta[i] * x[i];
        }
        return sigmoid(hypothesis);
    }

    //Feature mapping for more accurate data fitting
    private static void doFeatureMapping(List<Instance> instances) {
        for (Instance instance : instances) {
            double xArray[] = new double[5];
            xArray[0] = 1.0;
            xArray[1] = instance.xVariables[1];
            xArray[2] = instance.xVariables[2];
            xArray[3] = Math.pow(instance.xVariables[1] , 2);
            xArray[4] = Math.pow(instance.xVariables[2] , 2);
            double y = instance.yValue;
            Instance instanceFeatureMapping = new Instance(y, xArray);
            dataSetFeatureMapping.add(instanceFeatureMapping);
        }
    }

    //Create gradients
    private static void createGradients(List<Instance> instances) {
        double[] grad = new double[instances.get(0).xVariables.length];
        double sum[] = new double[instances.get(0).xVariables.length];
        for(int i = 0; i < instances.get(0).xVariables.length; i++) {
            sum[i] = 0.0;
        }
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double hypothesis = createHypothesis(x);
            double y = instance.yValue;
            for(int i = 0; i < instances.get(0).xVariables.length; i++) {
                sum[i] = hypothesis * x[i] - y * x[i];
            }
        }
        for(int i = 0; i < instances.get(0).xVariables.length; i++) {
            grad[i] = (1 / m) * sum[i];
        }
        System.out.println("Gradients with theta " + Arrays.toString(theta) + " : " + Arrays.toString(grad) + "");
    }

    //Do gradient descent (regularized)
    private static void doGradientDescent(List<Instance> instances) {
        for (int k = 0; k < 5000; k++) {
            double[] temp = new double[instances.get(0).xVariables.length];
            for (int i = 0; i < n; i++) {
                temp[i] = 0.0;
            }
            double costFunctionOld = createCostFunction(dataSetFeatureMapping);
            for (Instance instance : instances) {
                double[] x = instance.xVariables;
                double hypothesis = createHypothesis(x);
                double y = instance.yValue;
                for (int i = 0; i < n; i++) {
                    temp[i] += (hypothesis - y) * x[i];
                }
            }
            double alpha = 0.01;
            theta[0] = theta[0] - alpha * temp[0];
            for (int i = 1; i < instances.get(0).xVariables.length; i++) {
                theta[i] = theta[i]  -  alpha * (temp[i] + (lambda/m)*theta[i]);
            }
            iterations++;
            //Cost function to descend, theta after each iteration:
            //System.out.println("Iteration: " + iterations + " " + Arrays.toString(theta));
            //System.out.println(createCostFunction(dataSetFeatureMapping));
            if (costFunctionOld - createCostFunction(dataSetFeatureMapping) < 0.001) {
                System.out.println("Number of iterations: " + iterations);
                return;
            }
        }
        System.out.println("Number of iterations: " + iterations);
    }

    //Predict whether the label is 0 or 1 (not admitted or admitted) using
    //a threshold at 0.5
    private static double predict(double[] scores) {
        if (createHypothesis(scores) < 0.5) return 1;  //< 0.5 Because 1s(accepted) is inside circle
        return 0;
    }

    //Calculate accuracy of logistic regression
    private static double calcAccuracyOfModel(List<Instance> instances) {
        int counter = 0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double y = instance.yValue;
            if (predict(x) != y) counter++;
        }
        return (1 - counter / m) * 100;
    }

    //Get XYDataset data data for visualization
    private XYDataset getData(XYSeries series) {
        XYSeriesCollection xySeriesCollectionData = new XYSeriesCollection();
        xySeriesCollectionData.addSeries(series);
        return xySeriesCollectionData;
    }

    //Get XYDataset line data for visualization
    private XYDataset getLineData(XYSeries lineSeries) {
        XYSeriesCollection xySeriesCollectionLine = new XYSeriesCollection();
        xySeriesCollectionLine.addSeries(lineSeries);
        return xySeriesCollectionLine;
    }

    public static class Instance {
        double yValue;
        double[] xVariables;

        Instance(double yValue, double[] xVariables) {
            this.yValue = yValue;
            this.xVariables = xVariables;
        }
    }
}
