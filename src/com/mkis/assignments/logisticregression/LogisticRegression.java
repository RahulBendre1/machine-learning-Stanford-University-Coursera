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

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LogisticRegression extends ApplicationFrame {

    private static String file = "D:\\Projects-repos\\MachineLearning\\src\\com\\mkis\\assignments\\logisticregression\\data1.txt";
    private static double m; // number of training examples
    private static int n; // number of features

    private static double alpha = 0.0000001; // learning rate
    private static int iterations = 0;// number of iterations needed for gradient descent

    private static double theta[]; // parameters/weights array
    private static double grad[] = new double[3]; // gradients array
    private static List<Instance> dataSet = new ArrayList<>(); //list containing 1 row of training example

    //For plotting
    private LogisticRegression(final String title) {

        super(title);

        // Create a single plot containing both the scatter and line
        XYPlot plot = new XYPlot();

        //SETUP SCATTER graph
        // Create the scatter data, renderer, and axis
        XYSeries series = new XYSeries("Admittance depending on exam score");
        for (int i = 0; i < dataSet.size(); i++) {
            series.add(dataSet.get(i).xVariables[1], dataSet.get(i).xVariables[2]);
        }
        XYDataset dataSetVis = getData(series);
        XYItemRenderer rendererData = new XYLineAndShapeRenderer(false, true);
        ValueAxis xAxisData = new NumberAxis("Exam 1 score");
        ValueAxis yAxisData = new NumberAxis("Exam 2 score");
        xAxisData.setLowerBound(15);
        xAxisData.setUpperBound(115);
        yAxisData.setLowerBound(15);
        yAxisData.setUpperBound(115);

        // Set the scatter data, renderer, and axis into plot
        plot.setDataset(0, dataSetVis);
        plot.setRenderer(0, rendererData);
        plot.setDomainAxis(0, xAxisData);
        plot.setRangeAxis(0, yAxisData);

        //SETUP LINE graph
        // Create the line data, renderer, and axis
        XYSeries lineSeries = new XYSeries("Decision boundary");
        for (int i = 0; i < 1000; i++) {
            lineSeries.add((double) i, (theta[0] + theta[1] * i) / (-1 * theta[2]));
        }
        XYDataset lineDataSet = getLineData(lineSeries);
        XYItemRenderer rendererLine = new XYLineAndShapeRenderer(true, false);   // Lines only
        ValueAxis xAxisLine = new NumberAxis();
        ValueAxis yAxisLine = new NumberAxis();
        xAxisLine.setLowerBound(15);
        xAxisLine.setUpperBound(115);
        xAxisLine.setAxisLineVisible(true);
        yAxisLine.setLowerBound(15);
        yAxisLine.setUpperBound(115);
        yAxisLine.setAxisLineVisible(true);

        // Set the line data, renderer, and axis into plot
        plot.setDataset(1, lineDataSet);
        plot.setRenderer(1, rendererLine);
        plot.setDomainAxis(1, xAxisLine);
        plot.setRangeAxis(1, yAxisLine);

        // Map the line to the second xAxis and second yAxis
        plot.mapDatasetToDomainAxis(1, 1);
        plot.mapDatasetToRangeAxis(1, 1);

        // Create the chart with the plot and a legend
        JFreeChart chart = new JFreeChart("Data set - Logistic Regression", JFreeChart.DEFAULT_TITLE_FONT, plot, true);

        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 800));
        setContentPane(chartPanel);
    }

    public static void main(String[] args) {

        loadData();
        initTheta();
        System.out.println("Cost Function at theta: " + Arrays.toString(theta) + " : " + createCostFunction(dataSet));
        createGradients(dataSet);
        doGradientDescent(dataSet);
        //doLBFGS(dataSet);

        NumberFormat nf = new DecimalFormat("##.##");
        System.out.println("For a student with scores 45 and 85, we predict an admission probability: ");
        System.out.println(nf.format(100 * (createHypothesis(new double[]{1, 45, 85}))) + " %");
        System.out.println(predict(new double[]{1, 45, 85}) ? "He will be admitted." : "He will not be admitted.");

        LogisticRegression visualizationOfData = new LogisticRegression("Visualization of data");
        visualizationOfData.pack();
        RefineryUtilities.centerFrameOnScreen(visualizationOfData);
        visualizationOfData.setResizable(false);
        visualizationOfData.setVisible(true);
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

    //Initialize theta
    private static void initTheta() {
        theta = new double[n];
        theta[0] = -24;
        theta[1] = 0.2;
        theta[2] = 0.2;
        /*for (int i = 0; i < n; i++) {
            theta[i] = 0.0;
        }*/
    }

    /*h (x) = 1/(1+exp(-z))  - Sigmoid function for h (x)
        z(1) = theta0*x0(1) + theta1*x1(1) + theta2 * x2(1)
        Cost = -y(1)*log(z(1))-(1-y(1))*log(1-z(1))
        m = 1 in above example (first tr. example values)
         */
    private static double createCostFunction(List<Instance> instances) {
        double cost = 0.0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double y = instance.yValue;
            cost += -y * Math.log(createHypothesis(x)) - (1 - y) * Math.log(1 - createHypothesis(x));
        }
        double J = (1 / m) * cost;
        //System.out.println("Cost function value with theta " + Arrays.toString(theta) + ": " + J);
        return J;
    }

    //Create gradients
    private static void createGradients(List<Instance> instances) {
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (Instance instance : instances) {
            double[] x = instance.xVariables;
            double hypothesis = createHypothesis(x);
            double y = instance.yValue;
            sum0 += hypothesis * x[0] - y * x[0];
            sum1 += hypothesis * x[1] - y * x[1];
            sum2 += hypothesis * x[2] - y * x[2];
        }
        grad[0] = (1 / m) * sum0;
        grad[1] = (1 / m) * sum1;
        grad[2] = (1 / m) * sum2;
        System.out.println("Gradients with theta " + Arrays.toString(theta) + " : " + Arrays.toString(grad) + "");
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

    //Do gradient descent
    private static void doGradientDescent(List<Instance> instances) {
        for (int k = 0; k < 5000; k++) {
            double costFunctionOld = createCostFunction(dataSet);
            for (Instance instance : instances) {
                double[] x = instance.xVariables;
                double hypothesis = createHypothesis(x);
                double y = instance.yValue;
                //double error = y-hypothesis;
                theta[0] = theta[0] + alpha * (y - hypothesis) * hypothesis * (1 - hypothesis);
                for (int i = 1; i < n; i++) {
                    theta[i] = theta[i] + alpha * (y - hypothesis) * hypothesis * (1 - hypothesis) * x[i];
                }
                //System.out.println("Error: " + error);
            }
            iterations++;
            //Cost function to descend, theta after each iteration:
            //System.out.println("Iteration: " + iterations + " " + Arrays.toString(theta));
            if (costFunctionOld - createCostFunction(dataSet) < 0.0000001) return;
        }
    }

    //Predict whether the label is 0 or 1 using
    //a threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    private static boolean predict(double[] scores) {
        return createHypothesis(scores) >= 0.5;
    }

}
